from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import time
import contextlib
from datetime import datetime
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Response

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_SERVICE_ROOT / ".env", override=True)

from .core.settings import settings
from .schemas import (
    DeleteDocumentResponse,
    IndexChunksRequest,
    IndexChunksResponse,
    RetrieveRequest,
    RetrieveResponse,
    VectorSearchRequest,
)
from .services.ollama_client import embed_texts
from .services.qdrant_store import (
    collection_exists,
    collection_name,
    delete_document_vectors,
    search_hybrid_contexts,
    search_contexts,
    upsert_chunks,
)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Standalone Retrieval Service for embedding-backed Qdrant search.",
)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


@contextlib.contextmanager
def _timed_step(name: str, logger: logging.Logger, **context):
    params = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info("[retrieval][timing] step=%s status=start %s", name, params)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = _ms(start)
        logger.exception("[retrieval][timing] step=%s status=error elapsed_ms=%.2f %s", name, elapsed, params)
        raise
    else:
        elapsed = _ms(start)
        logger.info("[retrieval][timing] step=%s status=ok elapsed_ms=%.2f %s", name, elapsed, params)


def _configure_retrieval_file_logging() -> logging.Logger:
    root = logging.getLogger("app.main")
    root.setLevel(logging.INFO)
    log_dir = Path(__file__).resolve().parents[1] / "storage" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / ("retrieval_service_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".log")
    from logging.handlers import RotatingFileHandler

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(str(log_path), maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path) for h in root.handlers):
        root.addHandler(fh)
    return root


# configure logging early
logger = _configure_retrieval_file_logging()
logger = _configure_retrieval_file_logging()


def _qdrant_ready() -> dict[str, Any]:
    try:
        active_collection = collection_name()
        return {
            "ready": True,
            "collection": active_collection,
            "collection_exists": collection_exists(active_collection),
            "url": settings.qdrant_url or str(settings.qdrant_path),
        }
    except Exception as exc:
        return {
            "ready": False,
            "url": settings.qdrant_url or str(settings.qdrant_path),
            "error": str(exc),
        }


@app.get("/health")
def health() -> dict[str, Any]:
    request_start = time.perf_counter()
    active_collection = collection_name()
    qdrant_error: str | None = None
    try:
        collection_ready = collection_exists(active_collection)
        status = "ok"
    except Exception as exc:  # pragma: no cover - depends on external Qdrant state
        collection_ready = False
        status = "degraded"
        qdrant_error = str(exc)

    logger.info("[retrieval][health] request received collection=%s", active_collection)
    elapsed = _ms(request_start)
    logger.info("[retrieval][health] response sent elapsed_ms=%.2f collection_ready=%s", elapsed, collection_ready)
    return {
        "status": status,
        "service": "retrieval-service",
        "collection": active_collection,
        "collection_ready": collection_ready,
        "qdrant": settings.qdrant_url or str(settings.qdrant_path),
        "qdrant_error": qdrant_error,
        "ollama_service_url": settings.ollama_service_url,
    }


def _ollama_ready() -> dict[str, Any]:
    headers = {"x-api-key": settings.ollama_api_key} if settings.ollama_api_key else {}
    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            response = client.get(f"{settings.ollama_service_url}/ready")
            if response.status_code == 404:
                response = client.get(f"{settings.ollama_service_url}/health")
        return {
            "ready": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": settings.ollama_service_url,
            "error": response.text if response.status_code >= 400 else None,
        }
    except Exception as exc:
        return {
            "ready": False,
            "url": settings.ollama_service_url,
            "error": str(exc),
        }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    request_start = time.perf_counter()
    qdrant = _qdrant_ready()
    ollama = _ollama_ready()
    status = "ok" if qdrant["ready"] and ollama["ready"] else "degraded"
    if status != "ok":
        response.status_code = 503
    logger.info("[retrieval][ready] request received qdrant_ready=%s ollama_ready=%s", qdrant["ready"], ollama["ready"])
    elapsed = _ms(request_start)
    logger.info("[retrieval][ready] response sent elapsed_ms=%.2f status=%s", elapsed, status)
    return {
        "status": status,
        "service": "retrieval-service",
        "checks": {
            "qdrant": qdrant,
            "ollama_service": ollama,
        },
    }


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    request_start = time.perf_counter()
    name = collection_name(request.collection)
    logger.info("[retrieval][retrieve] request received collection=%s query_len=%d top_k=%s", name, len(request.query or ""), request.top_k)
    # embed
    embed_start = time.perf_counter()
    vector = (await embed_texts([request.query]))[0]
    logger.info("[retrieval][retrieve] step=embed_texts elapsed_ms=%.2f collection=%s", _ms(embed_start), name)
    # search
    search_start = time.perf_counter()
    contexts = search_contexts(
        vector=vector,
        name=name,
        top_k=request.top_k or settings.default_top_k,
        filters=request.filters,
    )
    logger.info("[retrieval][retrieve] step=search_contexts elapsed_ms=%.2f collection=%s returned=%d", _ms(search_start), name, len(contexts))
    total = _ms(request_start)
    logger.info("[retrieval][timing] route=/v1/retrieve status=ok total_elapsed_ms=%.2f collection=%s returned=%d", total, name, len(contexts))
    return RetrieveResponse(contexts=contexts)


@app.post("/v1/search/vector", response_model=RetrieveResponse)
def search_vector(request: VectorSearchRequest) -> RetrieveResponse:
    request_start = time.perf_counter()
    name = collection_name(request.collection)
    logger.info("[retrieval][search_vector] request received collection=%s top_k=%s", name, request.top_k)
    search_start = time.perf_counter()
    contexts = search_contexts(
        vector=request.vector,
        name=name,
        top_k=request.top_k,
        filters=request.filters,
    )
    logger.info("[retrieval][search_vector] step=search_contexts elapsed_ms=%.2f collection=%s returned=%d", _ms(search_start), name, len(contexts))
    logger.info("[retrieval][timing] route=/v1/search/vector status=ok total_elapsed_ms=%.2f collection=%s returned=%d", _ms(request_start), name, len(contexts))
    return RetrieveResponse(contexts=contexts)


@app.post("/v1/search/hybrid", response_model=RetrieveResponse)
async def search_hybrid(request: RetrieveRequest) -> RetrieveResponse:
    request_start = time.perf_counter()
    name = collection_name(request.collection)
    logger.info("[retrieval][search_hybrid] request received collection=%s query_len=%d top_k=%s", name, len(request.query or ""), request.top_k)
    embed_start = time.perf_counter()
    vector = (await embed_texts([request.query]))[0]
    logger.info("[retrieval][search_hybrid] step=embed_texts elapsed_ms=%.2f collection=%s", _ms(embed_start), name)
    search_start = time.perf_counter()
    contexts = search_hybrid_contexts(
        query=request.query,
        vector=vector,
        name=name,
        top_k=request.top_k or settings.default_top_k,
        filters=request.filters,
    )
    logger.info("[retrieval][search_hybrid] step=search_hybrid_contexts elapsed_ms=%.2f collection=%s returned=%d", _ms(search_start), name, len(contexts))
    logger.info("[retrieval][timing] route=/v1/search/hybrid status=ok total_elapsed_ms=%.2f collection=%s returned=%d", _ms(request_start), name, len(contexts))
    return RetrieveResponse(contexts=contexts)


@app.post("/v1/index/chunks", response_model=IndexChunksResponse)
async def index_chunks(request: IndexChunksRequest) -> IndexChunksResponse:
    request_start = time.perf_counter()
    name = collection_name(request.collection)
    logger.info("[retrieval][index_chunks] request received collection=%s chunks=%d purge=%s", name, len(request.chunks), bool(request.purge_document_ids))

    if request.purge_document_ids:
        for document_id in request.purge_document_ids:
            logger.info("[retrieval][index_chunks] step=delete_document_vectors document_id=%s collection=%s", document_id, name)
            delete_document_vectors(document_id, name)

    vectors_by_index: dict[int, list[float]] = {}
    texts_to_embed: list[str] = []
    text_indexes: list[int] = []
    for index, chunk in enumerate(request.chunks):
        if chunk.vector is not None:
            vectors_by_index[index] = chunk.vector
            continue
        texts_to_embed.append(chunk.content)
        text_indexes.append(index)

    if texts_to_embed:
        embed_start = time.perf_counter()
        embedded = await embed_texts(texts_to_embed)
        logger.info("[retrieval][index_chunks] step=embed_texts elapsed_ms=%.2f collection=%s to_embed=%d", _ms(embed_start), name, len(texts_to_embed))
        for index, vector in zip(text_indexes, embedded):
            vectors_by_index[index] = vector

    ordered_vectors = [vectors_by_index[index] for index in range(len(request.chunks))]
    upsert_start = time.perf_counter()
    indexed_count = upsert_chunks(request.chunks, ordered_vectors, name)
    logger.info("[retrieval][index_chunks] step=upsert_chunks elapsed_ms=%.2f collection=%s indexed=%d", _ms(upsert_start), name, indexed_count)
    logger.info("[retrieval][timing] route=/v1/index/chunks status=ok total_elapsed_ms=%.2f collection=%s indexed=%d", _ms(request_start), name, indexed_count)
    return IndexChunksResponse(indexed_chunks=indexed_count, collection=name)


@app.delete("/v1/index/document/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: int, collection: str | None = None) -> DeleteDocumentResponse:
    request_start = time.perf_counter()
    name = collection_name(collection)
    logger.info("[retrieval][delete_document] request received document_id=%s collection=%s", document_id, name)
    deleted = delete_document_vectors(document_id, name)
    logger.info("[retrieval][delete_document] deleted=%s document_id=%s collection=%s elapsed_ms=%.2f", deleted, document_id, name, _ms(request_start))
    return DeleteDocumentResponse(document_id=document_id, deleted=deleted, collection=name)
