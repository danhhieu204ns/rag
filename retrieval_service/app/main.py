from __future__ import annotations

from pathlib import Path
from typing import Any

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
    search_contexts,
    upsert_chunks,
)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Standalone Retrieval Service for embedding-backed Qdrant search.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    active_collection = collection_name()
    qdrant_error: str | None = None
    try:
        collection_ready = collection_exists(active_collection)
        status = "ok"
    except Exception as exc:  # pragma: no cover - depends on external Qdrant state
        collection_ready = False
        status = "degraded"
        qdrant_error = str(exc)

    return {
        "status": status,
        "service": "retrieval-service",
        "collection": active_collection,
        "collection_ready": collection_ready,
        "qdrant": settings.qdrant_url or str(settings.qdrant_path),
        "qdrant_error": qdrant_error,
        "ollama_service_url": settings.ollama_service_url,
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    qdrant = _qdrant_ready()
    ollama = _ollama_ready()
    status = "ok" if qdrant["ready"] and ollama["ready"] else "degraded"
    if status != "ok":
        response.status_code = 503
    return {
        "status": status,
        "service": "retrieval-service",
        "checks": {
            "qdrant": qdrant,
            "ollama_service": ollama,
        },
    }


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


def _ollama_ready() -> dict[str, Any]:
    headers = {"x-api-key": settings.ollama_api_key} if settings.ollama_api_key else {}
    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            response = client.get(f"{settings.ollama_service_url}/ready")
            if response.status_code == 404:
                response = client.get(f"{settings.ollama_service_url}/health")
        return {
            "ready": response.status_code < 500,
            "status_code": response.status_code,
            "url": settings.ollama_service_url,
        }
    except Exception as exc:
        return {
            "ready": False,
            "url": settings.ollama_service_url,
            "error": str(exc),
        }


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    name = collection_name(request.collection)
    vector = (await embed_texts([request.query]))[0]
    return RetrieveResponse(
        contexts=search_contexts(
            vector=vector,
            name=name,
            top_k=request.top_k or settings.default_top_k,
            filters=request.filters,
        )
    )


@app.post("/v1/search/vector", response_model=RetrieveResponse)
def search_vector(request: VectorSearchRequest) -> RetrieveResponse:
    name = collection_name(request.collection)
    return RetrieveResponse(
        contexts=search_contexts(
            vector=request.vector,
            name=name,
            top_k=request.top_k,
            filters=request.filters,
        )
    )


@app.post("/v1/search/hybrid", response_model=RetrieveResponse)
async def search_hybrid(request: RetrieveRequest) -> RetrieveResponse:
    # First version uses vector retrieval. Keep this route stable so BM25/rerank
    # can be added behind the same contract later.
    return await retrieve(request)


@app.post("/v1/index/chunks", response_model=IndexChunksResponse)
async def index_chunks(request: IndexChunksRequest) -> IndexChunksResponse:
    name = collection_name(request.collection)

    if request.purge_document_ids:
        for document_id in request.purge_document_ids:
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
        embedded = await embed_texts(texts_to_embed)
        for index, vector in zip(text_indexes, embedded):
            vectors_by_index[index] = vector

    ordered_vectors = [vectors_by_index[index] for index in range(len(request.chunks))]
    indexed_count = upsert_chunks(request.chunks, ordered_vectors, name)
    return IndexChunksResponse(indexed_chunks=indexed_count, collection=name)


@app.delete("/v1/index/document/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: int, collection: str | None = None) -> DeleteDocumentResponse:
    name = collection_name(collection)
    deleted = delete_document_vectors(document_id, name)
    return DeleteDocumentResponse(document_id=document_id, deleted=deleted, collection=name)
