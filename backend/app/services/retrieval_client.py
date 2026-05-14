from __future__ import annotations

import json
import time
from typing import Any

import httpx
from langchain_core.documents import Document

from ..core.settings import settings
from .rag.logging import _emit_query_progress


def enabled() -> bool:
    return bool(settings.retrieval_service_url)


def _require_base_url() -> str:
    service_url = settings.retrieval_service_url.strip()
    if not service_url:
        raise RuntimeError(
            "RETRIEVAL_SERVICE_URL is not configured. Backend requires retrieval_service for search/index/delete."
        )
    return service_url


def _headers() -> dict[str, str]:
    if not settings.ollama_api_key:
        return {}
    return {"x-api-key": settings.ollama_api_key}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    base_url = _require_base_url()
    timeout = httpx.Timeout(settings.retrieval_timeout_seconds, connect=10.0)
    with httpx.Client(timeout=timeout, headers=_headers()) as client:
        response = client.request(method, f"{base_url}{path}", **kwargs)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()
        raise RuntimeError(
            f"Retrieval Service request failed: {response.status_code} {response.reason_phrase}"
            f" for {path}. {detail}"
        ) from exc
    return response


def health_ready() -> bool:
    response = _request("GET", "/health")
    payload = response.json()
    return bool(payload.get("collection_ready"))


def delete_vectors_by_document_id(document_id: int) -> None:
    _request("DELETE", f"/v1/index/document/{document_id}")


def upsert_child_documents(
    documents: list[Document],
    *,
    purge_document_ids: list[int] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
) -> int:
    chunks: list[dict[str, Any]] = []
    for index, item in enumerate(documents):
        metadata = dict(item.metadata or {})
        chunk_id = metadata.get("parent_chunk_id") or metadata.get("chunk_id") or index
        document_id = metadata.get("document_id")
        if document_id is None:
            raise RuntimeError("document_id is required to index chunks through Retrieval Service.")

        chunk_payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": item.page_content,
            "page": metadata.get("source_page"),
            "metadata": _json_safe(metadata),
        }
        if precomputed_vectors is not None:
            chunk_payload["vector"] = precomputed_vectors[index]
        chunks.append(chunk_payload)

    response = _request(
        "POST",
        "/v1/index/chunks",
        json={
            "chunks": chunks,
            "purge_document_ids": purge_document_ids or [],
        },
    )
    payload = response.json()
    return int(payload.get("indexed_chunks") or 0)


def similarity_search(
    query: str,
    *,
    top_k: int | None,
    document_ids: list[int] | None = None,
    plan: Any | None = None,
) -> list[Document]:
    payload: dict[str, Any] = {
        "query": query,
        "filters": {
            "document_ids": document_ids or [],
            "metadata": {"index_type": "section_parent_child"},
        },
    }
    if top_k is not None:
        payload["top_k"] = top_k
    started_at = time.perf_counter()
    response = _request(
        "POST",
        "/v1/search/hybrid",
        json=payload,
    )
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    payload_response = response.json()
    debug = payload_response.get("debug") if isinstance(payload_response.get("debug"), dict) else {}
    contexts = payload_response.get("contexts")
    if not isinstance(contexts, list):
        _emit_retrieval_service_log(
            query=query,
            top_k=top_k,
            elapsed_ms=elapsed_ms,
            documents=[],
            raw_context_count=0,
            debug=debug,
        )
        return []

    documents: list[Document] = []
    for item in contexts:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_metadata = metadata.get("source_metadata")
        if isinstance(source_metadata, str):
            try:
                parsed = json.loads(source_metadata)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                source_metadata = parsed

        metadata = dict(metadata)
        metadata.update(
            {
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "parent_chunk_id": item.get("chunk_id"),
                "source_page": item.get("page"),
                "source_metadata": source_metadata if isinstance(source_metadata, dict) else metadata.get("source_metadata"),
                "retrieval_mode": metadata.get("retrieval_mode") or "retrieval_service",
                "retrieval_score": item.get("score"),
            }
        )
        documents.append(
            Document(
                page_content=str(item.get("content") or ""),
                metadata=metadata,
            )
        )
    _emit_retrieval_service_log(
        query=query,
        top_k=top_k,
        elapsed_ms=elapsed_ms,
        documents=documents,
        raw_context_count=len(contexts),
        debug=debug,
    )
    return documents


def _emit_retrieval_service_log(
    *,
    query: str,
    top_k: int | None,
    elapsed_ms: float,
    documents: list[Document],
    raw_context_count: int,
    debug: dict[str, Any],
) -> None:
    final_chunks: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = {}
    for rank, doc in enumerate(documents, start=1):
        metadata = dict(doc.metadata or {})
        mode = str(metadata.get("retrieval_mode") or "retrieval_service")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        final_chunks.append(
            {
                "rank": rank,
                "chunk_id": metadata.get("chunk_id") or metadata.get("parent_chunk_id"),
                "document_id": metadata.get("document_id"),
                "retrieval_mode": mode,
                "retrieval_score": metadata.get("retrieval_score"),
                "reranker_score": metadata.get("reranker_score"),
                "source_page": metadata.get("source_page"),
                "source_metadata": metadata.get("source_metadata"),
                "content": doc.page_content[:280],
            }
        )

    _emit_query_progress(
        "[retrieval_client] retrieval_service hybrid returned contexts=%d docs=%d",
        raw_context_count,
        len(documents),
        event="retrieval_service_done",
        details={
            "query_preview": query[:120],
            "top_k": top_k,
            "elapsed_ms": elapsed_ms,
            "raw_context_count": raw_context_count,
            "document_count": len(documents),
            "mode_counts": debug.get("mode_counts") if isinstance(debug.get("mode_counts"), dict) else mode_counts,
            "candidate_pool": debug.get("candidate_pool"),
            "vector_weight": debug.get("vector_weight"),
            "keyword_weight": debug.get("keyword_weight"),
            "rrf_k": debug.get("rrf_k"),
            "reranker_model": (debug.get("reranker") or {}).get("model")
            if isinstance(debug.get("reranker"), dict) else None,
            "reranker_status": (debug.get("reranker") or {}).get("status")
            if isinstance(debug.get("reranker"), dict) else None,
        },
    )
    _emit_retrieval_debug_events(debug=debug, elapsed_ms=elapsed_ms)
    _emit_query_progress(
        "[retrieval_client] final retrieved docs: count=%d",
        len(documents),
        event="similarity_search_done",
        details={"final_chunks": final_chunks},
    )
    _emit_query_progress(
        "[retrieval_client] retrieval service timing elapsed_ms=%.2f",
        elapsed_ms,
        event="similarity_search_timing",
        details={
            "stage_timings_ms": _stage_timings_from_debug(debug, elapsed_ms),
        },
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _debug_list(debug: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = debug.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _stage_timings_from_debug(debug: dict[str, Any], elapsed_ms: float) -> dict[str, float]:
    timings = debug.get("timings_ms") if isinstance(debug.get("timings_ms"), dict) else {}
    return {
        "retrieval_service": elapsed_ms,
        "query_embedding": _safe_float(timings.get("embed_query")),
        "semantic_candidates": _safe_float(timings.get("semantic_candidates")),
        "keyword_candidates": _safe_float(timings.get("keyword_candidates")),
        "rrf_merge": _safe_float(timings.get("rrf_merge")),
        "reranker": _safe_float(timings.get("reranker")),
    }


def _emit_retrieval_debug_events(*, debug: dict[str, Any], elapsed_ms: float) -> None:
    vector_candidates = _debug_list(debug, "vector_candidates")
    keyword_candidates = _debug_list(debug, "keyword_candidates")
    merged_candidates = _debug_list(debug, "merged_candidates")
    score_preview = _debug_list(debug, "rrf_score_preview")

    if vector_candidates:
        _emit_query_progress(
            "[retrieval_client] vector candidates: count=%d",
            len(vector_candidates),
            event="semantic_candidates_done",
            details={
                "semantic_parent_ids": [item.get("chunk_id") for item in vector_candidates],
                "semantic_selected_chunks": vector_candidates,
                "semantic_child_type_preview": {},
            },
        )

    if keyword_candidates:
        _emit_query_progress(
            "[retrieval_client] keyword candidates: count=%d",
            len(keyword_candidates),
            event="keyword_candidates_done",
            details={
                "keyword_selected_parent_ids": [item.get("chunk_id") for item in keyword_candidates],
                "keyword_selected_chunks": keyword_candidates,
            },
        )

    if merged_candidates or score_preview:
        _emit_query_progress(
            "[retrieval_client] rrf merged candidates: count=%d",
            len(merged_candidates),
            event="rrf_merge_done",
            details={
                "semantic_parent_ids": [item.get("chunk_id") for item in vector_candidates],
                "keyword_parent_ids": [item.get("chunk_id") for item in keyword_candidates],
                "rrf_merged_parent_ids": [item.get("chunk_id") for item in merged_candidates],
                "rrf_score_preview": score_preview or [
                    {"chunk_id": item.get("chunk_id"), "score": item.get("rrf_score") or item.get("score")}
                    for item in merged_candidates
                ],
                "retrieval_service_elapsed_ms": elapsed_ms,
            },
        )

    reranker = debug.get("reranker") if isinstance(debug.get("reranker"), dict) else {}
    if reranker:
        _emit_query_progress(
            "[retrieval_client] reranker status=%s input=%s output=%s",
            reranker.get("status", "?"),
            reranker.get("input_count", 0),
            reranker.get("output_count", 0),
            event="rerank_documents",
            details={
                "model": reranker.get("model"),
                "status": reranker.get("status"),
                "input_count": reranker.get("input_count", 0),
                "output_count": reranker.get("output_count", 0),
                "original_top_score": reranker.get("original_top_score"),
                "reranked_top_score": reranker.get("reranked_top_score"),
                "score_preview": reranker.get("score_preview", []),
                "error": reranker.get("error"),
            },
        )
