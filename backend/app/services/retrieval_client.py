from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.documents import Document

from ..core.settings import settings


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
    top_k: int,
    document_ids: list[int] | None = None,
    plan: Any | None = None,
) -> list[Document]:
    response = _request(
        "POST",
        "/v1/search/hybrid",
        json={
            "query": query,
            "top_k": top_k,
            "filters": {
                "document_ids": document_ids or [],
                "metadata": {"index_type": "section_parent_child"},
            },
        },
    )
    payload = response.json()
    contexts = payload.get("contexts")
    if not isinstance(contexts, list):
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
    return documents
