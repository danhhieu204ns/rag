from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ..core.settings import settings
from ..schemas import ChunkIndexItem, ContextItem, RetrievalFilters
from .chunk_store import load_chunks_by_ids

_client: QdrantClient | None = None


def collection_name(override: str | None = None) -> str:
    return (override or settings.qdrant_collection_name).strip() or settings.qdrant_collection_name


def get_qdrant_client() -> QdrantClient:
    global _client

    if _client is not None:
        return _client

    if settings.qdrant_url:
        kwargs: dict[str, str] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    else:
        _client = QdrantClient(path=str(settings.qdrant_path))
    return _client


def collection_exists(name: str) -> bool:
    client = get_qdrant_client()
    return any(item.name == name for item in client.get_collections().collections)


def ensure_collection(name: str, vector_size: int) -> None:
    if collection_exists(name):
        return
    get_qdrant_client().create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _point_id(name: str, document_id: int | str, chunk_id: int | str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{name}|document:{document_id}|chunk:{chunk_id}"))


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_filter(filters: RetrievalFilters | None) -> qdrant_models.Filter | None:
    if filters is None:
        return None

    must: list[qdrant_models.Condition] = []
    if filters.document_ids:
        must.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchAny(any=[int(item) for item in filters.document_ids]),
            )
        )

    for key, value in filters.metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            must.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchAny(any=value),
                )
            )
        else:
            must.append(
                qdrant_models.FieldCondition(
                    key=key,
                    match=qdrant_models.MatchValue(value=value),
                )
            )

    if not must:
        return None
    return qdrant_models.Filter(must=must)


def delete_document_vectors(document_id: int, name: str) -> bool:
    if not collection_exists(name):
        return False

    document_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchValue(value=document_id),
            )
        ]
    )
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=name,
            points_selector=document_filter,
            wait=True,
        )
    except TypeError:
        client.delete(
            collection_name=name,
            points_selector=qdrant_models.FilterSelector(filter=document_filter),
            wait=True,
        )
    return True


def upsert_chunks(chunks: list[ChunkIndexItem], vectors: list[list[float]], name: str) -> int:
    if len(chunks) != len(vectors):
        raise HTTPException(status_code=422, detail="chunks and vectors length mismatch.")
    if not chunks:
        return 0

    ensure_collection(name, len(vectors[0]))
    client = get_qdrant_client()
    indexed = 0

    for start in range(0, len(chunks), settings.vector_batch_size):
        end = min(start + settings.vector_batch_size, len(chunks))
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks[start:end], vectors[start:end]):
            source_metadata = chunk.metadata.get("source_metadata")
            if not isinstance(source_metadata, dict):
                source_metadata = {}

            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "parent_chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "child_text": chunk.content,
                "source": chunk.source,
                "source_page": chunk.page,
                "source_metadata": source_metadata,
            }
            points.append(
                PointStruct(
                    id=_point_id(name, chunk.document_id, chunk.chunk_id),
                    vector=vector,
                    payload=_json_safe(payload),
                )
            )

        client.upsert(collection_name=name, points=points, wait=False)
        indexed += len(points)

    return indexed


def search_contexts(
    *,
    vector: list[float],
    name: str,
    top_k: int,
    filters: RetrievalFilters | None,
) -> list[ContextItem]:
    if not collection_exists(name):
        return []

    client = get_qdrant_client()
    query_filter = _payload_filter(filters)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        points = list(getattr(response, "points", []) or [])
    else:
        points = list(
            client.search(
                collection_name=name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        )

    parent_ids: list[int] = []
    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        parent_id = _to_int(payload.get("parent_chunk_id")) or _to_int(payload.get("chunk_id"))
        if parent_id is not None and parent_id not in parent_ids:
            parent_ids.append(parent_id)

    stored_chunks = load_chunks_by_ids(parent_ids)

    contexts: list[ContextItem] = []
    seen_parent_ids: set[int] = set()
    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        parent_id = _to_int(payload.get("parent_chunk_id")) or _to_int(payload.get("chunk_id"))
        if parent_id is not None and parent_id in seen_parent_ids:
            continue
        if parent_id is not None:
            seen_parent_ids.add(parent_id)

        stored = stored_chunks.get(parent_id or -1)
        source_metadata = payload.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}

        content = str(payload.get("child_text") or "")
        document_id = _to_int(payload.get("document_id"))
        source_page = _to_int(payload.get("source_page"))
        if stored is not None:
            content = stored.content
            document_id = stored.document_id
            source_page = stored.source_page
            if stored.source_metadata:
                source_metadata = stored.source_metadata

        contexts.append(
            ContextItem(
                chunk_id=parent_id,
                document_id=document_id,
                content=content,
                source=str(payload.get("source") or source_metadata.get("source") or "") or None,
                page=source_page,
                score=_to_float(getattr(point, "score", None)),
                metadata={
                    key: value
                    for key, value in payload.items()
                    if key != "child_text"
                },
            )
        )
        if len(contexts) >= top_k:
            break

    return contexts
