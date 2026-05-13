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
from .chunk_store import StoredChunk, load_chunks_by_ids, search_keyword_candidates
from .parent_retrieval import ChildHit, ParentRecord, expand_child_hits_to_parent_contexts

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


def _point_id(
    name: str,
    document_id: int | str,
    chunk_id: int | str,
    child_type: int | str,
    child_index: int | str,
) -> str:
    raw = (
        f"{name}|document:{document_id}|chunk:{chunk_id}|"
        f"type:{child_type}|index:{child_index}"
    )
    return str(uuid5(NAMESPACE_URL, raw))


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
    must: list[qdrant_models.Condition] = []
    metadata_filters = filters.metadata if filters is not None else {}
    if "index_type" not in metadata_filters:
        must.append(
            qdrant_models.FieldCondition(
                key="index_type",
                match=qdrant_models.MatchValue(value="section_parent_child"),
            )
        )

    if filters is None:
        return qdrant_models.Filter(must=must)

    if filters.document_ids:
        must.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchAny(any=[int(item) for item in filters.document_ids]),
            )
        )

    for key, value in metadata_filters.items():
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

            parent_id = (
                chunk.metadata.get("parent_chunk_id")
                or chunk.metadata.get("parent_id")
                or source_metadata.get("parent_chunk_id")
                or source_metadata.get("parent_id")
            )
            child_text = str(chunk.metadata.get("child_text") or chunk.content)
            embedding_text = str(chunk.metadata.get("embedding_text") or chunk.content)
            page_start = chunk.metadata.get("page_start") or chunk.metadata.get("source_page") or chunk.page
            page_end = chunk.metadata.get("page_end")
            payload = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "child_chunk_id": chunk.chunk_id,
                "parent_id": parent_id,
                "parent_chunk_id": parent_id,
                "document_id": chunk.document_id,
                "child_text": child_text,
                "embedding_text": embedding_text,
                "source": chunk.source,
                "source_page": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "source_metadata": source_metadata,
            }
            child_type = payload.get("child_type") or "section_child"
            child_index = _to_int(payload.get("child_index")) or 0
            points.append(
                PointStruct(
                    id=_point_id(name, chunk.document_id, chunk.chunk_id, child_type, child_index),
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
    child_limit = max(top_k, settings.top_k_children) if settings.search_child_chunks else top_k

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=name,
            query=vector,
            query_filter=query_filter,
            limit=child_limit,
            with_payload=True,
        )
        points = list(getattr(response, "points", []) or [])
    else:
        points = list(
            client.search(
                collection_name=name,
                query_vector=vector,
                query_filter=query_filter,
                limit=child_limit,
                with_payload=True,
            )
        )

    child_hits: list[ChildHit] = []
    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        parent_id = _to_int(payload.get("parent_chunk_id")) or _to_int(payload.get("parent_id"))
        child_chunk_id = str(payload.get("child_chunk_id") or payload.get("chunk_id") or "")
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"child_text", "embedding_text"}
        }
        child_hits.append(
            ChildHit(
                child_chunk_id=child_chunk_id,
                parent_id=parent_id,
                document_id=_to_int(payload.get("document_id")),
                child_text=str(payload.get("child_text") or ""),
                score=_to_float(getattr(point, "score", None)),
                metadata=metadata,
            )
        )

    if not settings.expand_to_parent:
        return [
            ContextItem(
                chunk_id=hit.parent_id,
                document_id=hit.document_id,
                content=hit.child_text,
                source=str(hit.metadata.get("source") or "") or None,
                page=_to_int(hit.metadata.get("page_start") or hit.metadata.get("source_page")),
                score=hit.score,
                metadata=hit.metadata,
            )
            for hit in child_hits[:top_k]
        ]

    parent_ids = [hit.parent_id for hit in child_hits if hit.parent_id is not None]
    stored_chunks = load_chunks_by_ids(parent_ids)
    parent_records = {
        parent_id: ParentRecord(
            parent_id=stored.chunk_id,
            document_id=stored.document_id,
            text=stored.content,
            page=stored.source_page,
            metadata=stored.source_metadata,
        )
        for parent_id, stored in stored_chunks.items()
    }

    parent_contexts = expand_child_hits_to_parent_contexts(
        child_hits,
        parent_records,
        final_top_k=top_k,
        deduplicate=settings.deduplicate_parents,
    )

    contexts: list[ContextItem] = []
    for item in parent_contexts:
        source_metadata = item.metadata.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        if not source_metadata and item.parent_id is not None:
            stored = stored_chunks.get(item.parent_id)
            if stored is not None:
                source_metadata = stored.source_metadata

        contexts.append(
            ContextItem(
                chunk_id=item.parent_id,
                document_id=item.document_id,
                content=item.text,
                source=str(item.metadata.get("source") or source_metadata.get("source") or "") or None,
                page=item.page or _to_int(item.metadata.get("page_start") or item.metadata.get("source_page")),
                score=item.score,
                metadata=item.metadata,
            )
        )

    return contexts


def _source_from_metadata(metadata: dict[str, Any]) -> str | None:
    source_info = metadata.get("source_info")
    if isinstance(source_info, dict):
        source = str(source_info.get("file_name") or "").strip()
        if source:
            return source
    source = str(metadata.get("source") or "").strip()
    return source or None


def _context_from_stored_chunk(
    chunk: StoredChunk,
    *,
    score: float | None,
    retrieval_mode: str,
) -> ContextItem:
    metadata = dict(chunk.source_metadata or {})
    metadata["retrieval_mode"] = retrieval_mode
    if score is not None:
        metadata["retrieval_score"] = score

    return ContextItem(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        content=chunk.content,
        source=_source_from_metadata(metadata),
        page=chunk.source_page,
        score=score,
        metadata=metadata,
    )


def _merge_hybrid_ids(
    *,
    vector_ids: list[int],
    keyword_ids: list[int],
    top_k: int,
) -> tuple[list[int], dict[int, float]]:
    rrf_k = 60.0
    scores: dict[int, float] = {}

    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:top_k]], scores


def search_hybrid_contexts(
    *,
    query: str,
    vector: list[float],
    name: str,
    top_k: int,
    filters: RetrievalFilters | None,
) -> list[ContextItem]:
    probe_k = max(top_k * 4, top_k)

    vector_contexts = search_contexts(
        vector=vector,
        name=name,
        top_k=probe_k,
        filters=filters,
    )
    vector_by_id = {
        int(item.chunk_id): item
        for item in vector_contexts
        if item.chunk_id is not None
    }
    vector_ids = list(vector_by_id)

    keyword_candidates = search_keyword_candidates(
        query=query,
        limit=probe_k,
        filters=filters,
    )
    keyword_by_id = {item.chunk_id: (item, score) for item, score in keyword_candidates}
    keyword_ids = list(keyword_by_id)

    merged_ids, merged_scores = _merge_hybrid_ids(
        vector_ids=vector_ids,
        keyword_ids=keyword_ids,
        top_k=top_k,
    )
    if not merged_ids:
        return []

    stored_chunks = load_chunks_by_ids(merged_ids)
    contexts: list[ContextItem] = []
    for chunk_id in merged_ids:
        in_vector = chunk_id in vector_by_id
        in_keyword = chunk_id in keyword_by_id
        if in_vector and in_keyword:
            mode = "hybrid"
        elif in_keyword:
            mode = "keyword"
        else:
            mode = "vector"

        stored = stored_chunks.get(chunk_id)
        if stored is not None:
            contexts.append(
                _context_from_stored_chunk(
                    stored,
                    score=merged_scores.get(chunk_id),
                    retrieval_mode=mode,
                )
            )
            continue

        vector_context = vector_by_id.get(chunk_id)
        if vector_context is None:
            keyword_chunk = keyword_by_id.get(chunk_id)
            if keyword_chunk is None:
                continue
            contexts.append(
                _context_from_stored_chunk(
                    keyword_chunk[0],
                    score=merged_scores.get(chunk_id),
                    retrieval_mode=mode,
                )
            )
            continue

        metadata = dict(vector_context.metadata or {})
        metadata["retrieval_mode"] = mode
        metadata["retrieval_score"] = merged_scores.get(chunk_id)
        contexts.append(
            ContextItem(
                chunk_id=vector_context.chunk_id,
                document_id=vector_context.document_id,
                content=vector_context.content,
                source=vector_context.source,
                page=vector_context.page,
                score=merged_scores.get(chunk_id),
                metadata=metadata,
            )
        )

    return contexts[:top_k]
