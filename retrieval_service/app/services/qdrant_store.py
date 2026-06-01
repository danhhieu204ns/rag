from __future__ import annotations

import json
import re
import time
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
from .reranker import rerank_contexts

_client: QdrantClient | None = None
_LIST_OVERVIEW_PATTERN = re.compile(
    r"\bgồm\b.{0,80}\b(?:giai\s*đoạn|hoạt\s*động\s*chính|bước)\b",
    re.IGNORECASE | re.DOTALL,
)
_LIST_OVERVIEW_NEIGHBOR_COUNT = 4


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


def _looks_like_split_list_overview(content: str) -> bool:
    text = " ".join(str(content or "").split())
    return bool(_LIST_OVERVIEW_PATTERN.search(text))


def _expand_split_list_neighbors(
    contexts: list[ContextItem],
    *,
    top_k: int,
) -> list[ContextItem]:
    neighbor_map: dict[int, tuple[ContextItem, list[int]]] = {}
    neighbor_ids: list[int] = []
    for context in contexts:
        chunk_id = _to_int(context.chunk_id)
        if chunk_id is None or not _looks_like_split_list_overview(context.content):
            continue
        ids = [chunk_id + offset for offset in range(1, _LIST_OVERVIEW_NEIGHBOR_COUNT + 1)]
        neighbor_map[chunk_id] = (context, ids)
        neighbor_ids.extend(ids)

    stored_neighbors = load_chunks_by_ids(neighbor_ids)
    if not stored_neighbors:
        return contexts[:top_k]

    expanded: list[ContextItem] = []
    seen_ids: set[int] = set()

    def append_context(item: ContextItem) -> None:
        chunk_id = _to_int(item.chunk_id)
        if chunk_id is not None:
            if chunk_id in seen_ids:
                return
            seen_ids.add(chunk_id)
        expanded.append(item)

    for context in contexts:
        append_context(context)
        if len(expanded) >= top_k:
            break

        chunk_id = _to_int(context.chunk_id)
        if chunk_id is None or chunk_id not in neighbor_map:
            continue

        source_context, ids = neighbor_map[chunk_id]
        for neighbor_id in ids:
            stored = stored_neighbors.get(neighbor_id)
            if stored is None or stored.document_id != source_context.document_id:
                continue
            neighbor_context = _context_from_stored_chunk(
                stored,
                score=source_context.score,
                retrieval_mode="neighbor",
            )
            metadata = dict(neighbor_context.metadata or {})
            metadata["neighbor_of_chunk_id"] = chunk_id
            neighbor_context = ContextItem(
                chunk_id=neighbor_context.chunk_id,
                document_id=neighbor_context.document_id,
                content=neighbor_context.content,
                source=neighbor_context.source,
                page=neighbor_context.page,
                score=neighbor_context.score,
                metadata=metadata,
            )
            append_context(neighbor_context)
            if len(expanded) >= top_k:
                break

    return expanded[:top_k]


def _debug_preview_text(value: str, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _debug_source_file(metadata: dict[str, Any]) -> str | None:
    source_info = metadata.get("source_info")
    if isinstance(source_info, dict):
        file_name = str(source_info.get("file_name") or "").strip()
        if file_name:
            return file_name
    return _source_from_metadata(metadata)


def _context_debug_item(item: ContextItem, rank: int) -> dict[str, Any]:
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    return {
        "rank": rank,
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "score": item.score,
        "page": item.page,
        "file": _debug_source_file(metadata),
        "content": _debug_preview_text(item.content),
    }


def _stored_chunk_debug_item(chunk: StoredChunk, score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "score": score,
        "page": chunk.source_page,
        "file": _debug_source_file(chunk.source_metadata or {}),
        "content": _debug_preview_text(chunk.content),
    }


def _merge_hybrid_ids(
    *,
    vector_ids: list[int],
    keyword_ids: list[int],
    top_k: int,
    rrf_k: int,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> tuple[list[int], dict[int, float]]:
    rrf_k_float = float(rrf_k)
    scores: dict[int, float] = {}

    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (vector_weight / (rrf_k_float + rank))

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (keyword_weight / (rrf_k_float + rank))

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:top_k]], scores


def search_hybrid_contexts(
    *,
    query: str,
    vector: list[float],
    name: str,
    top_k: int,
    filters: RetrievalFilters | None,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
    candidate_pool: int | None = None,
    use_reranker: bool | None = None,
    debug: dict[str, Any] | None = None,
) -> list[ContextItem]:
    effective_use_reranker = settings.reranker_enabled if use_reranker is None else use_reranker
    if candidate_pool is None:
        candidate_pool = max(top_k * settings.hybrid_probe_multiplier, top_k)
        if effective_use_reranker:
            candidate_pool = max(candidate_pool, settings.reranker_candidate_pool)
    else:
        candidate_pool = max(candidate_pool, top_k)
    merge_limit = candidate_pool if effective_use_reranker else top_k

    vector_started_at = time.perf_counter()
    vector_contexts = search_contexts(
        vector=vector,
        name=name,
        top_k=candidate_pool,
        filters=filters,
    )
    vector_elapsed_ms = round((time.perf_counter() - vector_started_at) * 1000, 2)
    vector_by_id = {
        int(item.chunk_id): item
        for item in vector_contexts
        if item.chunk_id is not None
    }
    vector_ids = list(vector_by_id)

    keyword_started_at = time.perf_counter()
    keyword_candidates = search_keyword_candidates(
        query=query,
        limit=candidate_pool,
        filters=filters,
    )
    keyword_elapsed_ms = round((time.perf_counter() - keyword_started_at) * 1000, 2)
    keyword_by_id = {item.chunk_id: (item, score) for item, score in keyword_candidates}
    keyword_ids = list(keyword_by_id)

    merge_started_at = time.perf_counter()
    merged_ids, merged_scores = _merge_hybrid_ids(
        vector_ids=vector_ids,
        keyword_ids=keyword_ids,
        top_k=merge_limit,
        rrf_k=settings.hybrid_rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
    merge_elapsed_ms = round((time.perf_counter() - merge_started_at) * 1000, 2)
    if debug is not None:
        debug.update(
            {
                "top_k": top_k,
                "candidate_pool": candidate_pool,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "rrf_k": settings.hybrid_rrf_k,
                "reranker_enabled": effective_use_reranker,
                "timings_ms": {
                    "semantic_candidates": vector_elapsed_ms,
                    "keyword_candidates": keyword_elapsed_ms,
                    "rrf_merge": merge_elapsed_ms,
                },
                "vector_candidates": [
                    _context_debug_item(item, rank)
                    for rank, item in enumerate(vector_contexts, start=1)
                ],
                "keyword_candidates": [
                    _stored_chunk_debug_item(chunk, score, rank)
                    for rank, (chunk, score) in enumerate(keyword_candidates, start=1)
                ],
                "rrf_score_preview": [
                    {"chunk_id": chunk_id, "score": merged_scores.get(chunk_id, 0.0)}
                    for chunk_id in merged_ids
                ],
            }
        )
    if not merged_ids:
        if debug is not None:
            debug["merged_candidates"] = []
            debug["mode_counts"] = {}
            debug["reranker"] = {
                "enabled": effective_use_reranker,
                "model": settings.reranker_model,
                "input_count": 0,
                "output_count": 0,
                "status": "skipped_empty_pool",
            }
        return []

    stored_chunks = load_chunks_by_ids(merged_ids)
    contexts: list[ContextItem] = []
    mode_counts: dict[str, int] = {}
    for chunk_id in merged_ids:
        in_vector = chunk_id in vector_by_id
        in_keyword = chunk_id in keyword_by_id
        if in_vector and in_keyword:
            mode = "hybrid"
        elif in_keyword:
            mode = "keyword"
        else:
            mode = "vector"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

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

    if debug is not None:
        debug["mode_counts"] = mode_counts
        debug["merged_candidates"] = [
            {
                **_context_debug_item(item, rank),
                "mode": item.metadata.get("retrieval_mode")
                if isinstance(item.metadata, dict) else None,
                "rrf_score": item.score,
            }
            for rank, item in enumerate(contexts[:top_k], start=1)
        ]

    if effective_use_reranker:
        rerank_started_at = time.perf_counter()
        final_contexts, reranker_debug = rerank_contexts(
            query=query,
            contexts=contexts,
            top_k=top_k,
            enabled=True,
        )
        reranker_elapsed_ms = round((time.perf_counter() - rerank_started_at) * 1000, 2)
    else:
        final_contexts, reranker_debug = rerank_contexts(
            query=query,
            contexts=contexts,
            top_k=top_k,
            enabled=False,
        )
        reranker_elapsed_ms = 0.0

    final_contexts = _expand_split_list_neighbors(final_contexts, top_k=top_k)

    if debug is not None:
        timings = debug.setdefault("timings_ms", {})
        if isinstance(timings, dict):
            timings["reranker"] = reranker_elapsed_ms
        debug["reranker"] = reranker_debug
        debug["final_candidates"] = [
            {
                **_context_debug_item(item, rank),
                "mode": item.metadata.get("retrieval_mode")
                if isinstance(item.metadata, dict) else None,
                "rrf_score": item.score,
                "reranker_score": item.metadata.get("reranker_score")
                if isinstance(item.metadata, dict) else None,
            }
            for rank, item in enumerate(final_contexts, start=1)
        ]

    return final_contexts
