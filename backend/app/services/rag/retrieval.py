from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from ...core.settings import settings
from ...models import DocumentChunk
from ..chunk_metadata import build_keyword_blob, extract_document_codes
from ..query_rewriter import rewrite_for_vector
from .logging import _emit_query_progress, _timed_query_step, _query_trace_id_ctx
from .models import get_embeddings, get_reranker
from .qdrant import _get_qdrant_client, _qdrant_collection_exists, load_index_if_available
from .query import _maybe_rewrite_query, _generate_query_variants
from .utils import (
    _preview_text,
    _preview_ids,
    _elapsed_ms,
    _to_int,
    _to_float,
    _normalize_lookup_text,
    _lookup_terms,
    _compact_source_metadata,
)

logger = logging.getLogger(__name__)


def _search_qdrant_children(query: str, limit: int) -> list[Document]:
    if limit <= 0:
        _emit_query_progress(
            "[query][qdrant] Skip child search because limit=%d",
            limit,
            event="qdrant_child_search_skip",
            details={"limit": limit},
        )
        return []

    client = _get_qdrant_client()
    if not _qdrant_collection_exists(client):
        _emit_query_progress(
            "[query][qdrant] Collection '%s' not found. Returning 0 child hits.",
            settings.qdrant_collection_name,
            event="qdrant_collection_missing",
            details={"collection": settings.qdrant_collection_name},
        )
        return []

    _emit_query_progress(
        "[query][qdrant] Start child search: limit=%d, query='%s'",
        limit,
        _preview_text(query),
        event="qdrant_child_search_start",
        details={
            "collection": settings.qdrant_collection_name,
            "limit": limit,
            "query_preview": _preview_text(query),
        },
    )

    with _timed_query_step(
        "embed_query_vector",
        event_prefix="qdrant_embed_query",
        details={"limit": limit},
    ):
        query_vector = get_embeddings().embed_query(query)
    _emit_query_progress(
        "[query][qdrant] Query vector dimension=%d",
        len(query_vector),
        event="qdrant_query_vector",
        details={"vector_dim": len(query_vector)},
    )

    points: list[object]
    with _timed_query_step(
        "qdrant_query_points",
        event_prefix="qdrant_query_points",
        details={"limit": limit},
    ):
        if hasattr(client, "query_points"):
            query_response = client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = list(getattr(query_response, "points", []) or [])
        else:
            points = list(
                client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            )

    documents: list[Document] = []
    point_summaries: list[dict[str, object]] = []
    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        child_text = str(payload.get("child_text") or "")
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "child_text"
        }
        documents.append(Document(page_content=child_text, metadata=metadata))

        point_summaries.append(
            {
                "score": _to_float(getattr(point, "score", None)),
                "parent_chunk_id": _to_int(metadata.get("parent_chunk_id")),
                "document_id": _to_int(metadata.get("document_id")),
                "child_type": str(metadata.get("child_type") or ""),
            }
        )

    _emit_query_progress(
        "[query][qdrant] Child search done: hits=%d, preview=%s",
        len(documents),
        point_summaries[:5],
        event="qdrant_child_search_done",
        details={
            "hit_count": len(documents),
            "points_preview": point_summaries[:5],
        },
    )

    return documents


def _load_chunks_by_ids(db: Session, chunk_ids: list[int]) -> list[DocumentChunk]:
    if not chunk_ids:
        return []

    rows = db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
    row_by_id = {row.id: row for row in rows}
    return [row_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in row_by_id]


def _chunk_to_context_document(
    chunk: DocumentChunk,
    *,
    retrieval_mode: str | None = None,
    retrieval_score: float | None = None,
    child_type: str | None = None,
) -> Document:
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "source_page": chunk.source_page,
        "source_kind": chunk.source_kind,
    }

    source_metadata = _compact_source_metadata(chunk.source_metadata_json)
    if source_metadata:
        metadata["source_metadata"] = source_metadata

    if retrieval_mode:
        metadata["retrieval_mode"] = retrieval_mode
    if retrieval_score is not None:
        metadata["retrieval_score"] = retrieval_score
    if child_type:
        metadata["child_type"] = child_type

    return Document(page_content=chunk.content, metadata=metadata)


def _chunk_to_debug_payload(
    chunk: DocumentChunk,
    *,
    rank: int | None = None,
    score: float | None = None,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "source_page": chunk.source_page,
        "source_kind": chunk.source_kind,
        "content": chunk.content,
        "source_metadata": _compact_source_metadata(chunk.source_metadata_json),
    }
    if rank is not None:
        payload["rank"] = rank
    if score is not None:
        payload["score"] = score
    if retrieval_mode is not None:
        payload["retrieval_mode"] = retrieval_mode
    return payload


def _vector_parent_candidates(
    query: str,
    top_k: int,
    document_ids: list[int] | None,
) -> tuple[list[int], dict[int, str]]:
    if not load_index_if_available():
        _emit_query_progress(
            "[query][vector] Skip vector candidates because index is unavailable",
            event="semantic_candidates_skip",
            details={"reason": "index_unavailable"},
        )
        return [], {}

    probe_multiplier = max(1, settings.hybrid_probe_multiplier)
    probe_k = max(top_k * probe_multiplier, top_k)
    _emit_query_progress(
        "[query][vector] Build vector candidates: top_k=%d, probe_k=%d, document_filter=%s",
        top_k,
        probe_k,
        document_ids or [],
        event="semantic_candidates_start",
        details={
            "top_k": top_k,
            "probe_k": probe_k,
            "document_filter": document_ids or [],
        },
    )

    children = _search_qdrant_children(query, limit=probe_k)

    wanted_docs = {int(doc_id) for doc_id in document_ids} if document_ids else None
    parent_ids: list[int] = []
    parent_child_type: dict[int, str] = {}
    seen_parent_ids: set[int] = set()

    for child in children:
        parent_chunk_id = _to_int(child.metadata.get("parent_chunk_id"))
        if parent_chunk_id is None:
            parent_chunk_id = _to_int(child.metadata.get("chunk_id"))
        if parent_chunk_id is None:
            continue

        if parent_chunk_id in seen_parent_ids:
            continue

        if wanted_docs is not None:
            child_document_id = _to_int(child.metadata.get("document_id"))
            if child_document_id is None or child_document_id not in wanted_docs:
                continue

        seen_parent_ids.add(parent_chunk_id)
        parent_ids.append(parent_chunk_id)
        parent_child_type[parent_chunk_id] = str(child.metadata.get("child_type") or "summary")

        if len(parent_ids) >= probe_k:
            break

    _emit_query_progress(
        "[query][vector] Parent candidates done: count=%d, ids=%s, child_type_preview=%s",
        len(parent_ids),
        _preview_ids(parent_ids),
        dict(list(parent_child_type.items())[:5]),
        event="semantic_candidates_done",
        details={
            "semantic_parent_count": len(parent_ids),
            "semantic_parent_ids": parent_ids,
            "semantic_child_type_preview": dict(list(parent_child_type.items())[:5]),
        },
    )

    return parent_ids, parent_child_type


def _metadata_document_codes(metadata: dict[str, object]) -> set[str]:
    search_optimization = metadata.get("search_optimization")
    if not isinstance(search_optimization, dict):
        return set()

    raw_codes = search_optimization.get("document_codes")
    if not isinstance(raw_codes, list):
        return set()

    return {
        str(item).upper()
        for item in raw_codes
        if str(item).strip()
    }


def _keyword_match_score(
    *,
    query_terms: list[str],
    query_codes: list[str],
    metadata: dict[str, object],
    content: str,
) -> float:
    if not query_terms and not query_codes:
        return 0.0

    keyword_blob = build_keyword_blob(metadata, content)
    normalized_blob = _normalize_lookup_text(keyword_blob)
    metadata_codes = _metadata_document_codes(metadata)

    score = 0.0
    for code in query_codes:
        if code in metadata_codes:
            score += 12.0
        elif _normalize_lookup_text(code) in normalized_blob:
            score += 6.0

    for term in query_terms:
        if term in normalized_blob:
            score += 1.0

    return score


def _keyword_parent_candidates(
    query: str,
    top_k: int,
    db: Session,
    document_ids: list[int] | None,
) -> list[int]:
    query_terms = _lookup_terms(query)
    query_codes = [code.upper() for code in extract_document_codes(query)]
    _emit_query_progress(
        "[query][keyword] Start keyword candidates: terms=%d, codes=%d, top_k=%d, document_filter=%s",
        len(query_terms),
        len(query_codes),
        top_k,
        document_ids or [],
        event="keyword_candidates_start",
        details={
            "term_count": len(query_terms),
            "code_count": len(query_codes),
            "query_terms": query_terms[:20],
            "query_codes": query_codes[:20],
            "top_k": top_k,
            "document_filter": document_ids or [],
        },
    )

    if not query_terms and not query_codes:
        _emit_query_progress(
            "[query][keyword] Skip keyword candidates because query has no lookup terms/codes",
            event="keyword_candidates_skip",
            details={"reason": "no_terms_or_codes"},
        )
        return []

    with _timed_query_step(
        "load_keyword_candidates_from_db",
        event_prefix="keyword_candidates_db",
        details={"document_filter": document_ids or []},
    ):
        chunk_query = db.query(DocumentChunk)
        if document_ids:
            chunk_query = chunk_query.filter(DocumentChunk.document_id.in_([int(item) for item in document_ids]))
        candidates = chunk_query.order_by(DocumentChunk.id.asc()).all()

    scored: list[tuple[int, float]] = []
    scored_chunks: dict[int, DocumentChunk] = {}
    with _timed_query_step(
        "score_keyword_candidates",
        event_prefix="keyword_candidates_score",
        details={"candidate_count": len(candidates)},
    ):
        for candidate in candidates:
            metadata = _compact_source_metadata(candidate.source_metadata_json)
            score = _keyword_match_score(
                query_terms=query_terms,
                query_codes=query_codes,
                metadata=metadata,
                content=candidate.content,
            )
            if score <= 0:
                continue

            scored.append((candidate.id, score))
            scored_chunks[candidate.id] = candidate

    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    probe_multiplier = max(1, settings.hybrid_probe_multiplier)
    probe_k = max(top_k * probe_multiplier, top_k)
    selected = [chunk_id for chunk_id, _ in scored[:probe_k]]
    selected_details: list[dict[str, Any]] = []
    for rank, (chunk_id, score) in enumerate(scored[:probe_k], start=1):
        chunk = scored_chunks.get(chunk_id)
        if chunk is None:
            continue
        selected_details.append(
            _chunk_to_debug_payload(
                chunk,
                rank=rank,
                score=score,
                retrieval_mode="keyword",
            )
        )

    _emit_query_progress(
        "[query][keyword] Parent candidates done: matched=%d, selected=%d, top_preview=%s",
        len(scored),
        len(selected),
        scored[:5],
        event="keyword_candidates_done",
        details={
            "keyword_matched_count": len(scored),
            "keyword_selected_count": len(selected),
            "keyword_selected_parent_ids": selected,
            "keyword_top_scores": [
                {
                    "score": score,
                    "content": (scored_chunks.get(chunk_id).content if scored_chunks.get(chunk_id) else ""),
                }
                for chunk_id, score in scored[:20]
            ],
            "keyword_selected_chunks": selected_details,
        },
    )
    return selected


def _rrf_merge(
    vector_ids: list[int],
    keyword_ids: list[int],
    top_k: int,
) -> tuple[list[int], dict[int, float]]:
    if not vector_ids and not keyword_ids:
        _emit_query_progress(
            "[query][rrf] Skip merge because vector_ids and keyword_ids are empty",
            event="rrf_merge_skip",
            details={"reason": "empty_inputs"},
        )
        return [], {}

    rrf_k = max(1, settings.hybrid_rrf_k)
    scores: dict[int, float] = defaultdict(float)

    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] += settings.hybrid_vector_rrf_weight / (rrf_k + rank)

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        scores[chunk_id] += settings.hybrid_keyword_rrf_weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    merged_ids = [chunk_id for chunk_id, _ in ranked[:top_k]]
    _emit_query_progress(
        "[query][rrf] Merge done: vector_count=%d, keyword_count=%d, merged=%s, score_preview=%s",
        len(vector_ids),
        len(keyword_ids),
        _preview_ids(merged_ids),
        ranked[:5],
        event="rrf_merge_done",
        details={
            "semantic_parent_ids": vector_ids,
            "keyword_parent_ids": keyword_ids,
            "rrf_merged_parent_ids": merged_ids,
            "rrf_score_preview": [
                {"chunk_id": chunk_id, "score": score}
                for chunk_id, score in ranked[:20]
            ],
        },
    )
    return merged_ids, scores


def _rrf_merge_ranked_lists(
    ranked_lists: list[list[int]],
    top_k: int,
) -> tuple[list[int], dict[int, float]]:
    non_empty_lists = [items for items in ranked_lists if items]
    if not non_empty_lists:
        return [], {}

    rrf_k = max(1, settings.hybrid_rrf_k)
    scores: dict[int, float] = defaultdict(float)
    for ranked in non_empty_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    merged_ids = [chunk_id for chunk_id, _ in ranked[:top_k]]
    _emit_query_progress(
        "[query][multi] RRF merged %d ranked lists into %d parent ids",
        len(non_empty_lists),
        len(merged_ids),
        event="multi_query_rrf_merge",
        details={
            "input_list_count": len(non_empty_lists),
            "output_count": len(merged_ids),
            "output_ids": merged_ids,
        },
    )
    return merged_ids, scores


def _vector_parent_candidates_multi_query(
    queries: list[str],
    top_k: int,
    document_ids: list[int] | None,
) -> tuple[list[int], dict[int, str]]:
    filtered_queries = [item for item in queries if item.strip()]
    if not filtered_queries:
        _emit_query_progress(
            "[query][multi] Skip multi-query vector search because query list is empty",
            event="multi_query_vector_skip",
            details={"reason": "empty_query_list"},
        )
        return [], {}

    if len(filtered_queries) == 1:
        _emit_query_progress(
            "[query][multi] Single query mode, fallback to standard vector search",
            event="multi_query_vector_single_mode",
            details={"query_preview": _preview_text(filtered_queries[0])},
        )
        return _vector_parent_candidates(filtered_queries[0], top_k, document_ids)

    max_workers = min(max(1, settings.multi_query_max_workers), len(filtered_queries))
    _emit_query_progress(
        "[query][multi] Parallel vector search start: queries=%d, workers=%d",
        len(filtered_queries),
        max_workers,
        event="multi_query_vector_start",
        details={
            "query_count": len(filtered_queries),
            "queries": [_preview_text(item) for item in filtered_queries],
            "workers": max_workers,
            "top_k": top_k,
            "document_filter": document_ids or [],
        },
    )

    ranked_lists: list[list[int]] = []
    parent_child_type: dict[int, str] = {}
    query_to_result: dict[str, list[int]] = {}
    query_elapsed_ms: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        query_start_ts = {variant_query: time.perf_counter() for variant_query in filtered_queries}
        future_map = {
            executor.submit(_vector_parent_candidates, variant_query, top_k, document_ids): variant_query
            for variant_query in filtered_queries
        }
        for future in as_completed(future_map):
            variant_query = future_map[future]
            ids, child_type = future.result()
            query_elapsed_ms[variant_query] = round((time.perf_counter() - query_start_ts[variant_query]) * 1000, 2)
            ranked_lists.append(ids)
            query_to_result[variant_query] = ids
            for chunk_id, child_label in child_type.items():
                if chunk_id not in parent_child_type:
                    parent_child_type[chunk_id] = child_label

    probe_multiplier = max(1, settings.hybrid_probe_multiplier)
    probe_k = max(top_k * probe_multiplier, top_k)
    merged_ids, _ = _rrf_merge_ranked_lists(ranked_lists, probe_k)

    _emit_query_progress(
        "[query][multi] Multi-query vector candidates done: queries=%d, merged=%d",
        len(filtered_queries),
        len(merged_ids),
        event="multi_query_vector_done",
        details={
            "queries": filtered_queries,
            "query_results": {q: _preview_ids(ids) for q, ids in query_to_result.items()},
            "query_elapsed_ms": query_elapsed_ms,
            "merged_parent_ids": merged_ids,
        },
    )

    return merged_ids, parent_child_type


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int,
) -> list[Document]:
    """Rerank documents bằng CrossEncoder."""
    rerank_started_at = time.perf_counter()
    reranker = get_reranker()
    if reranker is None or not documents:
        _emit_query_progress(
            "[reranker] Skip reranking: reranker=%s, docs=%d (%.2fms)",
            "ready" if reranker is not None else "missing",
            len(documents),
            _elapsed_ms(rerank_started_at),
            event="rerank_documents_skip",
            details={
                "reranker_ready": reranker is not None,
                "document_count": len(documents),
                "elapsed_ms": _elapsed_ms(rerank_started_at),
            },
        )
        return documents[:top_k]

    _emit_query_progress(
        "[reranker] Starting reranking for %d documents",
        len(documents),
        event="rerank_documents_start",
        details={
            "input_count": len(documents),
            "top_k": top_k,
            "query": _preview_text(query),
        },
    )

    pairs = [(query, doc.page_content) for doc in documents]
    predict_started_at = time.perf_counter()
    try:
        scores: list[float] = reranker.predict(pairs).tolist()
    except Exception as e:
        logger.error("[reranker] Prediction failed: %s", str(e))
        _emit_query_progress(
            "[reranker] Reranking failed after %.2fms: %s",
            _elapsed_ms(rerank_started_at),
            str(e),
            event="rerank_documents_error",
            details={
                "error": str(e),
                "elapsed_ms": _elapsed_ms(rerank_started_at),
            },
        )
        return documents[:top_k]

    predict_elapsed_ms = _elapsed_ms(predict_started_at)

    for doc, score in zip(documents, scores):
        doc.metadata["reranker_score"] = round(float(score), 4)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in ranked[:top_k]]
    
    _emit_query_progress(
        "[reranker] Reranked %d documents to top %d in %.2fms (predict=%.2fms). "
        "Original top score: %.4f, Reranked top score: %.4f",
        len(documents),
        top_k,
        _elapsed_ms(rerank_started_at),
        predict_elapsed_ms,
        scores[0] if scores else 0.0,
        ranked[0][1] if ranked else 0.0,
        event="rerank_documents",
        details={
            "input_count": len(documents),
            "output_count": len(reranked_docs),
            "query": _preview_text(query),
            "top_k": top_k,
            "elapsed_ms": _elapsed_ms(rerank_started_at),
            "predict_elapsed_ms": predict_elapsed_ms,
            "original_top_score": float(scores[0]) if scores else 0.0,
            "reranked_top_score": float(ranked[0][1]) if ranked else 0.0,
        },
    )
    
    return reranked_docs


def similarity_search(
    query: str,
    top_k: int,
    db: Session | None = None,
    document_ids: list[int] | None = None,
) -> list[Document]:
    """Search relevant parent chunks using hybrid (vector + keyword) retrieval."""

    trace_id = f"q-{int(time.time() * 1000)}-{uuid4().hex[:8]}"
    token = _query_trace_id_ctx.set(trace_id)
    query_started_at = time.perf_counter()
    stage_timings_ms: dict[str, float] = {}
    started_at = time.perf_counter()
    try:
        with _timed_query_step(
            "prepare_effective_query",
            event_prefix="similarity_prepare_query",
            details={"top_k": top_k, "db_mode": "hybrid" if db is not None else "vector_only"},
        ):
            effective_query, rewrite_details = _maybe_rewrite_query(query)
            vector_queries = [effective_query]
            if settings.multi_query_enabled:
                variants = _generate_query_variants(effective_query, settings.multi_query_variants)
                vector_queries.extend(variants)

        _emit_query_progress(
            "[query] Query pipeline: original_terms=%d, effective_terms=%d, vector_queries=%d",
            len(_lookup_terms(query)),
            len(_lookup_terms(effective_query)),
            len(vector_queries),
            event="query_pipeline_summary",
            details={
                "original_query_preview": _preview_text(query),
                "effective_query_preview": _preview_text(effective_query),
                "original_term_count": len(_lookup_terms(query)),
                "effective_term_count": len(_lookup_terms(effective_query)),
                "vector_queries": [_preview_text(item) for item in vector_queries],
                "multi_query_enabled": settings.multi_query_enabled,
            },
        )

        _emit_query_progress(
            "[query] Start similarity_search: top_k=%d, db_mode=%s, document_filter=%s, query='%s'",
            top_k,
            "hybrid" if db is not None else "vector_only",
            document_ids or [],
            _preview_text(query),
            event="similarity_search_start",
            details={
                "top_k": top_k,
                "db_mode": "hybrid" if db is not None else "vector_only",
                "document_filter": document_ids or [],
                "query_preview": _preview_text(query),
                "effective_query_preview": _preview_text(effective_query),
                "rewrite": rewrite_details,
                "multi_query_enabled": settings.multi_query_enabled,
                "multi_query_count": len(vector_queries),
            },
        )

        index_check_started_at = time.perf_counter()
        index_available = load_index_if_available()
        stage_timings_ms["index_check"] = _elapsed_ms(index_check_started_at)
        with _timed_query_step(
            "check_index_available",
            event_prefix="similarity_check_index",
        ):
            index_available = load_index_if_available()

        if not index_available:
            _emit_query_progress(
                "[query] similarity_search stop: index not available",
                event="similarity_search_stop",
                details={
                    "reason": "index_unavailable",
                    "stage_timings_ms": stage_timings_ms,
                    "total_elapsed_ms": _elapsed_ms(query_started_at),
                },
            )
            return []

        if db is None:
            vector_only_started_at = time.perf_counter()
            vector_only_results = _search_qdrant_children(query, limit=top_k)
            stage_timings_ms["vector_only_search"] = _elapsed_ms(vector_only_started_at)
            with _timed_query_step(
                "vector_only_search",
                event_prefix="similarity_vector_only",
                details={"top_k": top_k},
            ):
                vector_only_results = _search_qdrant_children(effective_query, limit=top_k)
            _emit_query_progress(
                "[query] similarity_search done (vector_only): result_count=%d, elapsed=%.2fms",
                len(vector_only_results),
                stage_timings_ms["vector_only_search"],
                event="similarity_search_done_vector_only",
                details={
                    "result_count": len(vector_only_results),
                    "stage_timings_ms": stage_timings_ms,
                    "total_elapsed_ms": _elapsed_ms(query_started_at),
                    "effective_query_preview": _preview_text(effective_query),
                    "result_preview": [
                        {
                            "parent_chunk_id": _to_int(item.metadata.get("parent_chunk_id")),
                            "document_id": _to_int(item.metadata.get("document_id")),
                            "child_type": str(item.metadata.get("child_type") or ""),
                        }
                        for item in vector_only_results[:20]
                    ],
                },
            )
            return vector_only_results

        vector_query = rewrite_for_vector(query)
        if vector_query != query:
            _emit_query_progress(
                "[query][rewrite] Vector query expanded: '%s' → '%s'",
                _preview_text(query),
                _preview_text(vector_query),
                event="query_rewrite",
                details={"original_query": query, "vector_query": vector_query},
            )

        semantic_started_at = time.perf_counter()
        vector_parent_ids, parent_child_type = _vector_parent_candidates(
            vector_query,
            top_k,
            document_ids,
        )
        stage_timings_ms["semantic_candidates"] = _elapsed_ms(semantic_started_at)

        keyword_started_at = time.perf_counter()
        keyword_parent_ids = _keyword_parent_candidates(
            query,
            top_k,
            db,
            document_ids,
        )
        stage_timings_ms["keyword_candidates"] = _elapsed_ms(keyword_started_at)

        load_semantic_chunks_started_at = time.perf_counter()
        semantic_chunk_rows = _load_chunks_by_ids(db, vector_parent_ids)
        stage_timings_ms["load_semantic_chunks"] = _elapsed_ms(load_semantic_chunks_started_at)

        load_keyword_chunks_started_at = time.perf_counter()
        keyword_chunk_rows = _load_chunks_by_ids(db, keyword_parent_ids)
        stage_timings_ms["load_keyword_chunks"] = _elapsed_ms(load_keyword_chunks_started_at)

        with _timed_query_step(
            "vector_parent_candidates_multi_query",
            event_prefix="similarity_vector_candidates",
            details={"query_count": len(vector_queries), "top_k": top_k},
        ):
            vector_parent_ids, parent_child_type = _vector_parent_candidates_multi_query(
                vector_queries,
                top_k,
                document_ids,
            )

        with _timed_query_step(
            "keyword_parent_candidates",
            event_prefix="similarity_keyword_candidates",
            details={"top_k": top_k},
        ):
            keyword_parent_ids = _keyword_parent_candidates(
                query,
                top_k,
                db,
                document_ids,
            )

        with _timed_query_step(
            "load_candidate_chunks",
            event_prefix="similarity_load_candidate_chunks",
            details={
                "semantic_parent_count": len(vector_parent_ids),
                "keyword_parent_count": len(keyword_parent_ids),
            },
        ):
            semantic_chunk_rows = _load_chunks_by_ids(db, vector_parent_ids)
            keyword_chunk_rows = _load_chunks_by_ids(db, keyword_parent_ids)
        semantic_chunk_by_id = {item.id: item for item in semantic_chunk_rows}
        keyword_chunk_by_id = {item.id: item for item in keyword_chunk_rows}

        candidate_payload_started_at = time.perf_counter()
        semantic_chunk_details: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(vector_parent_ids, start=1):
            chunk = semantic_chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            semantic_chunk_details.append(
                _chunk_to_debug_payload(
                    chunk,
                    rank=rank,
                    retrieval_mode="semantic",
                )
            )

        keyword_chunk_details: list[dict[str, Any]] = []
        for rank, chunk_id in enumerate(keyword_parent_ids, start=1):
            chunk = keyword_chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            keyword_chunk_details.append(
                _chunk_to_debug_payload(
                    chunk,
                    rank=rank,
                    retrieval_mode="keyword",
                )
            )
        stage_timings_ms["build_candidate_debug_payloads"] = _elapsed_ms(candidate_payload_started_at)

        _emit_query_progress(
            "[query] Candidate chunk details: semantic=%d, keyword=%d",
            len(semantic_chunk_details),
            len(keyword_chunk_details),
            event="candidate_chunk_details",
            details={
                "semantic_chunks": semantic_chunk_details,
                "keyword_chunks": keyword_chunk_details,
            },
        )

        merge_k = settings.reranker_candidate_pool if settings.reranker_enabled else top_k
        rrf_started_at = time.perf_counter()
        merged_parent_ids, scores = _rrf_merge(vector_parent_ids, keyword_parent_ids, merge_k)
        stage_timings_ms["rrf_merge"] = _elapsed_ms(rrf_started_at)
        with _timed_query_step(
            "rrf_merge_candidates",
            event_prefix="similarity_rrf_merge",
            details={"merge_k": merge_k},
        ):
            merged_parent_ids, scores = _rrf_merge(vector_parent_ids, keyword_parent_ids, merge_k)
        if not merged_parent_ids:
            _emit_query_progress(
                "[query] similarity_search stop: no merged parent ids",
                event="similarity_search_stop",
                details={
                    "reason": "no_merged_parent_ids",
                    "stage_timings_ms": stage_timings_ms,
                    "total_elapsed_ms": _elapsed_ms(query_started_at),
                },
            )
            return []

        load_merged_chunks_started_at = time.perf_counter()
        chunks = _load_chunks_by_ids(db, merged_parent_ids)
        stage_timings_ms["load_merged_chunks"] = _elapsed_ms(load_merged_chunks_started_at)
        with _timed_query_step(
            "load_merged_chunks",
            event_prefix="similarity_load_merged_chunks",
            details={"merged_count": len(merged_parent_ids)},
        ):
            chunks = _load_chunks_by_ids(db, merged_parent_ids)
        vector_set = set(vector_parent_ids)
        keyword_set = set(keyword_parent_ids)

        build_context_docs_started_at = time.perf_counter()
        results: list[Document] = []
        with _timed_query_step(
            "build_context_documents",
            event_prefix="similarity_build_context_documents",
            details={"chunk_count": len(chunks)},
        ):
            for chunk in chunks:
                if chunk.id in vector_set and chunk.id in keyword_set:
                    retrieval_mode = "hybrid"
                elif chunk.id in keyword_set:
                    retrieval_mode = "keyword"
                else:
                    retrieval_mode = "vector"

                results.append(
                    _chunk_to_context_document(
                        chunk,
                        retrieval_mode=retrieval_mode,
                        retrieval_score=scores.get(chunk.id),
                        child_type=parent_child_type.get(chunk.id),
                    )
                )
        stage_timings_ms["build_context_documents"] = _elapsed_ms(build_context_docs_started_at)

        rerank_stage_started_at = time.perf_counter()
        if settings.reranker_enabled and len(results) > top_k:
            final_results = rerank_documents(query, results, top_k)
            stage_timings_ms["reranker"] = _elapsed_ms(rerank_stage_started_at)
            with _timed_query_step(
                "rerank_documents",
                event_prefix="similarity_rerank",
                details={"input_count": len(results), "top_k": top_k},
            ):
                final_results = rerank_documents(query, results, top_k)
        else:
            final_results = results[:top_k]
            stage_timings_ms["reranker"] = _elapsed_ms(rerank_stage_started_at)
            if not settings.reranker_enabled:
                stage_timings_ms["reranker_skipped"] = 1.0
            elif len(results) <= top_k:
                stage_timings_ms["reranker_skipped"] = 1.0
        
        mode_counts: dict[str, int] = defaultdict(int)
        for item in final_results:
            mode = str(item.metadata.get("retrieval_mode") or "unknown")
            mode_counts[mode] += 1

        final_chunk_ids = [
            _to_int(item.metadata.get("chunk_id")) or -1
            for item in final_results
        ]
        final_payload_started_at = time.perf_counter()
        final_chunk_details: list[dict[str, Any]] = []
        for rank, item in enumerate(final_results, start=1):
            final_chunk_details.append(
                {
                    "rank": rank,
                    "chunk_id": _to_int(item.metadata.get("chunk_id")),
                    "document_id": _to_int(item.metadata.get("document_id")),
                    "chunk_index": _to_int(item.metadata.get("chunk_index")),
                    "source_page": _to_int(item.metadata.get("source_page")),
                    "source_kind": str(item.metadata.get("source_kind") or ""),
                    "retrieval_mode": str(item.metadata.get("retrieval_mode") or ""),
                    "retrieval_score": _to_float(item.metadata.get("retrieval_score")),
                    "reranker_score": _to_float(item.metadata.get("reranker_score")) if settings.reranker_enabled else None,
                    "child_type": str(item.metadata.get("child_type") or ""),
                    "source_metadata": item.metadata.get("source_metadata"),
                    "content": item.page_content,
                }
            )
        stage_timings_ms["build_final_payload"] = _elapsed_ms(final_payload_started_at)
        stage_timings_ms["total"] = _elapsed_ms(query_started_at)

        _emit_query_progress(
            "[query][timing] semantic=%.2fms keyword=%.2fms rrf=%.2fms reranker=%.2fms total=%.2fms",
            stage_timings_ms.get("semantic_candidates", 0.0),
            stage_timings_ms.get("keyword_candidates", 0.0),
            stage_timings_ms.get("rrf_merge", 0.0),
            stage_timings_ms.get("reranker", 0.0),
            stage_timings_ms.get("total", 0.0),
            event="similarity_search_timing",
            details={
                "stage_timings_ms": stage_timings_ms,
                "top_k": top_k,
                "semantic_candidate_count": len(vector_parent_ids),
                "keyword_candidate_count": len(keyword_parent_ids),
                "merged_candidate_count": len(merged_parent_ids),
                "final_result_count": len(final_results),
            },
        )

        _emit_query_progress(
            "[query] similarity_search done: parent_chunks=%d, final_results=%d, modes=%s, chunk_ids=%s",
            len(chunks),
            len(final_results),
            dict(mode_counts),
            _preview_ids(final_chunk_ids),
            event="similarity_search_done",
            details={
                "semantic_parent_ids": vector_parent_ids,
                "keyword_parent_ids": keyword_parent_ids,
                "final_parent_count": len(chunks),
                "final_result_count": len(final_results),
                "final_modes": dict(mode_counts),
                "final_chunk_ids": final_chunk_ids,
                "final_chunks": final_chunk_details,
                "stage_timings_ms": stage_timings_ms,
                "total_elapsed_ms": stage_timings_ms.get("total", 0.0),
            },
        )

        return final_results
    finally:
        total_elapsed_ms = (time.perf_counter() - started_at) * 1000
        _emit_query_progress(
            "[timing][query] DONE step=similarity_search_total elapsed_ms=%.2f",
            total_elapsed_ms,
            event="similarity_search_total",
            details={
                "elapsed_ms": total_elapsed_ms,
                "top_k": top_k,
                "db_mode": "hybrid" if db is not None else "vector_only",
            },
        )
        _query_trace_id_ctx.reset(token)
