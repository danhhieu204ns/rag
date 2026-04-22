from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import threading
import time
import unicodedata
from uuid import NAMESPACE_URL, uuid4, uuid5
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..models import ChatMessage, DocumentChunk
from .chunk_metadata import build_hyq_children, build_keyword_blob, extract_document_codes

logger = logging.getLogger(__name__)
_QUERY_LOG_FILE_NAME = "query_trace.json"
_LEGACY_QUERY_LOG_FILE_NAME = "query_trace.jsonl"

_embeddings: Embeddings | None = None
_qdrant_client: QdrantClient | None = None
_llm: ChatOllama | None = None
_variant_llm: ChatOllama | None = None
_reranker: Any = None
_query_trace_id_ctx: ContextVar[str] = ContextVar("query_trace_id", default="-")
_embeddings_lock = threading.Lock()
_llm_lock = threading.Lock()


class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer-based embeddings with fixed max_length and optional fp16."""

    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        use_fp16: bool,
        batch_size: int,
        device: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        resolved_device = device.strip().lower() if device else "auto"
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs: dict[str, Any] = {}
        if use_fp16 and resolved_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name,
            device=resolved_device,
            model_kwargs=model_kwargs,
            local_files_only=settings.embedding_local_files_only,
        )
        self.model.max_seq_length = max_length
        self.batch_size = max(1, batch_size)
        self.use_fp16 = bool(use_fp16)
        self.device = resolved_device

        if self.use_fp16 and resolved_device.startswith("cuda"):
            self.model.half()
        elif self.use_fp16:
            logger.warning(
                "EMBEDDING_USE_FP16=true but embedding device is '%s'. fp16 is skipped.",
                resolved_device,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            [str(item or "") for item in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _query_log_file_path() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _QUERY_LOG_FILE_NAME


def _legacy_query_log_file_path() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _LEGACY_QUERY_LOG_FILE_NAME


def _load_query_log_entries(log_file: Path) -> list[dict[str, Any]]:
    if log_file.exists():
        try:
            raw = log_file.read_text(encoding="utf-8").strip()
            if not raw:
                return []

            payload = json.loads(raw)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict):
                entries = payload.get("entries")
                if isinstance(entries, list):
                    return [item for item in entries if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            return []

    legacy_log_file = _legacy_query_log_file_path()
    if not legacy_log_file.exists():
        return []

    migrated_entries: list[dict[str, Any]] = []
    try:
        for raw_line in legacy_log_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                migrated_entries.append(row)
    except OSError:
        return []

    return migrated_entries


def _append_query_log_entry(event: str, details: dict[str, Any] | None = None) -> None:
    trace_id = _query_trace_id_ctx.get()
    payload: dict[str, Any] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
        "event": event,
    }

    if details:
        payload["details"] = _json_safe_value(details)

    try:
        log_file = _query_log_file_path()
        entries = _load_query_log_entries(log_file)
        entries.append(payload)
        log_file.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("[query][file-log] Failed to write query log file: %s", exc)


def _emit_query_progress(
    message: str,
    *args: object,
    event: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raw_text = message % args if args else message
    trace_id = _query_trace_id_ctx.get()

    text = raw_text
    if trace_id and trace_id != "-":
        text = f"[trace={trace_id}] {raw_text}"

    logger.info(text)
    print(text, flush=True)

    _append_query_log_entry(
        event=event or "progress",
        details={
            "message": raw_text,
            **(details or {}),
        },
    )


def _preview_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _preview_ids(values: list[int], limit: int = 8) -> list[int | str]:
    if len(values) <= limit:
        return values
    return [*values[:limit], f"+{len(values) - limit} more"]


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_lookup_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.lower().split())


def _lookup_terms(query: str) -> list[str]:
    normalized = _normalize_lookup_text(query)
    terms = re.findall(r"[a-z0-9/-]+", normalized)
    return [term for term in terms if len(term) >= 2]


def _parse_chunk_source_metadata(raw_json: str | None) -> dict[str, object]:
    if not raw_json:
        return {}
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        return {}


def _compact_source_metadata(raw_json: str | None) -> dict[str, object]:
    raw_metadata = _parse_chunk_source_metadata(raw_json)
    if not raw_metadata:
        return {}

    wanted_keys = {
        "chunk_id",
        "source_info",
        "context",
        "search_optimization",
        "admin_tags",
        "hyq",
    }
    compact = {
        key: value
        for key, value in raw_metadata.items()
        if key in wanted_keys and value is not None
    }
    if compact:
        return compact
    return raw_metadata


def _emit_reindex_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    logger.info(text)
    print(text, flush=True)


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    if settings.qdrant_url:
        kwargs: dict[str, str] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _qdrant_client = QdrantClient(**kwargs)
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(settings.qdrant_path))

    return _qdrant_client


def _qdrant_collection_exists(client: QdrantClient) -> bool:
    collections = client.get_collections().collections
    return any(item.name == settings.qdrant_collection_name for item in collections)


def _clear_qdrant_collection(client: QdrantClient) -> None:
    if _qdrant_collection_exists(client):
        client.delete_collection(collection_name=settings.qdrant_collection_name)


def delete_vectors_by_document_id(document_id: int) -> None:
    """Delete all child vectors in Qdrant for one document id."""

    client = _get_qdrant_client()
    if not _qdrant_collection_exists(client):
        _emit_reindex_progress(
            "[vector-delete] Collection '%s' not found. Skip document_id=%s.",
            settings.qdrant_collection_name,
            document_id,
        )
        return

    document_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchValue(value=document_id),
            )
        ]
    )

    # Compatibility: older qdrant-client may require FilterSelector explicitly.
    try:
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=document_filter,
            wait=True,
        )
    except TypeError:
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=qdrant_models.FilterSelector(filter=document_filter),
            wait=True,
        )

    _emit_reindex_progress(
        "[vector-delete] Deleted vectors for document_id=%s in collection '%s'.",
        document_id,
        settings.qdrant_collection_name,
    )


def _serialize_qdrant_payload(metadata: dict[str, object], child_text: str) -> dict[str, object]:
    payload = dict(metadata)
    payload["child_text"] = child_text
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _stable_child_point_id(metadata: dict[str, object]) -> str:
    """Build deterministic UUID so upsert overwrites previous child vectors."""

    parent_chunk_id = _to_int(metadata.get("parent_chunk_id")) or 0
    child_type = str(metadata.get("child_type") or "summary")
    child_index = _to_int(metadata.get("child_index")) or 0

    raw = (
        f"{settings.qdrant_collection_name}|"
        f"parent:{parent_chunk_id}|"
        f"type:{child_type}|"
        f"index:{child_index}"
    )
    return str(uuid5(NAMESPACE_URL, raw))


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

    query_vector = get_embeddings().embed_query(query)
    _emit_query_progress(
        "[query][qdrant] Query vector dimension=%d",
        len(query_vector),
        event="qdrant_query_vector",
        details={"vector_dim": len(query_vector)},
    )

    points: list[object]
    if hasattr(client, "query_points"):
        query_response = client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        points = list(getattr(query_response, "points", []) or [])
    else:
        # Backward compatibility for older qdrant-client versions.
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


def _upsert_qdrant_collection_with_batch_embeddings(
    documents: list[Document],
    *,
    purge_document_ids: list[int] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
) -> int:
    texts = [item.page_content for item in documents]
    metadatas = [dict(item.metadata or {}) for item in documents]

    unique_document_ids = sorted({int(item) for item in (purge_document_ids or [])})
    for document_id in unique_document_ids:
        delete_vectors_by_document_id(document_id)

    if not texts:
        return 0

    total = len(texts)
    all_vectors: list[list[float]] = []
    if precomputed_vectors is not None:
        if len(precomputed_vectors) != total:
            raise RuntimeError(
                "Precomputed vectors length does not match document count for upsert."
            )
        all_vectors = precomputed_vectors
        _emit_reindex_progress(
            "[reindex] Using %d precomputed child vectors.",
            len(all_vectors),
        )
    else:
        embeddings_client = get_embeddings()
        batch_size = max(1, settings.vector_batch_size)
        batch_count = (total + batch_size - 1) // batch_size

        for batch_index, start in enumerate(range(0, total, batch_size), start=1):
            end = min(start + batch_size, total)
            batch_started_at = time.perf_counter()
            batch_vectors = embeddings_client.embed_documents(texts[start:end])
            all_vectors.extend(batch_vectors)
            elapsed = time.perf_counter() - batch_started_at

            _emit_reindex_progress(
                "[reindex] Embedded child docs %d/%d (batch %d/%d, %.2fs)",
                end,
                total,
                batch_index,
                batch_count,
                elapsed,
            )

    if not all_vectors:
        return 0

    client = _get_qdrant_client()

    vector_size = len(all_vectors[0])
    if not _qdrant_collection_exists(client):
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        _emit_reindex_progress(
            "[reindex] Created collection '%s' with vector_size=%d",
            settings.qdrant_collection_name,
            vector_size,
        )

    upsert_batch_size = max(1, settings.vector_batch_size)
    upsert_batch_count = (total + upsert_batch_size - 1) // upsert_batch_size
    for batch_index, start in enumerate(range(0, total, upsert_batch_size), start=1):
        end = min(start + upsert_batch_size, total)
        points: list[PointStruct] = []

        for idx in range(start, end):
            points.append(
                PointStruct(
                    id=_stable_child_point_id(metadatas[idx]),
                    vector=all_vectors[idx],
                    payload=_serialize_qdrant_payload(metadatas[idx], texts[idx]),
                )
            )

        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=False,
        )
        _emit_reindex_progress(
            "[reindex] Upserted child docs %d/%d (batch %d/%d)",
            end,
            total,
            batch_index,
            upsert_batch_count,
        )

    return total


def upsert_child_documents(
    documents: list[Document],
    *,
    purge_document_ids: list[int] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
) -> int:
    """Upsert prepared child documents, optionally reusing precomputed vectors."""

    return _upsert_qdrant_collection_with_batch_embeddings(
        documents,
        purge_document_ids=purge_document_ids,
        precomputed_vectors=precomputed_vectors,
    )


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

    chunk_query = db.query(DocumentChunk)
    if document_ids:
        chunk_query = chunk_query.filter(DocumentChunk.document_id.in_([int(item) for item in document_ids]))
    candidates = chunk_query.order_by(DocumentChunk.id.asc()).all()

    scored: list[tuple[int, float]] = []
    scored_chunks: dict[int, DocumentChunk] = {}
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



def get_embeddings() -> Embeddings:
    """Create or return cached Ollama embedding model instance."""

    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                backend = settings.embedding_backend.strip().lower()
                if backend == "sentence-transformers":
                    try:
                        _embeddings = SentenceTransformerEmbeddings(
                            model_name=settings.embedding_model_name,
                            max_length=settings.embedding_max_length,
                            use_fp16=settings.embedding_use_fp16,
                            batch_size=settings.embedding_batch_size,
                            device=settings.embedding_device,
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "Embedding backend 'sentence-transformers' requires package 'sentence-transformers'."
                        ) from exc
                else:
                    _embeddings = OllamaEmbeddings(
                        model=settings.embedding_model_name,
                        base_url=settings.ollama_base_url,
                        num_thread=settings.ollama_num_thread,
                        keep_alive=settings.embedding_keep_alive,
                        client_kwargs={"timeout": 180},
                    )

    return _embeddings



def get_llm() -> ChatOllama:
    """Create or return cached Ollama chat model instance."""

    global _llm

    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = ChatOllama(
                    model=settings.llm_model,
                    base_url=settings.ollama_base_url,
                    temperature=settings.llm_temperature,
                    num_thread=settings.ollama_num_thread,
                    num_ctx=settings.llm_num_ctx,
                    keep_alive=settings.llm_keep_alive,
                )
    return _llm


def warmup_embedding_model() -> None:
    """Warm embedding model once so Ollama can keep it resident with keep_alive."""

    get_embeddings().embed_documents(["warmup embedding model"])


def warmup_chat_model() -> None:
    """Warm chat model with minimal output to reduce first-request cold start."""

    get_llm().invoke("Trả về đúng 1 từ: OK")
def _get_variant_llm() -> ChatOllama:
    """Return cached LLM instance for multi-query variant generation.

    Uses MULTI_QUERY_MODEL if set, otherwise falls back to the main LLM.
    Always runs with format="json" to guarantee structured output and
    eliminate the markdown-wrapper parse failure that affects json.loads.
    A separate cache avoids sharing state with the answer-generation LLM.
    """
    global _variant_llm

    if _variant_llm is None:
        model = settings.multi_query_model or settings.llm_model
        _variant_llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_thread=settings.ollama_num_thread,
            format="json",
        )
    return _variant_llm


def _should_rewrite_query(query: str) -> tuple[bool, int, str]:
    terms = _lookup_terms(query)
    term_count = len(terms)
    min_terms = max(1, settings.query_rewrite_min_terms)
    max_terms = max(min_terms, settings.query_rewrite_max_terms)
    if term_count < min_terms:
        return False, term_count, "too_short"
    if term_count >= max_terms:
        return False, term_count, "already_specific"
    return True, term_count, "rewrite_window"


def _rewrite_query(query: str) -> str:
    llm = get_llm()
    prompt = (
        "Viết lại câu hỏi sau thành phiên bản đầy đủ hơn để tối ưu truy xuất tài liệu nội bộ. "
        "Giữ nguyên ý nghĩa gốc. Chỉ trả về đúng một câu đã viết lại, không giải thích.\n\n"
        f"Câu hỏi gốc: {query}\n"
        "Câu hỏi đã viết lại:"
    )

    started_at = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        rewritten = str(response.content or "").strip()
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        if not rewritten:
            _emit_query_progress(
                "[query][rewrite] Empty rewrite result, fallback to original query (%.2fms)",
                elapsed_ms,
                event="query_rewrite_empty",
                details={
                    "elapsed_ms": elapsed_ms,
                    "original_query_preview": _preview_text(query),
                },
            )
            return query
        rewritten_terms = len(_lookup_terms(rewritten))
        original_terms = len(_lookup_terms(query))
        if rewritten_terms < original_terms:
            _emit_query_progress(
                "[query][rewrite] Rewritten query has fewer terms (%d < %d), keep original (%.2fms)",
                rewritten_terms,
                original_terms,
                elapsed_ms,
                event="query_rewrite_rejected",
                details={
                    "elapsed_ms": elapsed_ms,
                    "original_term_count": original_terms,
                    "rewritten_term_count": rewritten_terms,
                    "original_query_preview": _preview_text(query),
                    "rewritten_query_preview": _preview_text(rewritten),
                },
            )
            return query
        _emit_query_progress(
            "[query][rewrite] Rewrite success in %.2fms: '%s' => '%s'",
            elapsed_ms,
            _preview_text(query),
            _preview_text(rewritten),
            event="query_rewrite_success",
            details={
                "elapsed_ms": elapsed_ms,
                "original_term_count": original_terms,
                "rewritten_term_count": rewritten_terms,
                "original_query_preview": _preview_text(query),
                "rewritten_query_preview": _preview_text(rewritten),
            },
        )
        return rewritten
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][rewrite] Rewrite failed after %.2fms: %s",
            elapsed_ms,
            str(exc),
            event="query_rewrite_error",
            details={
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
                "original_query_preview": _preview_text(query),
            },
        )
        return query


def _maybe_rewrite_query(query: str) -> tuple[str, dict[str, Any]]:
    should_rewrite, term_count, reason = _should_rewrite_query(query)
    details: dict[str, Any] = {
        "enabled": settings.query_rewrite_enabled,
        "term_count": term_count,
        "min_terms": settings.query_rewrite_min_terms,
        "max_terms": settings.query_rewrite_max_terms,
        "decision_reason": reason,
        "rewritten": False,
    }

    if not settings.query_rewrite_enabled:
        _emit_query_progress(
            "[query][rewrite] Disabled, skip rewrite (terms=%d)",
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    if settings.multi_query_enabled:
        details["decision_reason"] = "skipped_multi_query_active"
        _emit_query_progress(
            "[query][rewrite] Skip rewrite: multi-query active, variants cover query diversity (terms=%d)",
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    if not should_rewrite:
        _emit_query_progress(
            "[query][rewrite] Skip rewrite: reason=%s, terms=%d",
            reason,
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    rewritten = _rewrite_query(query)
    details["rewritten"] = rewritten != query
    details["effective_query_preview"] = _preview_text(rewritten)
    _emit_query_progress(
        "[query][rewrite] Rewrite decision done: rewritten=%s",
        details["rewritten"],
        event="query_rewrite_done",
        details=details,
    )
    return rewritten, details


def _extract_json_from_llm_text(raw_text: str) -> str:
    """Strip markdown code fences and leading prose that LLMs sometimes prepend.

    Handles the two most common non-compliant output patterns:
      1. ```json\\n{...}\\n```  — markdown code block
      2. "Here are the variants:\\n{...}"  — prose prefix before the JSON object
    Returns the cleaned string; callers still own the json.loads call.
    """
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return text


def _generate_query_variants(query: str, variant_count: int) -> list[str]:
    llm = _get_variant_llm()
    prompt = (
        f"Sinh {variant_count} biến thể câu hỏi có cùng ý nghĩa để tìm tài liệu. "
        "Mỗi biến thể dùng từ vựng khác nhau nhưng vẫn đúng ý định ban đầu. "
        "Trả về JSON hợp lệ dạng: {\"variants\":[\"...\",\"...\"]}.\n\n"
        f"Câu hỏi gốc: {query}"
    )

    _emit_query_progress(
        "[query][multi] Start generating %d variants for '%s'",
        max(1, variant_count),
        _preview_text(query),
        event="multi_query_variants_start",
        details={
            "variant_count": max(1, variant_count),
            "query_preview": _preview_text(query),
        },
    )

    started_at = time.perf_counter()
    raw_text = ""
    try:
        response = llm.invoke(prompt)
        raw_text = str(response.content or "").strip()
        clean_text = _extract_json_from_llm_text(raw_text)
        parsed = json.loads(clean_text)
        raw_variants = parsed.get("variants", []) if isinstance(parsed, dict) else []
        variants = [str(item).strip() for item in raw_variants if str(item).strip()]
    except json.JSONDecodeError as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][multi] JSON parse error after %.2fms — raw preview: %s",
            elapsed_ms,
            raw_text[:120],
            event="multi_query_variants_json_error",
            details={
                "elapsed_ms": elapsed_ms,
                "query_preview": _preview_text(query),
                "raw_preview": raw_text[:200],
                "error": str(exc),
            },
        )
        return []
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][multi] Variant generation failed after %.2fms: %s",
            elapsed_ms,
            str(exc),
            event="multi_query_variants_error",
            details={
                "elapsed_ms": elapsed_ms,
                "query_preview": _preview_text(query),
                "error": str(exc),
            },
        )
        return []

    deduped: list[str] = []
    seen: set[str] = {query.strip().lower()}
    for variant in variants:
        normalized = variant.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(variant)
        if len(deduped) >= max(1, variant_count):
            break

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    _emit_query_progress(
        "[query][multi] Generated %d/%d variants in %.2fms",
        len(deduped),
        max(1, variant_count),
        elapsed_ms,
        event="multi_query_variants_done",
        details={
            "elapsed_ms": elapsed_ms,
            "query_preview": _preview_text(query),
            "variant_count_requested": max(1, variant_count),
            "variant_count_generated": len(deduped),
            "variants_preview": [_preview_text(item) for item in deduped],
        },
    )
    return deduped


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



def load_index_if_available() -> bool:
    """Check whether Qdrant child collection exists."""

    client = _get_qdrant_client()
    return _qdrant_collection_exists(client)



def rebuild_index_from_chunks(chunks: list[DocumentChunk]) -> int:
    """Incrementally upsert Qdrant vectors from the provided parent chunks."""

    started_at = time.perf_counter()

    if not chunks:
        _emit_reindex_progress("[reindex] No chunks provided for incremental upsert.")
        return 0

    _emit_reindex_progress("[reindex] Start incremental upsert from %d parent chunks.", len(chunks))

    touched_document_ids = sorted({chunk.document_id for chunk in chunks})

    documents: list[Document] = []
    for chunk in chunks:
        metadata: dict[str, object] = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "parent_chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "source_page": chunk.source_page,
            "source_kind": chunk.source_kind,
        }

        source_metadata = _compact_source_metadata(chunk.source_metadata_json)
        if source_metadata:
            metadata["source_metadata"] = source_metadata

        child_chunks = build_hyq_children(source_metadata, chunk.content)
        for child_index, (child_type, child_text) in enumerate(child_chunks):
            child_metadata = dict(metadata)
            child_metadata["child_type"] = child_type
            child_metadata["child_index"] = child_index

            documents.append(
                Document(
                    page_content=child_text,
                    metadata=child_metadata,
                )
            )

    if not documents:
        _emit_reindex_progress("[reindex] No child documents generated for incremental upsert.")
        return 0

    _emit_reindex_progress(
        "[reindex] Generated %d child documents. Creating embeddings with model '%s'.",
        len(documents),
        settings.embedding_model_name,
    )

    indexed_count = _upsert_qdrant_collection_with_batch_embeddings(
        documents,
        purge_document_ids=touched_document_ids,
    )

    elapsed = time.perf_counter() - started_at
    _emit_reindex_progress(
        "[reindex] Completed. Indexed %d child documents in %.2f seconds.",
        indexed_count,
        elapsed,
    )

    return indexed_count



def get_reranker() -> Any:
    """Lazy-load CrossEncoder reranker."""
    global _reranker
    if _reranker is None and settings.reranker_enabled:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(
                settings.reranker_model,
                max_length=512,
            )
            logger.info(
                "[reranker] Loaded CrossEncoder: model=%s",
                settings.reranker_model,
            )
        except ImportError:
            logger.error(
                "[reranker] sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error("[reranker] Failed to load reranker: %s", str(e))
            return None
    return _reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int,
) -> list[Document]:
    """
    Rerank documents bằng CrossEncoder.
    
    Input:  query string + list Document (đã qua RRF)
    Output: top_k Document được sắp xếp lại theo relevance score
    """
    reranker = get_reranker()
    if reranker is None or not documents:
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
    try:
        scores: list[float] = reranker.predict(pairs).tolist()
    except Exception as e:
        logger.error("[reranker] Prediction failed: %s", str(e))
        _emit_query_progress(
            "[reranker] Reranking failed: %s",
            str(e),
            event="rerank_documents_error",
            details={"error": str(e)},
        )
        return documents[:top_k]

    # Gắn reranker score vào metadata để debug/tracing
    for doc, score in zip(documents, scores):
        doc.metadata["reranker_score"] = round(float(score), 4)

    # Sort theo score giảm dần
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, _ in ranked[:top_k]]
    
    _emit_query_progress(
        "[reranker] Reranked %d documents to top %d. "
        "Original top score: %.4f, Reranked top score: %.4f",
        len(documents),
        top_k,
        scores[0] if scores else 0.0,
        ranked[0][1] if ranked else 0.0,
        event="rerank_documents",
        details={
            "input_count": len(documents),
            "output_count": len(reranked_docs),
            "query": _preview_text(query),
            "top_k": top_k,
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
    try:
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

        if not load_index_if_available():
            _emit_query_progress(
                "[query] similarity_search stop: index not available",
                event="similarity_search_stop",
                details={"reason": "index_unavailable"},
            )
            return []

        if db is None:
            vector_only_results = _search_qdrant_children(effective_query, limit=top_k)
            _emit_query_progress(
                "[query] similarity_search done (vector_only): result_count=%d",
                len(vector_only_results),
                event="similarity_search_done_vector_only",
                details={
                    "result_count": len(vector_only_results),
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

        vector_parent_ids, parent_child_type = _vector_parent_candidates_multi_query(
            vector_queries,
            top_k,
            document_ids,
        )
        keyword_parent_ids = _keyword_parent_candidates(
            query,
            top_k,
            db,
            document_ids,
        )

        semantic_chunk_rows = _load_chunks_by_ids(db, vector_parent_ids)
        keyword_chunk_rows = _load_chunks_by_ids(db, keyword_parent_ids)
        semantic_chunk_by_id = {item.id: item for item in semantic_chunk_rows}
        keyword_chunk_by_id = {item.id: item for item in keyword_chunk_rows}

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
        merged_parent_ids, scores = _rrf_merge(vector_parent_ids, keyword_parent_ids, merge_k)
        if not merged_parent_ids:
            _emit_query_progress(
                "[query] similarity_search stop: no merged parent ids",
                event="similarity_search_stop",
                details={"reason": "no_merged_parent_ids"},
            )
            return []

        chunks = _load_chunks_by_ids(db, merged_parent_ids)
        vector_set = set(vector_parent_ids)
        keyword_set = set(keyword_parent_ids)

        results: list[Document] = []
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

        if settings.reranker_enabled and len(results) > top_k:
            final_results = rerank_documents(query, results, top_k)
        else:
            final_results = results[:top_k]
        
        mode_counts: dict[str, int] = defaultdict(int)
        for item in final_results:
            mode = str(item.metadata.get("retrieval_mode") or "unknown")
            mode_counts[mode] += 1

        final_chunk_ids = [
            _to_int(item.metadata.get("chunk_id")) or -1
            for item in final_results
        ]
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
            },
        )

        return final_results
    finally:
        _query_trace_id_ctx.reset(token)



def build_sources(context_docs: list[Document]) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Extract compact source payload from retrieved chunks."""

    sources: list[dict[str, int | str | float | dict[str, object] | None]] = []
    for doc in context_docs:
        text = doc.page_content.strip().replace("\n", " ")

        source_metadata = doc.metadata.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = None

        sources.append(
            {
                "document_id": int(doc.metadata.get("document_id")) if doc.metadata.get("document_id") is not None else None,
                "chunk_id": _to_int(doc.metadata.get("chunk_id")),
                "chunk_index": _to_int(doc.metadata.get("chunk_index")),
                "page": _to_int(doc.metadata.get("source_page")),
                "source_kind": str(doc.metadata.get("source_kind")) if doc.metadata.get("source_kind") is not None else None,
                "source_metadata": source_metadata,
                "retrieval_mode": str(doc.metadata.get("retrieval_mode")) if doc.metadata.get("retrieval_mode") is not None else None,
                "retrieval_score": _to_float(doc.metadata.get("retrieval_score")),
                "excerpt": text[:280],
            }
        )
    return sources



def generate_answer(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> str:
    """Generate answer from question, retrieval context, and chat history."""

    llm = get_llm()

    history_block = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
    )

    context_lines: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        source_metadata = doc.metadata.get("source_metadata")
        source_info = source_metadata.get("source_info") if isinstance(source_metadata, dict) else None
        context = source_metadata.get("context") if isinstance(source_metadata, dict) else None
        search_optimization = source_metadata.get("search_optimization") if isinstance(source_metadata, dict) else None

        meta_parts: list[str] = []
        if isinstance(source_info, dict):
            if source_info.get("file_name"):
                meta_parts.append(f"file={source_info.get('file_name')}")
            if source_info.get("page_number"):
                meta_parts.append(f"page={source_info.get('page_number')}")
            if source_info.get("doc_type"):
                meta_parts.append(f"doc_type={source_info.get('doc_type')}")

        if isinstance(context, dict):
            if context.get("h2"):
                meta_parts.append(f"h2={context.get('h2')}")
            if context.get("h3"):
                meta_parts.append(f"h3={context.get('h3')}")

        if isinstance(search_optimization, dict):
            document_codes = search_optimization.get("document_codes")
            if isinstance(document_codes, list) and document_codes:
                meta_parts.append(f"document_codes={', '.join(str(item) for item in document_codes[:3])}")

        retrieval_mode = doc.metadata.get("retrieval_mode")
        if retrieval_mode is not None:
            meta_parts.append(f"retrieval={retrieval_mode}")

        prefix = f"[Chunk {index}]"
        if meta_parts:
            prefix += " " + " | ".join(meta_parts)

        context_lines.append(prefix)
        context_lines.append(doc.page_content)

    context_block = "\n\n".join(context_lines)

    if not context_block:
        context_block = "No retrieved context available."

    prompt = (
        "Bạn là Tử Kỳ, một người bạn đồng hành cùng học hỏi về kiến thức công nghệ.\n"
        "Sử dụng đại từ 'mình' và 'bạn' khi trả lời để tạo cảm giác thân thiện và gần gũi.\n"
        "Use the context to answer accurately and avoid hallucination.\n"
        "Luôn bắt đầu bằng một lời dẫn dắt liên quan đến câu hỏi.\n"
        "Nếu không có đủ thông tin trong ngữ cảnh, hãy nói rõ điều đó một cách khéo léo.\n\n"
        f"Lịch sử trò chuyện:\n{history_block or 'No previous messages.'}\n\n"
        f"Ngữ cảnh:\n{context_block}\n\n"
        f"Câu hỏi: {question}\n"
        "Trả lời:"
    )

    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)



def parse_sources(raw_json: str | None) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Parse serialized sources from chat message payload."""

    if not raw_json:
        return []
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []
