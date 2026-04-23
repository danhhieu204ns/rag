from __future__ import annotations

import json
import time
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from ...core.settings import settings
from ...models import DocumentChunk
from ..chunk_metadata import build_hyq_children
from .logging import _emit_reindex_progress, _emit_query_progress, _timed_query_step
from .models import get_embeddings
from .utils import _to_int, _preview_text, _compact_source_metadata, _build_full_text_search

_qdrant_client: QdrantClient | None = None


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

    ensure_full_text_index()

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


def ensure_full_text_index() -> None:
    """Create full_text_search payload index if it doesn't exist yet.

    No-op (with info log) for local Qdrant — payload indexes only apply to
    server Qdrant. MatchText filtering still works via linear scan in local mode.
    """
    import warnings

    client = _get_qdrant_client()
    if not _qdrant_collection_exists(client):
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            client.create_payload_index(
                collection_name=settings.qdrant_collection_name,
                field_name="full_text_search",
                field_schema=qdrant_models.TextIndexParams(
                    type="text",
                    tokenizer=qdrant_models.TokenizerType.WORD,
                    min_token_len=2,
                    lowercase=True,
                ),
            )
        _emit_reindex_progress(
            "[index] Ensured full_text_search payload index on collection '%s'.",
            settings.qdrant_collection_name,
        )
    except Exception:
        pass  # Index already exists or unsupported — safe to ignore


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

        full_text_search = _build_full_text_search(source_metadata, chunk.content)

        child_chunks = build_hyq_children(source_metadata, chunk.content)
        for child_index, (child_type, child_text) in enumerate(child_chunks):
            child_metadata = dict(metadata)
            child_metadata["child_type"] = child_type
            child_metadata["child_index"] = child_index
            child_metadata["full_text_search"] = full_text_search

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
