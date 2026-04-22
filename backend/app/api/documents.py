from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import OperationalError
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..db import SessionLocal, get_db
from ..models import AdminUser, ChunkMetadataCache, Document, DocumentChunk, DocumentIndexState
from ..schemas import DocumentChunkListResponse, DocumentChunkRead, DocumentRead, DocumentUpdate, EmbedDocumentResponse
from ..services.chunk_metadata import build_hyq_children, build_structured_chunk_metadata_batch
from ..services.document_processing import load_source_documents, split_source_documents
from .auth import require_admin
from ..services.rag_runtime import delete_vectors_by_document_id, get_embeddings, upsert_child_documents

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


def _emit_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    logger.info(text)
    print(text, flush=True)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _extract_source_page(metadata: dict[str, Any]) -> int | None:
    source_page = _to_int(metadata.get("source_page"))
    if source_page is not None and source_page > 0:
        return source_page

    page_number = _to_int(metadata.get("page_number"))
    if page_number is not None and page_number > 0:
        return page_number

    return None


def _extract_source_kind(metadata: dict[str, Any], file_suffix: str) -> str:
    source_parser = str(metadata.get("source_parser") or "legacy").lower()
    source_type = str(metadata.get("source_type") or "").lower()

    if not source_type:
        source_type = "pdf" if file_suffix == ".pdf" else "text"

    if source_type == "pdf" and source_parser == "marker":
        return "pdf_marker_page"
    if source_type == "pdf":
        return "pdf_page"
    if source_type == "text":
        return "text_chunk"
    return source_type


def _serialize_source_metadata(metadata: dict[str, Any]) -> str | None:
    if not metadata:
        return None

    safe_metadata = _json_safe_value(metadata)
    if not isinstance(safe_metadata, dict):
        return None
    if not safe_metadata:
        return None
    return json.dumps(safe_metadata, ensure_ascii=False)


def _parse_source_metadata(raw_json: str | None) -> dict[str, Any] | None:
    if not raw_json:
        return None
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _compute_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _compute_chunk_fingerprint(
    *,
    chunk_text: str,
    raw_metadata: dict[str, Any],
    source_page: int | None,
    source_kind: str,
) -> str:
    payload = {
        "chunk_text": chunk_text,
        "source_page": source_page,
        "source_kind": source_kind,
        "raw_metadata": _json_safe_value(raw_metadata),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_metadata_cache(
    *,
    db: Session,
    document_id: int,
    file_hash: str,
    chunk_fingerprints: list[str],
) -> dict[str, dict[str, Any]]:
    if not chunk_fingerprints:
        return {}

    rows = (
        db.query(ChunkMetadataCache)
        .filter(ChunkMetadataCache.document_id == document_id)
        .filter(ChunkMetadataCache.file_hash == file_hash)
        .filter(ChunkMetadataCache.chunk_fingerprint.in_(chunk_fingerprints))
        .all()
    )

    cache_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        parsed = _parse_source_metadata(row.metadata_json)
        if parsed is None:
            continue
        cache_map[row.chunk_fingerprint] = parsed
    return cache_map


def _save_metadata_cache(
    *,
    db: Session,
    document_id: int,
    file_hash: str,
    cached_payloads: list[tuple[str, str]],
) -> None:
    write_attempts = 3
    for attempt in range(write_attempts):
        try:
            db.query(ChunkMetadataCache).filter(
                ChunkMetadataCache.document_id == document_id,
                ChunkMetadataCache.file_hash == file_hash,
            ).delete(synchronize_session=False)

            if cached_payloads:
                rows = [
                    ChunkMetadataCache(
                        document_id=document_id,
                        file_hash=file_hash,
                        chunk_fingerprint=fingerprint,
                        metadata_json=metadata_json,
                    )
                    for fingerprint, metadata_json in cached_payloads
                ]
                db.add_all(rows)

            db.commit()
            return
        except OperationalError as exc:
            db.rollback()
            if "database is locked" not in str(exc).lower() or attempt == write_attempts - 1:
                raise RuntimeError(
                    "Database is busy while saving metadata cache. Please retry after a few seconds."
                ) from exc
            time.sleep(0.4 * (attempt + 1))



def _to_document_read(document: Document, chunk_count: int) -> DocumentRead:
    return DocumentRead(
        id=document.id,
        title=document.title,
        original_filename=document.original_filename,
        content_type=document.content_type,
        status=document.status,
        chunk_count=chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _to_document_chunk_read(chunk: DocumentChunk) -> DocumentChunkRead:
    return DocumentChunkRead(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        source_page=chunk.source_page,
        source_kind=chunk.source_kind,
        source_metadata=_parse_source_metadata(chunk.source_metadata_json),
        created_at=chunk.created_at,
    )


def _build_document_chunks_for_indexing(
    *,
    db: Session,
    document_id: int,
    original_filename: str,
    file_path: Path,
    file_hash: str,
) -> tuple[list[DocumentChunk], list[tuple[str, str]], list[dict[str, Any]]]:
    load_started_at = time.perf_counter()
    loaded_documents = load_source_documents(file_path)
    load_elapsed = time.perf_counter() - load_started_at
    _emit_progress(
        "[embed_document][bg] Loaded %d source document blocks for document_id=%s",
        len(loaded_documents),
        document_id,
    )

    loaded_source_type_counter: Counter[str] = Counter()
    loaded_parser_counter: Counter[str] = Counter()
    loaded_text_lengths: list[int] = []
    for item in loaded_documents:
        metadata = dict(item.metadata or {})
        loaded_source_type_counter[str(metadata.get("source_type") or "unknown")] += 1
        loaded_parser_counter[str(metadata.get("source_parser") or "legacy")] += 1
        loaded_text_lengths.append(len(str(item.page_content or "")))

    avg_loaded_chars = (
        sum(loaded_text_lengths) / len(loaded_text_lengths)
        if loaded_text_lengths
        else 0.0
    )
    _emit_progress(
        "[embed_document][bg][debug] Load stats document_id=%s elapsed=%.2fs avg_chars=%.1f source_types=%s parsers=%s",
        document_id,
        load_elapsed,
        avg_loaded_chars,
        dict(loaded_source_type_counter),
        dict(loaded_parser_counter),
    )

    split_started_at = time.perf_counter()
    split_documents = split_source_documents(
        loaded_documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    split_elapsed = time.perf_counter() - split_started_at
    _emit_progress(
        "[embed_document][bg] Split into %d chunks for document_id=%s",
        len(split_documents),
        document_id,
    )

    split_lengths = [len(str(item.page_content or "")) for item in split_documents]
    split_avg_chars = sum(split_lengths) / len(split_lengths) if split_lengths else 0.0
    split_min_chars = min(split_lengths) if split_lengths else 0
    split_max_chars = max(split_lengths) if split_lengths else 0
    _emit_progress(
        "[embed_document][bg][debug] Split stats document_id=%s elapsed=%.2fs min_chars=%d avg_chars=%.1f max_chars=%d chunk_size=%d overlap=%d",
        document_id,
        split_elapsed,
        split_min_chars,
        split_avg_chars,
        split_max_chars,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    candidates: list[dict[str, Any]] = []
    source_kind_counter: Counter[str] = Counter()
    source_pages: list[int] = []
    skipped_empty_chunks = 0
    for item in split_documents:
        text = item.page_content.strip()
        if not text:
            skipped_empty_chunks += 1
            continue

        item_metadata = dict(item.metadata or {})
        source_page = _extract_source_page(item_metadata)
        source_kind = _extract_source_kind(item_metadata, file_path.suffix.lower())
        source_kind_counter[source_kind] += 1
        if source_page is not None:
            source_pages.append(source_page)
        chunk_index = len(candidates)
        fingerprint = _compute_chunk_fingerprint(
            chunk_text=text,
            raw_metadata=item_metadata,
            source_page=source_page,
            source_kind=source_kind,
        )
        candidates.append(
            {
                "chunk_index": chunk_index,
                "chunk_text": text,
                "raw_metadata": item_metadata,
                "source_page": source_page,
                "source_kind": source_kind,
                "fingerprint": fingerprint,
            }
        )

    metadata_cache = _load_metadata_cache(
        db=db,
        document_id=document_id,
        file_hash=file_hash,
        chunk_fingerprints=[str(item["fingerprint"]) for item in candidates],
    )
    cached_count = sum(1 for item in candidates if item["fingerprint"] in metadata_cache)
    uncached_count = len(candidates) - cached_count
    cache_ratio = (cached_count / len(candidates) * 100.0) if candidates else 0.0
    metadata_batch_size = max(1, settings.metadata_llm_batch_size)
    metadata_batch_count = (uncached_count + metadata_batch_size - 1) // metadata_batch_size if uncached_count else 0

    _emit_progress(
        "[embed_document][bg] Preparing metadata for %d chunks (cache_hit=%d, document_id=%s)",
        len(candidates),
        cached_count,
        document_id,
    )
    _emit_progress(
        "[embed_document][bg][debug] Candidate stats document_id=%s non_empty=%d empty_skipped=%d cache_hit_ratio=%.1f%% uncached=%d batch_size=%d batch_count=%d source_kinds=%s page_span=%s",
        document_id,
        len(candidates),
        skipped_empty_chunks,
        cache_ratio,
        uncached_count,
        metadata_batch_size,
        metadata_batch_count,
        dict(source_kind_counter),
        (
            f"{min(source_pages)}-{max(source_pages)}"
            if source_pages
            else "n/a"
        ),
    )

    prepared_metadata: list[dict[str, Any] | None] = [None] * len(candidates)
    uncached_candidates: list[dict[str, Any]] = []

    for item in candidates:
        cached_metadata = metadata_cache.get(str(item["fingerprint"]))
        if cached_metadata is not None:
            prepared_metadata[int(item["chunk_index"])] = cached_metadata
            continue
        uncached_candidates.append(item)

    precomputed_child_rows: list[dict[str, Any]] = []

    async def _prepare_with_overlap() -> None:
        if not candidates:
            return

        llm_batch_size = max(1, settings.metadata_llm_batch_size)
        batches: list[list[dict[str, Any]]] = [
            candidates[start : start + llm_batch_size]
            for start in range(0, len(candidates), llm_batch_size)
        ]

        loop = asyncio.get_running_loop()
        embeddings_client = get_embeddings()

        def _prepare_batch_metadata(batch: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
            uncached_batch = [item for item in batch if prepared_metadata[int(item["chunk_index"])] is None]
            generated_by_chunk_index: dict[int, dict[str, Any]] = {}

            if uncached_batch:
                generated = build_structured_chunk_metadata_batch(
                    document_id=document_id,
                    file_name=original_filename,
                    chunks=[
                        {
                            "chunk_index": item["chunk_index"],
                            "source_page": item["source_page"],
                            "raw_metadata": item["raw_metadata"],
                            "chunk_text": item["chunk_text"],
                        }
                        for item in uncached_batch
                    ],
                )
                for idx, item in enumerate(uncached_batch):
                    generated_by_chunk_index[int(item["chunk_index"])] = generated[idx]

            prepared_batch: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for item in batch:
                chunk_index = int(item["chunk_index"])
                structured_metadata = prepared_metadata[chunk_index]
                if structured_metadata is None:
                    structured_metadata = generated_by_chunk_index.get(chunk_index)
                if structured_metadata is None:
                    raise RuntimeError(
                        f"Failed to build chunk metadata for chunk_index={chunk_index}."
                    )

                prepared_metadata[chunk_index] = structured_metadata
                prepared_batch.append((item, structured_metadata))

            return prepared_batch

        def _embed_children_from_batch(
            prepared_batch: list[tuple[dict[str, Any], dict[str, Any]]],
        ) -> list[dict[str, Any]]:
            payload_rows: list[dict[str, Any]] = []
            texts: list[str] = []

            for candidate_item, structured_metadata in prepared_batch:
                chunk_text = str(candidate_item["chunk_text"])
                child_chunks = build_hyq_children(structured_metadata, chunk_text)
                for child_index, (child_type, child_text) in enumerate(child_chunks):
                    row = {
                        "document_id": document_id,
                        "chunk_index": int(candidate_item["chunk_index"]),
                        "source_page": _to_int(candidate_item.get("source_page")),
                        "source_kind": str(candidate_item.get("source_kind") or ""),
                        "source_metadata": structured_metadata,
                        "child_type": child_type,
                        "child_index": child_index,
                        "child_text": child_text,
                    }
                    payload_rows.append(row)
                    texts.append(child_text)

            if not texts:
                return []

            vectors = embeddings_client.embed_documents(texts)
            if len(vectors) != len(payload_rows):
                raise RuntimeError("Child embedding count mismatch while building index payload.")

            for idx, vector in enumerate(vectors):
                payload_rows[idx]["vector"] = vector

            return payload_rows

        if not batches:
            return

        metadata_future = loop.run_in_executor(None, _prepare_batch_metadata, batches[0])

        for batch_index in range(len(batches)):
            prepared_batch = await metadata_future

            child_embedding_future = loop.run_in_executor(
                None,
                _embed_children_from_batch,
                prepared_batch,
            )

            if batch_index + 1 < len(batches):
                metadata_future = loop.run_in_executor(
                    None,
                    _prepare_batch_metadata,
                    batches[batch_index + 1],
                )

            precomputed_child_rows.extend(await child_embedding_future)

    if uncached_candidates or candidates:
        asyncio.run(_prepare_with_overlap())

    if any(item is None for item in prepared_metadata):
        raise RuntimeError("Failed to build chunk metadata for all chunks.")

    new_chunks: list[DocumentChunk] = []
    cached_payloads: list[tuple[str, str]] = []
    for item in candidates:
        chunk_index = int(item["chunk_index"])
        structured_metadata = prepared_metadata[chunk_index]
        if structured_metadata is None:
            continue

        text = str(item["chunk_text"])
        source_page = _to_int(item["source_page"])
        source_kind = str(item["source_kind"])
        serialized = _serialize_source_metadata(structured_metadata)
        if serialized is not None:
            cached_payloads.append((str(item["fingerprint"]), serialized))

        new_chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=text,
                source_page=source_page,
                source_kind=source_kind,
                source_metadata_json=serialized,
            )
        )

    return new_chunks, cached_payloads, precomputed_child_rows


def _write_document_chunks(
    *,
    db: Session,
    document_id: int,
    new_chunks: list[DocumentChunk],
) -> None:
    write_attempts = 3
    for attempt in range(write_attempts):
        try:
            target_document = db.get(Document, document_id)
            if target_document is None:
                raise RuntimeError("Document not found while writing chunks.")

            _emit_progress(
                "[embed_document][bg] Writing chunks to DB document_id=%s (attempt=%d/%d, chunks=%d)",
                document_id,
                attempt + 1,
                write_attempts,
                len(new_chunks),
            )
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete(
                synchronize_session=False
            )

            if new_chunks:
                db.add_all(new_chunks)

            db.commit()
            _emit_progress("[embed_document][bg] DB write committed for document_id=%s", document_id)
            return
        except OperationalError as exc:
            db.rollback()
            if "database is locked" not in str(exc).lower() or attempt == write_attempts - 1:
                raise RuntimeError(
                    "Database is busy while embedding document. Please retry after a few seconds."
                ) from exc
            time.sleep(0.4 * (attempt + 1))


def _save_index_state(
    *,
    db: Session,
    document_id: int,
    file_hash: str,
    indexed_parent_chunks: int,
    indexed_child_chunks: int,
) -> None:
    write_attempts = 3
    for attempt in range(write_attempts):
        try:
            document = db.get(Document, document_id)
            if document is None:
                return

            document.status = "embedded" if indexed_parent_chunks > 0 else "uploaded"
            db.add(document)

            index_state = db.get(DocumentIndexState, document_id)
            if index_state is None:
                index_state = DocumentIndexState(
                    document_id=document_id,
                    file_hash=file_hash,
                )

            index_state.file_hash = file_hash
            index_state.indexed_parent_chunks = indexed_parent_chunks
            index_state.indexed_child_chunks = indexed_child_chunks
            index_state.indexed_at = datetime.utcnow()
            db.add(index_state)
            db.commit()
            return
        except OperationalError as exc:
            db.rollback()
            if "database is locked" not in str(exc).lower() or attempt == write_attempts - 1:
                raise RuntimeError(
                    "Database is busy while saving index state. Please retry after a few seconds."
                ) from exc
            time.sleep(0.4 * (attempt + 1))


def _run_document_indexing_job(document_id: int, file_hash: str) -> None:
    db = SessionLocal()
    started_at = time.perf_counter()

    try:
        document = db.get(Document, document_id)
        if document is None:
            _emit_progress("[embed_document][bg] Skip missing document_id=%s", document_id)
            return

        original_filename = document.original_filename
        file_path = settings.uploads_dir / document.stored_filename
        if not file_path.exists():
            raise RuntimeError(f"Stored file does not exist for document_id={document_id}")

        _emit_progress(
            "[embed_document][bg] Start indexing document_id=%s file='%s'",
            document_id,
            original_filename,
        )

        new_chunks = _build_document_chunks_for_indexing(
            db=db,
            document_id=document_id,
            original_filename=original_filename,
            file_path=file_path,
            file_hash=file_hash,
        )
        chunk_rows, metadata_cache_payloads, precomputed_child_rows = new_chunks

        _write_document_chunks(
            db=db,
            document_id=document_id,
            new_chunks=chunk_rows,
        )

        _save_metadata_cache(
            db=db,
            document_id=document_id,
            file_hash=file_hash,
            cached_payloads=metadata_cache_payloads,
        )

        document_chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.id.asc())
            .all()
        )
        db.expunge_all()
        db.rollback()

        if document_chunks:
            chunk_by_index = {int(chunk.chunk_index): chunk for chunk in document_chunks}
            child_documents = []
            child_vectors: list[list[float]] = []

            for row in precomputed_child_rows:
                chunk_index = _to_int(row.get("chunk_index"))
                if chunk_index is None:
                    continue

                parent_chunk = chunk_by_index.get(chunk_index)
                if parent_chunk is None:
                    continue

                child_documents.append(
                    {
                        "page_content": str(row.get("child_text") or ""),
                        "metadata": {
                            "document_id": parent_chunk.document_id,
                            "chunk_id": parent_chunk.id,
                            "parent_chunk_id": parent_chunk.id,
                            "chunk_index": parent_chunk.chunk_index,
                            "source_page": parent_chunk.source_page,
                            "source_kind": parent_chunk.source_kind,
                            "source_metadata": row.get("source_metadata") if isinstance(row.get("source_metadata"), dict) else {},
                            "child_type": str(row.get("child_type") or "summary"),
                            "child_index": _to_int(row.get("child_index")) or 0,
                        },
                    }
                )

                vector = row.get("vector")
                if isinstance(vector, list):
                    child_vectors.append(vector)

            if len(child_documents) != len(child_vectors):
                raise RuntimeError("Precomputed child vectors are incomplete for indexing.")

            from langchain_core.documents import Document as LCDocument

            indexed_count = upsert_child_documents(
                [
                    LCDocument(
                        page_content=item["page_content"],
                        metadata=item["metadata"],
                    )
                    for item in child_documents
                ],
                purge_document_ids=[document_id],
                precomputed_vectors=child_vectors,
            )
        else:
            delete_vectors_by_document_id(document_id)
            indexed_count = 0

        _save_index_state(
            db=db,
            document_id=document_id,
            file_hash=file_hash,
            indexed_parent_chunks=len(document_chunks),
            indexed_child_chunks=indexed_count,
        )

        elapsed = time.perf_counter() - started_at
        _emit_progress(
            "[embed_document][bg] Completed document_id=%s chunks_created=%s indexed_children=%s elapsed=%.2fs",
            document_id,
            len(chunk_rows),
            indexed_count,
            elapsed,
        )
    except Exception as exc:
        db.rollback()
        _emit_progress(
            "[embed_document][bg] Failed document_id=%s root_cause=%s",
            document_id,
            exc,
        )

        try:
            failed_document = db.get(Document, document_id)
            if failed_document is not None:
                failed_document.status = "index_failed"
                db.add(failed_document)
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()


@router.get("", response_model=list[DocumentRead])
def list_documents(
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> list[DocumentRead]:
    """Return all uploaded documents with chunk counters."""

    rows = (
        db.query(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [_to_document_read(doc, int(count)) for doc, count in rows]


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> DocumentRead:
    """Upload one document into storage and persist metadata in database."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    safe_original_name = Path(file.filename).name
    generated_name = f"{uuid.uuid4().hex}_{safe_original_name}"
    target_path = settings.uploads_dir / generated_name

    with target_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    document = Document(
        title=(title or safe_original_name).strip() or safe_original_name,
        original_filename=safe_original_name,
        stored_filename=generated_name,
        content_type=file.content_type,
        status="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    return _to_document_read(document, chunk_count=0)


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> DocumentRead:
    """Return one document metadata by id."""

    row = (
        db.query(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)
        .filter(Document.id == document_id)
        .group_by(Document.id)
        .first()
    )

    if row is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    document, chunk_count = row
    return _to_document_read(document, int(chunk_count))


@router.get("/{document_id}/chunks", response_model=DocumentChunkListResponse)
def list_document_chunks(
    document_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> DocumentChunkListResponse:
    """Return paginated chunk list with full content and source metadata."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    total_chunks = (
        db.query(func.count(DocumentChunk.id))
        .filter(DocumentChunk.document_id == document_id)
        .scalar()
        or 0
    )

    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return DocumentChunkListResponse(
        document_id=document_id,
        total_chunks=int(total_chunks),
        offset=offset,
        limit=limit,
        items=[_to_document_chunk_read(chunk) for chunk in chunks],
    )


@router.put("/{document_id}", response_model=DocumentRead)
def update_document(
    document_id: int,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> DocumentRead:
    """Update document title."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    document.title = payload.title.strip()
    db.add(document)
    db.commit()
    db.refresh(document)

    chunk_count = db.query(func.count(DocumentChunk.id)).filter(DocumentChunk.document_id == document_id).scalar() or 0
    return _to_document_read(document, int(chunk_count))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> None:
    """Delete one document from DB, source storage, and VectorDB."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        delete_vectors_by_document_id(document_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to delete vectors for document_id={document_id}. Root cause: {exc}",
        ) from exc

    file_path = settings.uploads_dir / document.stored_filename

    try:
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete(
            synchronize_session=False
        )
        db.delete(document)
        db.commit()
    except OperationalError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503,
            detail="Database is busy while deleting document. Please retry after a few seconds.",
        ) from exc

    if file_path.exists():
        file_path.unlink()

    _emit_progress(
        "[delete_document] Deleted document_id=%s from DB and VectorDB.",
        document_id,
    )


@router.post(
    "/{document_id}/embed",
    response_model=EmbedDocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def embed_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> EmbedDocumentResponse:
    """Queue one document for background HyQ + embedding indexing."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if document.status == "indexing":
        raise HTTPException(status_code=409, detail="Document is already indexing.")

    file_path = settings.uploads_dir / document.stored_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file does not exist.")

    file_hash = _compute_file_hash(file_path)
    existing_index_state = db.get(DocumentIndexState, document_id)
    if (
        existing_index_state is not None
        and existing_index_state.file_hash == file_hash
        and document.status == "embedded"
    ):
        parent_chunk_count = (
            db.query(func.count(DocumentChunk.id))
            .filter(DocumentChunk.document_id == document_id)
            .scalar()
            or 0
        )
        _emit_progress(
            "[embed_document] Skip document_id=%s because file_hash is unchanged and already indexed.",
            document_id,
        )
        return EmbedDocumentResponse(
            document_id=document_id,
            chunks_created=int(parent_chunk_count),
            indexed_chunks=int(existing_index_state.indexed_child_chunks),
        )

    document.status = "indexing"
    db.add(document)
    db.commit()

    background_tasks.add_task(_run_document_indexing_job, document_id, file_hash)
    _emit_progress(
        "[embed_document] Queued background indexing for document_id=%s",
        document_id,
    )

    return EmbedDocumentResponse(
        document_id=document_id,
        chunks_created=0,
        indexed_chunks=0,
    )


@router.post("/reindex", response_model=EmbedDocumentResponse)
def rebuild_global_index(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> EmbedDocumentResponse:
    """Queue pending documents for background incremental indexing."""

    documents = db.query(Document).order_by(Document.id.asc()).all()
    indexed_state_rows = db.query(DocumentIndexState).all()
    state_by_document_id = {row.document_id: row for row in indexed_state_rows}

    pending: list[tuple[int, str]] = []
    for document in documents:
        file_path = settings.uploads_dir / document.stored_filename
        if not file_path.exists():
            continue

        current_hash = _compute_file_hash(file_path)
        tracked = state_by_document_id.get(document.id)
        if tracked is not None and tracked.file_hash == current_hash and document.status == "embedded":
            continue
        pending.append((document.id, current_hash))

    _emit_progress(
        "[rebuild_global_index] Pending documents for incremental indexing: %d",
        len(pending),
    )

    if not pending:
        return EmbedDocumentResponse(
            document_id=0,
            chunks_created=0,
            indexed_chunks=0,
        )

    queued_documents = 0

    for document_id, file_hash in pending:
        target_document = db.get(Document, document_id)
        if target_document is None:
            continue
        if target_document.status == "indexing":
            continue

        target_document.status = "indexing"
        db.add(target_document)
        background_tasks.add_task(_run_document_indexing_job, document_id, file_hash)
        queued_documents += 1

    db.commit()
    _emit_progress(
        "[rebuild_global_index] Queued %d documents for background indexing.",
        queued_documents,
    )

    return EmbedDocumentResponse(
        document_id=0,
        chunks_created=queued_documents,
        indexed_chunks=0,
    )
