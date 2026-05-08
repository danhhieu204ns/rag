from __future__ import annotations

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
from ..core.request_logger import request_logging_context
from ..db import SessionLocal, get_db
from ..models import ChunkMetadataCache, Document, DocumentChunk, DocumentIndexState
from ..schemas import (
    DocumentChunkListResponse,
    DocumentChunkRead,
    DocumentRead,
    DocumentUpdate,
    EmbedDocumentResponse,
    ParseDocumentResponse,
)
from ..services.ingestion_client import (
    build_index_bundle,
    parse_source_to_markdown,
    upsert_index_bundle,
)
from .auth import require_admin
from ..models import User as AdminUser
from ..services.rag_runtime import delete_vectors_by_document_id
from ..services.rag.utils import _json_safe_value, _to_int

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


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


def _queue_full_indexing_job(
    *,
    document: Document,
    document_id: int,
    background_tasks: BackgroundTasks,
    log_prefix: str,
) -> EmbedDocumentResponse:
    if document.status == "indexing":
        raise HTTPException(status_code=409, detail="Document is already indexing.")

    logger.info("[%s] Queueing background indexing for document_id=%s status=%s", log_prefix, document_id, document.status)
    background_tasks.add_task(_run_full_indexing_job, document_id)
    logger.info("[%s] queued document_id=%s", log_prefix, document_id)
    return EmbedDocumentResponse(
        document_id=document_id,
        chunks_created=0,
        indexed_chunks=0,
    )


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
    del db, original_filename, file_hash
    payload = build_index_bundle(file_path)
    parent_chunks = payload.get("parent_chunks")
    child_rows = payload.get("child_rows")
    if not isinstance(parent_chunks, list) or not isinstance(child_rows, list):
        raise RuntimeError("Ingestion index build response is invalid.")

    new_chunks: list[DocumentChunk] = []
    for idx, item in enumerate(parent_chunks):
        if not isinstance(item, dict):
            continue
        chunk_index = _to_int(item.get("chunk_index"))
        if chunk_index is None:
            chunk_index = idx
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        source_page = _to_int(item.get("source_page"))
        source_kind = str(item.get("source_kind") or "text_chunk")
        source_metadata = item.get("source_metadata") if isinstance(item.get("source_metadata"), dict) else {}
        serialized = _serialize_source_metadata(source_metadata)
        new_chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=content,
                source_page=source_page,
                source_kind=source_kind,
                source_metadata_json=serialized,
            )
        )

    precomputed_child_rows = [row for row in child_rows if isinstance(row, dict)]
    cached_payloads: list[tuple[str, str]] = []
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

            logger.info(
                "[index] writing_chunks document_id=%s attempt=%d/%d chunks=%d",
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

            document.status = "embedded" if indexed_parent_chunks > 0 else "parsed"
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


def _run_full_indexing_job(
    document_id: int,
) -> None:
    """Unified background task: Parse + Chunking + Metadata + Vectors."""
    logger.info("[process_document] Background indexing job started for document_id=%s", document_id)
    with request_logging_context("index", doc=document_id) as log:
        try:
            _do_full_indexing_job(document_id=document_id, log=log)
        finally:
            logger.info("[process_document] Background indexing job finished for document_id=%s", document_id)


def _do_full_indexing_job(
    document_id: int,
    log: Any,
) -> None:
    db = SessionLocal()
    started_at = time.perf_counter()

    try:
        log.step_start("init_indexing", doc_id=document_id)
        document = db.get(Document, document_id)
        if document is None:
            log.step_fail("init_indexing", "Document not found")
            return

        document.status = "indexing"
        db.add(document)
        db.commit()
        log.step_done("init_indexing")

        original_filename = document.original_filename
        file_path = settings.uploads_dir / document.stored_filename
        if not file_path.exists():
            raise RuntimeError(f"Stored file does not exist for document_id={document_id}")

        file_hash = _compute_file_hash(file_path)

        # --- Indexing (Parse + split via ingestion service, then metadata + vectors) ---
        log.step_start("build_document_chunks")
        new_chunks_data = _build_document_chunks_for_indexing(
            db=db,
            document_id=document_id,
            original_filename=original_filename,
            file_path=file_path,
            file_hash=file_hash,
        )
        chunk_rows, metadata_cache_payloads, precomputed_child_rows = new_chunks_data
        log.step_done(
            "build_document_chunks",
            parent_chunks=len(chunk_rows),
            child_rows=len(precomputed_child_rows),
        )

        log.step_start("write_db_chunks", count=len(chunk_rows))
        _write_document_chunks(
            db=db,
            document_id=document_id,
            new_chunks=chunk_rows,
        )
        log.step_done("write_db_chunks")

        log.step_start("save_metadata_cache", count=len(metadata_cache_payloads))
        _save_metadata_cache(
            db=db,
            document_id=document_id,
            file_hash=file_hash,
            cached_payloads=metadata_cache_payloads,
        )
        log.step_done("save_metadata_cache")

        log.step_start("prepare_indexing_payload")
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
            upsert_rows: list[dict[str, Any]] = []

            for row in precomputed_child_rows:
                chunk_index = _to_int(row.get("chunk_index"))
                if chunk_index is None:
                    continue
                parent_chunk = chunk_by_index.get(chunk_index)
                if parent_chunk is None:
                    continue

                upsert_rows.append(
                    {
                        "document_id": parent_chunk.document_id,
                        "parent_chunk_id": parent_chunk.id,
                        "source_page": parent_chunk.source_page,
                        "child_type": str(row.get("child_type") or "summary"),
                        "child_index": _to_int(row.get("child_index")) or 0,
                        "child_text": str(row.get("child_text") or ""),
                        "source_metadata": (
                            row.get("source_metadata")
                            if isinstance(row.get("source_metadata"), dict)
                            else {}
                        ),
                    }
                )
            log.step_done("prepare_indexing_payload", count=len(upsert_rows))

            log.step_start("upsert_via_indexing_service", count=len(upsert_rows))
            indexed_count = upsert_index_bundle(document_id=document_id, child_rows=upsert_rows)
            log.step_done("upsert_via_indexing_service", indexed_count=indexed_count)
        else:
            log.step_start("delete_vectors")
            delete_vectors_by_document_id(document_id)
            indexed_count = 0
            log.step_done("delete_vectors")

        log.step_start("finalize_state")
        _save_index_state(
            db=db,
            document_id=document_id,
            file_hash=file_hash,
            indexed_parent_chunks=len(document_chunks),
            indexed_child_chunks=indexed_count,
        )
        log.step_done("finalize_state")

        elapsed = time.perf_counter() - started_at
        log.info("Full indexing completed successfully in %.2fs", elapsed)

    except Exception as exc:
        db.rollback()
        log.error("Indexing failed: %s", exc)
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



@router.post("/{document_id}/process", response_model=EmbedDocumentResponse, status_code=status.HTTP_202_ACCEPTED)
def process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> EmbedDocumentResponse:
    """Unified endpoint: Start the full indexing process (Parse + Embed)."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    return _queue_full_indexing_job(
        document=document,
        document_id=document_id,
        background_tasks=background_tasks,
        log_prefix="process_document",
    )


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
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
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

    logger.info("[delete_document] deleted document_id=%s", document_id)


@router.post(
    "/{document_id}/parse",
    response_model=ParseDocumentResponse,
)
def parse_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> ParseDocumentResponse:
    """Validate parsing by calling ingestion service directly."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = settings.uploads_dir / document.stored_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file does not exist.")

    try:
        markdown, source_parser, source_type = parse_source_to_markdown(file_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not markdown.strip():
        raise HTTPException(status_code=422, detail="Parsed markdown is empty.")

    if document.status != "embedded":
        document.status = "parsed"
        db.add(document)
        db.commit()

    logger.info(
        "[parse_document] parsed document_id=%s parser=%s type=%s chars=%d",
        document_id,
        source_parser,
        source_type,
        len(markdown),
    )

    return ParseDocumentResponse(
        document_id=document_id,
        parsed_markdown_path=f"ingestion://v1/parse/{document_id}",
        parser=source_parser,
        source_type=source_type,
        reused=False,
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
    """Queue one document for background indexing."""

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
        logger.info("[embed_document] skip unchanged document_id=%s", document_id)
        return EmbedDocumentResponse(
            document_id=document_id,
            chunks_created=int(parent_chunk_count),
            indexed_chunks=int(existing_index_state.indexed_child_chunks),
        )

    document.status = "indexing"
    db.add(document)
    db.commit()

    return _queue_full_indexing_job(
        document=document,
        document_id=document_id,
        background_tasks=background_tasks,
        log_prefix="embed_document",
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

    logger.info("[rebuild_global_index] pending_documents=%d", len(pending))

    if not pending:
        return EmbedDocumentResponse(
            document_id=0,
            chunks_created=0,
            indexed_chunks=0,
        )

    queued_documents = 0

    for document_id, _ in pending:
        target_document = db.get(Document, document_id)
        if target_document is None:
            continue
        if target_document.status == "indexing":
            continue

        target_document.status = "indexing"
        db.add(target_document)
        background_tasks.add_task(_run_full_indexing_job, document_id)
        queued_documents += 1

    db.commit()
    logger.info("[rebuild_global_index] queued_documents=%d", queued_documents)

    return EmbedDocumentResponse(
        document_id=0,
        chunks_created=queued_documents,
        indexed_chunks=0,
    )
