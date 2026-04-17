from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import OperationalError
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..db import get_db
from ..models import AdminUser, Document, DocumentChunk
from ..schemas import DocumentChunkListResponse, DocumentChunkRead, DocumentRead, DocumentUpdate, EmbedDocumentResponse
from ..services.chunk_metadata import build_structured_chunk_metadata
from ..services.document_processing import load_source_documents, split_source_documents
from .auth import require_admin
from ..services.rag_runtime import delete_vectors_by_document_id, rebuild_index_from_chunks

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


@router.post("/{document_id}/embed", response_model=EmbedDocumentResponse)
def embed_document(
    document_id: int,
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> EmbedDocumentResponse:
    """Create chunks for one document and rebuild global Qdrant index."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    original_filename = document.original_filename
    stored_filename = document.stored_filename
    started_at = time.perf_counter()
    _emit_progress(
        "[embed_document] Start embedding document_id=%s file='%s'",
        document_id,
        original_filename,
    )
    # Release transaction early before expensive OCR/chunking work.
    db.rollback()

    file_path = settings.uploads_dir / stored_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file does not exist.")

    try:
        loaded_documents = load_source_documents(file_path)
        _emit_progress(
            "[embed_document] Loaded %d source document blocks for document_id=%s",
            len(loaded_documents),
            document_id,
        )
        split_documents = split_source_documents(
            loaded_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        _emit_progress(
            "[embed_document] Split into %d chunks for document_id=%s",
            len(split_documents),
            document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidates: list[tuple[int, str, dict[str, Any], int | None, str]] = []
    for item in split_documents:
        text = item.page_content.strip()
        if not text:
            continue

        item_metadata = dict(item.metadata or {})
        source_page = _extract_source_page(item_metadata)
        source_kind = _extract_source_kind(item_metadata, file_path.suffix.lower())
        candidates.append((len(candidates), text, item_metadata, source_page, source_kind))

    metadata_workers = max(1, settings.metadata_max_workers)
    _emit_progress(
        "[embed_document] Preparing metadata for %d chunks with workers=%d (document_id=%s)",
        len(candidates),
        metadata_workers,
        document_id,
    )

    prepared_rows: list[tuple[str, int | None, str, dict[str, Any]] | None] = [None] * len(candidates)

    def _build_metadata_row(
        candidate: tuple[int, str, dict[str, Any], int | None, str],
    ) -> tuple[int, tuple[str, int | None, str, dict[str, Any]]]:
        chunk_index, text, item_metadata, source_page, source_kind = candidate
        structured_metadata = build_structured_chunk_metadata(
            document_id=document_id,
            chunk_index=chunk_index,
            file_name=original_filename,
            source_page=source_page,
            raw_metadata=item_metadata,
            chunk_text=text,
        )
        return chunk_index, (text, source_page, source_kind, structured_metadata)

    if metadata_workers <= 1 or len(candidates) <= 1:
        for processed_count, candidate in enumerate(candidates, start=1):
            chunk_index, payload = _build_metadata_row(candidate)
            prepared_rows[chunk_index] = payload
            if processed_count % 20 == 0 or processed_count == len(candidates):
                _emit_progress(
                    "[embed_document] Prepared chunk metadata %d/%d for document_id=%s",
                    processed_count,
                    len(candidates),
                    document_id,
                )
    else:
        with ThreadPoolExecutor(max_workers=metadata_workers) as executor:
            futures = [executor.submit(_build_metadata_row, candidate) for candidate in candidates]
            processed_count = 0

            for future in as_completed(futures):
                chunk_index, payload = future.result()
                prepared_rows[chunk_index] = payload
                processed_count += 1

                if processed_count % 20 == 0 or processed_count == len(candidates):
                    _emit_progress(
                        "[embed_document] Prepared chunk metadata %d/%d for document_id=%s",
                        processed_count,
                        len(candidates),
                        document_id,
                    )

    if any(item is None for item in prepared_rows):
        raise HTTPException(
            status_code=500,
            detail="Failed to build chunk metadata for all chunks.",
        )

    new_chunks: list[DocumentChunk] = []
    for chunk_index, payload in enumerate(prepared_rows):
        if payload is None:
            continue

        text, source_page, source_kind, structured_metadata = payload
        new_chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_index,
                content=text,
                source_page=source_page,
                source_kind=source_kind,
                source_metadata_json=_serialize_source_metadata(structured_metadata),
            )
        )

    write_attempts = 3
    for attempt in range(write_attempts):
        try:
            target_document = db.get(Document, document_id)
            if target_document is None:
                raise HTTPException(status_code=404, detail="Document not found.")

            _emit_progress(
                "[embed_document] Writing chunks to DB document_id=%s (attempt=%d/%d, chunks=%d)",
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

            target_document.status = "embedded" if new_chunks else "uploaded"
            db.add(target_document)
            db.commit()
            _emit_progress("[embed_document] DB write committed for document_id=%s", document_id)
            break
        except OperationalError as exc:
            db.rollback()
            _emit_progress(
                "[embed_document] DB locked on document_id=%s (attempt=%d/%d)",
                document_id,
                attempt + 1,
                write_attempts,
            )
            if "database is locked" not in str(exc).lower() or attempt == write_attempts - 1:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Database is busy while embedding document. "
                        "Please retry after a few seconds."
                    ),
                ) from exc
            time.sleep(0.4 * (attempt + 1))

    all_chunks = db.query(DocumentChunk).order_by(DocumentChunk.id.asc()).all()
    _emit_progress(
        "[embed_document] Rebuilding Qdrant from %d parent chunks after embedding document_id=%s",
        len(all_chunks),
        document_id,
    )
    db.expunge_all()
    db.rollback()
    try:
        indexed_count = rebuild_index_from_chunks(all_chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to rebuild Qdrant collection using Ollama embeddings. "
                f"base_url={settings.ollama_base_url}, embedding_model={settings.embedding_model_name}. "
                f"Root cause: {exc}"
            ),
        ) from exc

    elapsed = time.perf_counter() - started_at
    _emit_progress(
        "[embed_document] Completed document_id=%s chunks_created=%s indexed_children=%s elapsed=%.2fs",
        document_id,
        len(new_chunks),
        indexed_count,
        elapsed,
    )

    return EmbedDocumentResponse(
        document_id=document_id,
        chunks_created=len(new_chunks),
        indexed_chunks=indexed_count,
    )


@router.post("/reindex", response_model=EmbedDocumentResponse)
def rebuild_global_index(
    db: Session = Depends(get_db),
    _: AdminUser = Depends(require_admin),
) -> EmbedDocumentResponse:
    """Rebuild global vector index from all saved chunks."""

    all_chunks = db.query(DocumentChunk).order_by(DocumentChunk.id.asc()).all()
    _emit_progress(
        "[rebuild_global_index] Triggered with %d parent chunks.",
        len(all_chunks),
    )
    db.expunge_all()
    db.rollback()
    try:
        indexed_count = rebuild_index_from_chunks(all_chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to rebuild Qdrant collection using Ollama embeddings. "
                f"base_url={settings.ollama_base_url}, embedding_model={settings.embedding_model_name}. "
                f"Root cause: {exc}"
            ),
        ) from exc
    return EmbedDocumentResponse(
        document_id=0,
        chunks_created=indexed_count,
        indexed_chunks=indexed_count,
    )
