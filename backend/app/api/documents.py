from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..db import get_db
from ..models import Document, DocumentChunk
from ..schemas import DocumentRead, DocumentUpdate, EmbedDocumentResponse
from ..services.document_processing import load_source_documents, split_source_documents
from ..services.rag_runtime import rebuild_index_from_chunks

router = APIRouter(prefix="/documents", tags=["documents"])



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


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentRead]:
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
def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentRead:
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


@router.put("/{document_id}", response_model=DocumentRead)
def update_document(
    document_id: int,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
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
def delete_document(document_id: int, db: Session = Depends(get_db)) -> None:
    """Delete document metadata, source file, and rebuild global vector index."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = settings.uploads_dir / document.stored_filename
    if file_path.exists():
        file_path.unlink()

    db.delete(document)
    db.commit()

    remaining_chunks = db.query(DocumentChunk).order_by(DocumentChunk.id.asc()).all()
    rebuild_index_from_chunks(remaining_chunks)


@router.post("/{document_id}/embed", response_model=EmbedDocumentResponse)
def embed_document(document_id: int, db: Session = Depends(get_db)) -> EmbedDocumentResponse:
    """Create chunks for one document and rebuild global FAISS index."""

    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = settings.uploads_dir / document.stored_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file does not exist.")

    try:
        loaded_documents = load_source_documents(file_path)
        split_documents = split_source_documents(
            loaded_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunking_method=settings.chunking_method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()

    new_chunks: list[DocumentChunk] = []
    for index, item in enumerate(split_documents):
        text = item.page_content.strip()
        if not text:
            continue
        new_chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                content=text,
            )
        )

    if new_chunks:
        db.add_all(new_chunks)

    document.status = "embedded" if new_chunks else "uploaded"
    db.add(document)
    db.commit()

    all_chunks = db.query(DocumentChunk).order_by(DocumentChunk.id.asc()).all()
    try:
        indexed_count = rebuild_index_from_chunks(all_chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to rebuild embedding index using Ollama. "
                f"base_url={settings.ollama_base_url}, embedding_model={settings.embedding_model_name}. "
                f"Root cause: {exc}"
            ),
        ) from exc

    return EmbedDocumentResponse(
        document_id=document_id,
        chunks_created=len(new_chunks),
        indexed_chunks=indexed_count,
    )


@router.post("/reindex", response_model=EmbedDocumentResponse)
def rebuild_global_index(db: Session = Depends(get_db)) -> EmbedDocumentResponse:
    """Rebuild global vector index from all saved chunks."""

    all_chunks = db.query(DocumentChunk).order_by(DocumentChunk.id.asc()).all()
    try:
        indexed_count = rebuild_index_from_chunks(all_chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to rebuild embedding index using Ollama. "
                f"base_url={settings.ollama_base_url}, embedding_model={settings.embedding_model_name}. "
                f"Root cause: {exc}"
            ),
        ) from exc
    return EmbedDocumentResponse(
        document_id=0,
        chunks_created=indexed_count,
        indexed_chunks=indexed_count,
    )
