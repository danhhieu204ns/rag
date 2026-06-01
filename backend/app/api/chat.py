from __future__ import annotations

import json
import logging
import re
from datetime import datetime
import time
import unicodedata

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..core.query_logger import query_logging_context
from ..core.request_logger import request_logging_context, get_request_logger
from ..db import get_db
from ..models import ChatMessage, ChatSession, Document, DocumentChunk, User
from ..schemas import (
    CitationDocumentRead,
    CitationSourceRead,
    ChatMessageRead,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionCreate,
    ChatSessionRead,
    DocumentChunkRead,
    SourceItem,
)
from .auth import get_current_user
from ..services.rag.orchestrator import classify_query
from ..services.rag_runtime import (
    build_sources,
    parse_sources,
    similarity_search,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

_FAST_GREETING_INPUTS = {
    "hi",
    "hello",
    "hey",
    "helo",
    "alo",
    "aloo",
    "chao",
    "xin chao",
    "chao a",
    "chao ban",
    "chao ad",
    "xin chao ban",
    "xin chao a",
    "hello bot",
    "hi bot",
}
_FAST_THANKS_INPUTS = {
    "cam on",
    "cam on a",
    "cam on ban",
    "cam on nhe",
    "thanks",
    "thank",
    "thank you",
    "thank u",
    "ok thanks",
    "oke thanks",
}
_FAST_IDENTITY_INPUTS = {
    "ban la ai",
    "ban la gi",
    "ten ban la gi",
    "who are you",
    "what are you",
}


def _emit_query_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    get_request_logger().info(text)


def _preview_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_fast_reply_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _fast_chat_reply(user_text: str) -> tuple[str, str, str] | None:
    normalized = _normalize_fast_reply_text(user_text)
    if normalized in _FAST_GREETING_INPUTS:
        return (
            "greeting",
            "Xin chào! Mình là Trợ lý tri thức VTAca. Bạn muốn tra cứu nội dung nào trong tài liệu?",
            "matched greeting fast path",
        )
    if normalized in _FAST_THANKS_INPUTS:
        return (
            "chitchat",
            "Không có gì. Bạn cần mình hỗ trợ thêm nội dung nào?",
            "matched thanks fast path",
        )
    if normalized in _FAST_IDENTITY_INPUTS:
        return (
            "chitchat",
            "Mình là Trợ lý tri thức VTAca, hỗ trợ hỏi đáp và tra cứu thông tin từ tài liệu của hệ thống.",
            "matched identity fast path",
        )
    return None


def _record_fast_path_plan(_qlog, user_text: str, query_type: str, output_mode: str, reason: str) -> None:
    _qlog.record(
        "orchestrator_plan",
        {
            "query_preview": _preview_text(user_text),
            "query_type": query_type,
            "output_mode": output_mode,
            "strategy": "balanced",
            "top_k": None,
            "retrieval_defaults_source": "local_fast_path",
            "max_iterations": 0,
            "expand_query": False,
            "signals": ["low_latency_direct_reply"],
            "orchestrator_source": "local_fast_path",
            "confidence": 1.0,
            "requires_retrieval": False,
            "llm_reason": reason,
            "elapsed_ms": 0.0,
        },
    )


def _build_title_from_first_question(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:255]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _enrich_sources_with_document_name(db: Session, sources: list[dict]) -> list[dict]:
    """Backfill filename in source_metadata when retrieval metadata is missing."""
    if not sources:
        return sources

    document_ids = {
        int(src.get("document_id"))
        for src in sources
        if isinstance(src, dict) and src.get("document_id") is not None
    }
    if not document_ids:
        return sources

    documents = (
        db.query(Document.id, Document.original_filename, Document.title)
        .filter(Document.id.in_(document_ids))
        .all()
    )
    by_id = {int(doc_id): {"original_filename": original_filename, "title": title} for doc_id, original_filename, title in documents}

    for src in sources:
        if not isinstance(src, dict):
            continue
        document_id = src.get("document_id")
        if document_id is None:
            continue
        doc_info = by_id.get(int(document_id))
        if not doc_info:
            continue

        source_metadata = src.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = {}
            src["source_metadata"] = source_metadata

        source_info = source_metadata.get("source_info")
        if not isinstance(source_info, dict):
            source_info = {}
            source_metadata["source_info"] = source_info

        if not source_info.get("file_name"):
            source_info["file_name"] = doc_info["original_filename"] or doc_info["title"]
        source_metadata.setdefault("original_filename", doc_info["original_filename"])
        source_metadata.setdefault("title", doc_info["title"])

    return sources



def _session_to_read(item: ChatSession) -> ChatSessionRead:
    return ChatSessionRead(
        id=item.id,
        title=item.title,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )



def _message_to_read(item: ChatMessage, db: Session | None = None) -> ChatMessageRead:
    parsed_sources = parse_sources(item.sources_json)
    if db is not None:
        parsed_sources = _enrich_sources_with_document_name(db, parsed_sources)
    sources = [SourceItem(**source) for source in parsed_sources]
    return ChatMessageRead(
        id=item.id,
        session_id=item.session_id,
        role=item.role,
        content=item.content,
        sources=sources,
        created_at=item.created_at,
    )


def _parse_chunk_metadata(raw_json: str | None) -> dict:
    if not raw_json:
        return {}
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_positive_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _metadata_heading_path(metadata: dict) -> list[str]:
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list):
        return [str(item) for item in heading_path if str(item).strip()]

    context = metadata.get("context")
    if isinstance(context, dict):
        context_path = context.get("heading_path")
        if isinstance(context_path, list):
            return [str(item) for item in context_path if str(item).strip()]

    return []


def _metadata_section_title(metadata: dict) -> str | None:
    section_title = metadata.get("section_title") or metadata.get("title")
    if section_title is not None and str(section_title).strip():
        return str(section_title).strip()

    context = metadata.get("context")
    if isinstance(context, dict):
        for key in ("h6", "h5", "h4", "h3", "h2", "h1"):
            value = context.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

    heading_path = _metadata_heading_path(metadata)
    return heading_path[-1] if heading_path else None


def _chunk_to_read(chunk: DocumentChunk, source_metadata: dict) -> DocumentChunkRead:
    return DocumentChunkRead(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        source_page=chunk.source_page,
        source_kind=chunk.source_kind,
        source_metadata=source_metadata,
        created_at=chunk.created_at,
    )


@router.get("/source", response_model=CitationSourceRead)
def get_citation_source(
    document_id: int = Query(..., ge=1),
    chunk_id: int | None = Query(default=None, ge=1),
    chunk_index: int | None = Query(default=None, ge=0),
    page: int | None = Query(default=None, ge=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> CitationSourceRead:
    """Return the exact stored chunk behind a chat citation for the source panel."""

    del current_user
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    base_query = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
    chunk: DocumentChunk | None = None
    if chunk_id is not None:
        chunk = base_query.filter(DocumentChunk.id == chunk_id).first()
    if chunk is None and chunk_index is not None:
        chunk = (
            base_query
            .filter(DocumentChunk.chunk_index == chunk_index)
            .order_by(DocumentChunk.id.asc())
            .first()
        )
    if chunk is None and page is not None:
        chunk = (
            base_query
            .filter(DocumentChunk.source_page == page)
            .order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.id.asc())
            .first()
        )

    if chunk is None:
        raise HTTPException(status_code=404, detail="Source chunk not found.")

    source_metadata = _parse_chunk_metadata(chunk.source_metadata_json)
    page_start = (
        _to_positive_int(chunk.source_page)
        or _to_positive_int(source_metadata.get("page_start"))
        or _to_positive_int(source_metadata.get("source_page"))
    )
    page_end = _to_positive_int(source_metadata.get("page_end"))
    file_path = settings.uploads_dir / document.stored_filename

    return CitationSourceRead(
        document=CitationDocumentRead(
            id=document.id,
            title=document.title,
            original_filename=document.original_filename,
            content_type=document.content_type,
            status=document.status,
        ),
        chunk=_chunk_to_read(chunk, source_metadata),
        page=page_start,
        page_end=page_end,
        section_title=_metadata_section_title(source_metadata),
        heading_path=_metadata_heading_path(source_metadata),
        file_available=file_path.exists(),
    )


@router.get("/source/file/{document_id}")
def get_citation_source_file(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """Stream the original uploaded document for the source panel preview."""

    del current_user
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_path = settings.uploads_dir / document.stored_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Stored file does not exist.")

    return FileResponse(
        path=file_path,
        media_type=document.content_type or "application/octet-stream",
        filename=document.original_filename,
        content_disposition_type="inline",
    )


@router.get("/sessions", response_model=list[ChatSessionRead])
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ChatSessionRead]:
    """List chat sessions sorted by recent update."""

    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [_session_to_read(session) for session in sessions]


@router.post("/sessions", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: ChatSessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatSessionRead:
    """Create an empty chat session."""

    title = (payload.title or "New chat").strip() or "New chat"
    session = ChatSession(title=title, user_id=current_user.id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_to_read(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete one chat session and all messages in it."""

    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.delete(session)
    db.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageRead])
def list_messages(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[ChatMessageRead]:
    """Return all messages from selected chat session."""

    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    return [_message_to_read(message, db=db) for message in messages]


@router.post("/query")
def query_chat(
    payload: ChatQueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Run one RAG query, save both user and assistant messages, and stream the response."""

    user_text = payload.message.strip()
    top_k = payload.top_k

    return StreamingResponse(
        _run_query_chat_stream(payload, user_text, top_k, db, current_user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _run_query_chat_stream(
    payload: ChatQueryRequest,
    user_text: str,
    top_k: int | None,
    db: Session,
    current_user: User,
):
    with query_logging_context(
        query=user_text,
        session_id=payload.session_id,
        top_k=top_k,
    ) as _qlog:
        yield from _run_query_chat_stream_inner(payload, user_text, top_k, db, _qlog, current_user)


def _run_query_chat_stream_inner(
    payload: ChatQueryRequest,
    user_text: str,
    top_k: int | None,
    db: Session,
    _qlog,
    current_user: User,
):
    request_started_at = time.perf_counter()

    _emit_query_progress(
        "[chat.query] Start stream request: session_id=%s, top_k=%s, document_filter=%s, message='%s'",
        payload.session_id,
        top_k if top_k is not None else "retrieval_service_default",
        payload.document_ids or [],
        _preview_text(user_text),
    )

    session: ChatSession | None = None
    session_stage_started_at = time.perf_counter()
    if payload.session_id is not None:
        session = db.get(ChatSession, payload.session_id)
        if session is None or session.user_id != current_user.id:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Session not found.'})}\n\n"
            return

    first_question_title = _build_title_from_first_question(user_text)

    if session is None:
        session = ChatSession(title=first_question_title, user_id=current_user.id)
        db.add(session)
        db.commit()
        db.refresh(session)
        _emit_query_progress("[chat.query] Created new session: session_id=%d", session.id)
    else:
        first_user_message_exists = (
            db.query(ChatMessage.id)
            .filter(ChatMessage.session_id == session.id, ChatMessage.role == "user")
            .first()
            is not None
        )
        if not first_user_message_exists:
            session.title = first_question_title
            db.add(session)
            db.commit()
            db.refresh(session)

        _emit_query_progress("[chat.query] Use existing session: session_id=%d", session.id)
    
    _emit_query_progress(
        "[timing][chat.query] DONE step=resolve_session session_id=%d elapsed_ms=%.2f",
        session.id,
        (time.perf_counter() - session_stage_started_at) * 1000,
    )

    save_user_started_at = time.perf_counter()
    user_message = ChatMessage(
        session_id=session.id,
        role="user",
        content=user_text,
        sources_json=None,
    )
    db.add(user_message)
    db.commit()
    _emit_query_progress("[chat.query] Saved user message: session_id=%d", session.id)

    # Send the first event before orchestration/retrieval so the UI stays alive
    # while slower backend steps are running.
    yield ": stream-start\n\n"
    yield _sse({"type": "session", "session_id": session.id})
    yield _sse({"type": "status", "message": "Đang xử lý câu hỏi..."})

    fast_reply = _fast_chat_reply(user_text)
    if fast_reply is not None:
        query_type, answer, fast_reason = fast_reply
        output_mode = payload.output_mode or "qa"
        sources: list[dict] = []
        _record_fast_path_plan(_qlog, user_text, query_type, output_mode, fast_reason)
        _emit_query_progress(
            "[chat.query] Fast direct reply: query_type=%s reason=%s",
            query_type,
            fast_reason,
        )

        yield _sse({"type": "output_mode", "mode": output_mode})
        yield _sse({"type": "sources", "sources": sources})

        answer_started_at = time.perf_counter()
        _qlog.record_generation_start()
        yield _sse({"type": "token", "content": answer})
        _qlog.record_generation_done(len(answer))

        _emit_query_progress(
            "[timing][chat.query] DONE step=fast_direct_answer session_id=%d answer_len=%d elapsed_ms=%.2f",
            session.id,
            len(answer),
            (time.perf_counter() - answer_started_at) * 1000,
        )

        assistant_message = ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            sources_json=json.dumps(sources, ensure_ascii=False),
            created_at=datetime.utcnow(),
        )
        db.add(assistant_message)
        live_session = db.get(ChatSession, session.id)
        if live_session is not None:
            live_session.updated_at = datetime.utcnow()
        db.commit()

        _emit_query_progress("[chat.query] Completed fast direct request: session_id=%d", session.id)
        yield _sse({"type": "done"})
        return

    history_started_at = time.perf_counter()
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    _emit_query_progress("[chat.query] Loaded history messages: count=%d", len(history))

    # ── Orchestrate: classify query + resolve output mode ─────────────────────
    plan = None
    if settings.orchestrator_enabled:
        yield _sse({"type": "status", "message": "Đang phân loại câu hỏi..."})
        explicit_mode = payload.output_mode or None
        plan = classify_query(
            user_text,
            top_k,
            output_mode_override=explicit_mode,  # type: ignore[arg-type]
        )
    output_mode: str = plan.output_mode if plan else (payload.output_mode or "qa")

    retrieved_docs = []
    if plan is None or getattr(plan, "requires_retrieval", True):
        retrieval_started_at = time.perf_counter()
        yield _sse({"type": "status", "message": "Đang tìm kiếm tài liệu liên quan..."})
        retrieved_docs = similarity_search(
            user_text,
            top_k=top_k,
            db=db,
            document_ids=payload.document_ids,
            plan=plan,
        )
        _emit_query_progress("[chat.query] Retrieved context docs: count=%d", len(retrieved_docs))
    else:
        _emit_query_progress("[chat.query] Orchestrator chose direct answer without retrieval")

    sources = _enrich_sources_with_document_name(db, build_sources(retrieved_docs))

    yield _sse({"type": "output_mode", "mode": output_mode})
    yield _sse({"type": "sources", "sources": sources})
    yield _sse({"type": "status", "message": "Đang soạn câu trả lời..."})

    answer_parts = []
    answer_started_at = time.perf_counter()
    _qlog.record_generation_start()

    try:
        from ..services.rag_runtime import generate_answer, generate_answer_stream
        for chunk in generate_answer_stream(
            question=user_text,
            context_docs=retrieved_docs,
            history_messages=history,
            output_mode=output_mode,
        ):
            answer_parts.append(chunk)
            yield _sse({"type": "token", "content": chunk})
    except Exception as exc:  # pragma: no cover
        _emit_query_progress("[chat.query] Generate answer stream failed: %s", exc)
        yield _sse({"type": "error", "detail": str(exc)})
        return

    answer = "".join(answer_parts)
    if not answer.strip():
        _emit_query_progress("[chat.query] Empty streamed answer; retrying with non-stream generation")
        yield _sse({"type": "status", "message": "Đang thử tạo lại câu trả lời..."})
        try:
            answer = generate_answer(
                question=user_text,
                context_docs=retrieved_docs,
                history_messages=history,
                output_mode=output_mode,
            ).strip()
        except Exception as exc:  # pragma: no cover
            _emit_query_progress("[chat.query] Generate answer fallback failed: %s", exc)
            answer = ""

        if answer:
            yield _sse({"type": "token", "content": answer})
        else:
            answer = "Mình chưa tạo được nội dung trả lời cho câu hỏi này. Bạn vui lòng thử gửi lại hoặc diễn đạt cụ thể hơn."
            yield _sse({"type": "error", "detail": answer})
            yield _sse({"type": "done"})
            return

    _qlog.record_generation_done(len(answer))

    _emit_query_progress(
        "[timing][chat.query] DONE step=generate_answer session_id=%d answer_len=%d elapsed_ms=%.2f",
        session.id,
        len(answer),
        (time.perf_counter() - answer_started_at) * 1000,
    )

    save_assistant_started_at = time.perf_counter()
    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources_json=json.dumps(sources, ensure_ascii=False),
        created_at=datetime.utcnow(),
    )
    db.add(assistant_message)
    # Re-fetch session để tránh StaleDataError do object bị expire sau commit trước đó
    live_session = db.get(ChatSession, session.id)
    if live_session is not None:
        live_session.updated_at = datetime.utcnow()
    db.commit()
    
    _emit_query_progress("[chat.query] Completed request: session_id=%d", session.id)
    
    yield _sse({"type": "done"})
