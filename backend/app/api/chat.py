from __future__ import annotations

import json
import logging
from datetime import datetime
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..core.query_logger import query_logging_context
from ..core.request_logger import request_logging_context, get_request_logger
from ..db import get_db
from ..models import ChatMessage, ChatSession, Document, User
from ..schemas import (
    ChatMessageRead,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionCreate,
    ChatSessionRead,
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


def _emit_query_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    get_request_logger().info(text)


def _preview_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


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

    # Send comment prelude early to reduce proxy buffering on some deployments.
    yield ": stream-start\n\n"
    yield _sse({"type": "session", "session_id": session.id})
    yield _sse({"type": "output_mode", "mode": output_mode})
    yield _sse({"type": "sources", "sources": sources})

    answer_parts = []
    answer_started_at = time.perf_counter()
    _qlog.record_generation_start()

    try:
        from ..services.rag_runtime import generate_answer_stream
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
