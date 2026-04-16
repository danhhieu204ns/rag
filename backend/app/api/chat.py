from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..db import get_db
from ..models import ChatMessage, ChatSession
from ..schemas import (
    ChatMessageRead,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionCreate,
    ChatSessionRead,
    SourceItem,
)
from ..services.rag_runtime import (
    build_sources,
    generate_answer,
    parse_sources,
    similarity_search,
)

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _emit_query_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    logger.info(text)
    print(text, flush=True)


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



def _session_to_read(item: ChatSession) -> ChatSessionRead:
    return ChatSessionRead(
        id=item.id,
        title=item.title,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )



def _message_to_read(item: ChatMessage) -> ChatMessageRead:
    sources = [SourceItem(**source) for source in parse_sources(item.sources_json)]
    return ChatMessageRead(
        id=item.id,
        session_id=item.session_id,
        role=item.role,
        content=item.content,
        sources=sources,
        created_at=item.created_at,
    )


@router.get("/sessions", response_model=list[ChatSessionRead])
def list_sessions(db: Session = Depends(get_db)) -> list[ChatSessionRead]:
    """List chat sessions sorted by recent update."""

    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [_session_to_read(session) for session in sessions]


@router.post("/sessions", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_session(payload: ChatSessionCreate, db: Session = Depends(get_db)) -> ChatSessionRead:
    """Create an empty chat session."""

    title = (payload.title or "New chat").strip() or "New chat"
    session = ChatSession(title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return _session_to_read(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: int, db: Session = Depends(get_db)) -> None:
    """Delete one chat session and all messages in it."""

    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    db.delete(session)
    db.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageRead])
def list_messages(session_id: int, db: Session = Depends(get_db)) -> list[ChatMessageRead]:
    """Return all messages from selected chat session."""

    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    return [_message_to_read(message) for message in messages]


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(payload: ChatQueryRequest, db: Session = Depends(get_db)) -> ChatQueryResponse:
    """Run one RAG query, save both user and assistant messages."""

    user_text = payload.message.strip()
    top_k = payload.top_k or settings.retriever_k

    _emit_query_progress(
        "[chat.query] Start request: session_id=%s, top_k=%d, document_filter=%s, message='%s'",
        payload.session_id,
        top_k,
        payload.document_ids or [],
        _preview_text(user_text),
    )

    session: ChatSession | None = None
    if payload.session_id is not None:
        session = db.get(ChatSession, payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")

    first_question_title = _build_title_from_first_question(user_text)

    if session is None:
        session = ChatSession(title=first_question_title)
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

    user_message = ChatMessage(
        session_id=session.id,
        role="user",
        content=user_text,
        sources_json=None,
    )
    db.add(user_message)
    db.commit()
    _emit_query_progress("[chat.query] Saved user message: session_id=%d", session.id)

    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all()
    )
    _emit_query_progress("[chat.query] Loaded history messages: count=%d", len(history))

    retrieved_docs = similarity_search(
        user_text,
        top_k=top_k,
        db=db,
        document_ids=payload.document_ids,
    )
    _emit_query_progress("[chat.query] Retrieved context docs: count=%d", len(retrieved_docs))

    try:
        answer = generate_answer(
            question=user_text,
            context_docs=retrieved_docs,
            history_messages=history,
        )
        _emit_query_progress("[chat.query] Generated answer: length=%d chars", len(answer))
    except Exception as exc:  # pragma: no cover - external API/network failure
        _emit_query_progress("[chat.query] Generate answer failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = build_sources(retrieved_docs)
    _emit_query_progress("[chat.query] Built sources payload: count=%d", len(sources))

    assistant_message = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources_json=json.dumps(sources, ensure_ascii=False),
        created_at=datetime.utcnow(),
    )
    db.add(assistant_message)
    session.updated_at = datetime.utcnow()
    db.add(session)
    db.commit()
    _emit_query_progress("[chat.query] Completed request: session_id=%d", session.id)

    return ChatQueryResponse(
        session_id=session.id,
        answer=answer,
        sources=[SourceItem(**item) for item in sources],
    )
