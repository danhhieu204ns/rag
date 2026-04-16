from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class DocumentUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class DocumentRead(DocumentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    content_type: str | None
    status: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class EmbedDocumentResponse(BaseModel):
    document_id: int
    chunks_created: int
    indexed_chunks: int


class SourceItem(BaseModel):
    document_id: int | None = None
    chunk_id: int | None = None
    chunk_index: int | None = None
    page: int | None = Field(default=None, ge=1)
    source_kind: str | None = None
    source_metadata: dict[str, Any] | None = None
    excerpt: str


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    sources: list[SourceItem] = Field(default_factory=list)
    created_at: datetime


class ChatQueryRequest(BaseModel):
    session_id: int | None = None
    message: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[int] | None = None


class ChatQueryResponse(BaseModel):
    session_id: int
    answer: str
    sources: list[SourceItem]
