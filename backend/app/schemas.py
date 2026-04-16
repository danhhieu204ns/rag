from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AdminUserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    is_active: bool


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


class ChunkSourceInfo(BaseModel):
    file_name: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    doc_type: str | None = None


class ChunkContextMetadata(BaseModel):
    h2: str | None = None
    h3: str | None = None


class ChunkSearchOptimization(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    document_codes: list[str] = Field(default_factory=list)


class ChunkAdminTags(BaseModel):
    security_level: str | None = None
    department: str | None = None


class ChunkHyQMetadata(BaseModel):
    summary: str | None = None
    questions: list[str] = Field(default_factory=list)


class ChunkMetadataRead(BaseModel):
    chunk_id: str | None = None
    source_info: ChunkSourceInfo = Field(default_factory=ChunkSourceInfo)
    context: ChunkContextMetadata = Field(default_factory=ChunkContextMetadata)
    search_optimization: ChunkSearchOptimization = Field(default_factory=ChunkSearchOptimization)
    admin_tags: ChunkAdminTags = Field(default_factory=ChunkAdminTags)
    hyq: ChunkHyQMetadata = Field(default_factory=ChunkHyQMetadata)


class DocumentChunkRead(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    source_page: int | None = Field(default=None, ge=1)
    source_kind: str | None = None
    source_metadata: ChunkMetadataRead | dict[str, Any] | None = None
    created_at: datetime


class DocumentChunkListResponse(BaseModel):
    document_id: int
    total_chunks: int
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    items: list[DocumentChunkRead]


class SourceItem(BaseModel):
    document_id: int | None = None
    chunk_id: int | None = None
    chunk_index: int | None = None
    page: int | None = Field(default=None, ge=1)
    source_kind: str | None = None
    source_metadata: ChunkMetadataRead | dict[str, Any] | None = None
    retrieval_mode: str | None = None
    retrieval_score: float | None = None
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
