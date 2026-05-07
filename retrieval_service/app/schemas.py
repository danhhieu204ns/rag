from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    document_ids: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: RetrievalFilters | None = None


class VectorSearchRequest(BaseModel):
    vector: list[float] = Field(..., min_length=1)
    collection: str | None = None
    top_k: int = Field(default=5, ge=1, le=100)
    filters: RetrievalFilters | None = None


class ContextItem(BaseModel):
    chunk_id: int | None = None
    document_id: int | None = None
    content: str
    source: str | None = None
    page: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    contexts: list[ContextItem]


class ChunkIndexItem(BaseModel):
    chunk_id: int | str
    document_id: int | str
    content: str = Field(..., min_length=1)
    source: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None


class IndexChunksRequest(BaseModel):
    collection: str | None = None
    chunks: list[ChunkIndexItem] = Field(..., min_length=1, max_length=1000)
    purge_document_ids: list[int] | None = None


class IndexChunksResponse(BaseModel):
    indexed_chunks: int
    collection: str


class DeleteDocumentResponse(BaseModel):
    document_id: int
    deleted: bool
    collection: str
