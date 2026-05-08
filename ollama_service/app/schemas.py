from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    options: dict[str, Any] | None = None


class GenerateRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1)
    system: str | None = None
    options: dict[str, Any] | None = None


class IndexingBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    instruction: str | None = None
    options: dict[str, Any] | None = None


class EmbedRequest(BaseModel):
    input: str | list[str] = Field(...)
    options: dict[str, Any] | None = None


class OllamaNativeEmbedRequest(BaseModel):
    model: str | None = None
    input: str | list[str] | None = None
    prompt: str | None = None
    truncate: bool | None = None
    options: dict[str, Any] | None = None
    keep_alive: float | str | None = None
    dimensions: int | None = None


class OllamaNativeEmbeddingsRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1)
    options: dict[str, Any] | None = None
    keep_alive: float | str | None = None


class OllamaNativeChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(..., min_length=1)
    stream: bool | None = False
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    keep_alive: float | str | None = None


class OllamaNativeGenerateRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1)
    system: str | None = None
    stream: bool | None = False
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] | None = None
    keep_alive: float | str | None = None
