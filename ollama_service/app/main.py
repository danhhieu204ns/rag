from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_SERVICE_ROOT / ".env", override=True)

from .core.security import enforce_rate_limit, verify_api_key
from .core.settings import settings
from .ollama_client import (
    enforce_num_predict,
    get_ollama,
    model_dump,
    parse_ollama_json_response,
    post_ollama,
    usage_payload,
    validate_text_batch,
    validate_text_length,
)
from .prompts import INDEXING_INSTRUCTION, build_indexing_prompt
from .schemas import (
    ChatRequest,
    EmbedRequest,
    GenerateRequest,
    IndexingBatchRequest,
    IndexingRequest,
    OllamaNativeChatRequest,
    OllamaNativeEmbedRequest,
    OllamaNativeEmbeddingsRequest,
    OllamaNativeGenerateRequest,
)


app = FastAPI(
    title=settings.app_name,
    version="2.1.0",
    description="Protected API gateway for Ollama models: chat, indexing, embedding.",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ollama-fastapi-shield",
        "models": {
            "chat": settings.chat_model,
            "indexing": settings.indexing_model,
            "embedding": settings.embedding_model,
        },
    }


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "models")
    return {
        "configured_models": {
            "chat": settings.chat_model,
            "indexing": settings.indexing_model,
            "embedding": settings.embedding_model,
        },
        "ollama_models": await get_ollama("/api/tags", timeout_seconds=30.0),
    }


@app.get("/api/tags")
async def native_tags(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "api_tags")
    return await get_ollama("/api/tags", timeout_seconds=30.0)


@app.post("/v1/chat")
async def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "chat")
    _validate_messages(req.messages)

    payload = {
        "model": settings.chat_model,
        "messages": [model_dump(item) for item in req.messages],
        "options": req.options or {},
    }
    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    return await post_ollama(
        "/api/chat",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )


@app.post("/v1/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "generate")
    validate_text_length(req.prompt, settings.max_chat_chars, "prompt")
    if req.system:
        validate_text_length(req.system, settings.max_chat_chars, "system")

    payload: dict[str, Any] = {
        "model": settings.chat_model,
        "prompt": req.prompt,
        "options": req.options or {},
    }
    if req.system:
        payload["system"] = req.system

    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    return await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )


@app.post("/v1/indexing")
async def indexing(req: IndexingRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "indexing")
    validate_text_length(req.text, settings.max_indexing_chars, "text")

    ollama_result = await _call_indexing_model(
        text=req.text,
        instruction=req.instruction or INDEXING_INSTRUCTION,
        options=req.options or {},
    )
    return _indexing_response_from_ollama(ollama_result)


@app.post("/v1/indexing/batch")
async def indexing_batch(req: IndexingBatchRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "indexing_batch")

    if len(req.texts) > 100:
        from fastapi import HTTPException

        raise HTTPException(status_code=413, detail="Tối đa 100 đoạn text/lần indexing.")

    validate_text_batch(
        req.texts,
        max_items=100,
        max_chars=settings.max_indexing_chars,
        field_name="texts",
    )

    instruction = req.instruction or INDEXING_INSTRUCTION
    options = req.options or {}
    semaphore = asyncio.Semaphore(settings.indexing_concurrency)

    async def call_one(index: int, text: str) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            try:
                ollama_result = await _call_indexing_model(
                    text=text,
                    instruction=instruction,
                    options=options,
                )
                return index, _indexing_response_from_ollama(ollama_result)
            except Exception as exc:
                return index, {"error": str(exc)}

    tasks = [call_one(index, text) for index, text in enumerate(req.texts)]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    for task in asyncio.as_completed(tasks):
        index, item = await task
        results[index] = item

    return {"items": results}


@app.post("/v1/embed")
async def embed(req: EmbedRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "embed")
    validate_text_batch(
        req.input,
        max_items=100,
        max_chars=settings.max_indexing_chars,
        field_name="input",
    )

    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "input": req.input,
    }
    if req.options:
        payload["options"] = req.options

    return await post_ollama(
        "/api/embed",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )


@app.post("/api/embed")
async def native_embed(req: OllamaNativeEmbedRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "api_embed")
    input_text = req.input if req.input is not None else req.prompt
    if input_text is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail="Thiếu input hoặc prompt.")

    validate_text_batch(
        input_text,
        max_items=100,
        max_chars=settings.max_indexing_chars,
        field_name="input",
    )

    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "input": input_text,
    }
    for key in ("truncate", "options", "keep_alive", "dimensions"):
        value = getattr(req, key)
        if value is not None:
            payload[key] = value

    return await post_ollama(
        "/api/embed",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )


@app.post("/api/embeddings")
async def native_embeddings(
    req: OllamaNativeEmbeddingsRequest,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    enforce_rate_limit(api_key, "api_embeddings")
    validate_text_length(req.prompt, settings.max_indexing_chars, "prompt")

    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "prompt": req.prompt,
    }
    if req.options:
        payload["options"] = req.options
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive

    return await post_ollama(
        "/api/embeddings",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )


@app.post("/api/chat")
async def native_chat(req: OllamaNativeChatRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "api_chat")
    _validate_messages(req.messages)

    payload: dict[str, Any] = {
        "model": settings.chat_model,
        "messages": [model_dump(item) for item in req.messages],
        "options": req.options or {},
    }
    if req.format:
        payload["format"] = req.format
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive

    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    return await post_ollama(
        "/api/chat",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )


@app.post("/api/generate")
async def native_generate(req: OllamaNativeGenerateRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    enforce_rate_limit(api_key, "api_generate")
    validate_text_length(req.prompt, settings.max_chat_chars, "prompt")

    payload: dict[str, Any] = {
        "model": settings.chat_model,
        "prompt": req.prompt,
        "options": req.options or {},
    }
    if req.system:
        payload["system"] = req.system
    if req.format:
        payload["format"] = req.format
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive

    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    return await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )


async def _call_indexing_model(
    *,
    text: str,
    instruction: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    prompt = build_indexing_prompt(instruction=instruction, text=text)
    payload: dict[str, Any] = {
        "model": settings.indexing_model,
        "prompt": prompt,
        "format": "json",
        "options": dict(options),
    }
    payload = enforce_num_predict(payload, settings.max_indexing_num_predict)
    return await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_indexing_timeout_seconds,
    )


def _indexing_response_from_ollama(ollama_result: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_ollama_json_response(ollama_result)
    return {
        "summary": parsed.get("summary"),
        "hyq": parsed.get("hyq", []),
        "metadata": parsed.get("metadata", {}),
        "language": parsed.get("language"),
        "usage": usage_payload(ollama_result),
    }


def _validate_messages(messages: list[Any]) -> None:
    if len(messages) > settings.max_messages:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=413,
            detail=f"Quá nhiều message. Tối đa {settings.max_messages} message.",
        )

    total_chars = sum(len(str(item.content)) for item in messages)
    if total_chars > settings.max_chat_chars:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=413,
            detail=f"chat content quá dài. Tối đa {settings.max_chat_chars} ký tự.",
        )
