from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Response

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
    OllamaNativeChatRequest,
    OllamaNativeEmbedRequest,
    OllamaNativeEmbeddingsRequest,
    OllamaNativeGenerateRequest,
)


logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _configure_ollama_file_logging() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ollama_service_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(getattr(handler, "baseFilename", "")) == log_path:
                handler.setFormatter(formatter)
                return log_path

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return log_path


app = FastAPI(
    title=settings.app_name,
    version="2.2.0",
    description="Protected API gateway for Ollama inference: chat, generate, embed.",
)
_OLLAMA_LOG_PATH = _configure_ollama_file_logging()
logger.info("[ollama-service] file logging enabled path=%s", _OLLAMA_LOG_PATH)


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


@app.get("/ready")
async def ready(response: Response) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_connect_timeout_seconds) as client:
            upstream_response = await client.get(f"{settings.ollama_base_url}/api/tags")
        if upstream_response.status_code >= 500:
            response.status_code = 503
        return {
            "status": "ok" if upstream_response.status_code < 500 else "degraded",
            "service": "ollama-service",
            "upstream": settings.ollama_base_url,
            "upstream_status_code": upstream_response.status_code,
        }
    except Exception as exc:
        response.status_code = 503
        return {
            "status": "degraded",
            "service": "ollama-service",
            "upstream": settings.ollama_base_url,
            "error": str(exc),
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

    model_name = (req.model or "").strip() or settings.indexing_model
    payload: dict[str, Any] = {
        "model": model_name,
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


async def _call_indexing_model(
    *,
    text: str,
    instruction: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model": settings.indexing_model,
        "prompt": build_indexing_prompt(instruction, text),
        "format": "json",
        "options": options,
    }
    payload = enforce_num_predict(payload, settings.max_indexing_num_predict)
    return await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_indexing_timeout_seconds,
    )


def _normalize_indexing_response(parsed: dict[str, Any], ollama_result: dict[str, Any]) -> dict[str, Any]:
    summary = str(parsed.get("summary") or "").strip()
    hyq = parsed.get("hyq")
    questions = [str(item).strip()[:240] for item in hyq if str(item).strip()] if isinstance(hyq, list) else []
    metadata = parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {}
    language = str(parsed.get("language") or "vi").strip() or "vi"
    return {
        "summary": summary[:1200],
        "hyq": questions[:3],
        "metadata": metadata,
        "language": language[:16],
        "usage": usage_payload(ollama_result),
    }


def _fallback_indexing_response(text: str, ollama_result: dict[str, Any], error: Exception) -> dict[str, Any]:
    raw_response = str(ollama_result.get("response") or "")
    summary = " ".join(str(text or "").split())[:700] or "Không có tóm tắt."
    logger.warning(
        "[ollama-service] indexing JSON parse failed; using fallback text_chars=%d raw_chars=%d error=%s",
        len(text),
        len(raw_response),
        error,
    )
    return {
        "summary": summary,
        "hyq": [
            "Nội dung chính của đoạn này là gì?",
            "Đoạn này chứa các thông tin quan trọng nào?",
            "Có thể dùng đoạn này để trả lời câu hỏi nào?",
        ],
        "metadata": {"keywords": [], "risk_level": "low"},
        "language": "vi",
        "usage": usage_payload(ollama_result),
        "fallback": {
            "reason": "invalid_model_json",
            "raw_response_preview": raw_response[:500],
        },
    }


def _indexing_response_from_ollama(text: str, ollama_result: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = parse_ollama_json_response(ollama_result)
    except Exception as exc:
        return _fallback_indexing_response(text, ollama_result, exc)
    return _normalize_indexing_response(parsed, ollama_result)


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

    logger.info(
        "[ollama-service] /v1/indexing/batch items=%d instruction_chars=%d model=%s concurrency=%d",
        len(req.texts),
        len(instruction),
        settings.indexing_model,
        settings.indexing_concurrency,
    )

    async def call_one(index: int, text: str) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            try:
                ollama_result = await _call_indexing_model(
                    text=text,
                    instruction=instruction,
                    options=options,
                )
                return index, _indexing_response_from_ollama(text, ollama_result)
            except Exception as exc:
                return index, {"error": str(exc)}

    tasks = [call_one(index, text) for index, text in enumerate(req.texts)]
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    for task in asyncio.as_completed(tasks):
        index, item = await task
        results[index] = item

    error_count = sum(1 for item in results if isinstance(item, dict) and item.get("error"))
    logger.info(
        "[ollama-service] /v1/indexing/batch done items=%d errors=%d",
        len(results),
        error_count,
    )

    return {"items": results}


@app.post("/v1/embed")
async def embed(req: EmbedRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    started = time.perf_counter()
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

    input_batch = req.input if isinstance(req.input, list) else [req.input]
    logger.info(
        "[ollama-service][timing] step=generate_embedding status=start model_name=%s batch_size=%d input_chars=%d",
        settings.embedding_model,
        len(input_batch),
        sum(len(str(item or "")) for item in input_batch),
    )
    result = await post_ollama(
        "/api/embed",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )
    embeddings = result.get("embeddings")
    output_vectors = len(embeddings) if isinstance(embeddings, list) else 0
    logger.info(
        "[ollama-service][timing] step=generate_embedding status=ok model_name=%s output_vectors=%d elapsed_ms=%.2f",
        settings.embedding_model,
        output_vectors,
        _ms(started),
    )
    return result


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
        "model": settings.indexing_model,
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
