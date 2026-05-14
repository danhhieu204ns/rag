from __future__ import annotations

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_SERVICE_ROOT / ".env", override=True)

from .core.security import enforce_rate_limit, verify_api_key
from .core.settings import settings
from .ollama_client import (
    enforce_num_predict,
    get_ollama,
    iter_ollama_stream,
    model_dump,
    open_ollama_stream,
    post_ollama,
    validate_text_batch,
    validate_text_length,
)
from .schemas import (
    ChatRequest,
    EmbedRequest,
    GenerateRequest,
    IndexingBatchRequest,
    OllamaNativeChatRequest,
    OllamaNativeEmbedRequest,
    OllamaNativeEmbeddingsRequest,
    OllamaNativeGenerateRequest,
    OrchestratorClassifyRequest,
)


logger = logging.getLogger(__name__)


_ORCHESTRATOR_SYSTEM_PROMPT = """You are a strict query router for a Vietnamese RAG assistant.
Return ONLY valid JSON with this schema:
{
  "query_type": "greeting|chitchat|needs_retrieval",
  "requires_retrieval": true|false,
  "confidence": 0.0-1.0,
  "reason": "short reason",
  "signals": ["..."]
}
Rules:
- greeting/chitchat (hello, thanks, social small talk, identity questions about the assistant)
  => requires_retrieval=false
- questions requiring factual/document knowledge => requires_retrieval=true
- be conservative: if unsure, requires_retrieval=true
"""


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
    request_start = time.perf_counter()
    logger.info("[ollama-service][health] request received")
    result = {
        "status": "ok",
        "service": "ollama-fastapi-shield",
        "models": {
            "chat": settings.chat_model,
            "orchestrator": settings.orchestrator_model,
            "embedding": settings.embedding_model,
        },
    }
    logger.info("[ollama-service][health] response sent elapsed_ms=%.2f", _ms(request_start))
    return result


@app.get("/ready")
async def ready(response: Response) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][ready] request received upstream=%s", settings.ollama_base_url)
    try:
        logger.debug("[ollama-service][ready] step=check_upstream_connectivity")
        async with httpx.AsyncClient(timeout=settings.ollama_connect_timeout_seconds) as client:
            upstream_response = await client.get(f"{settings.ollama_base_url}/api/tags")
        logger.info("[ollama-service][ready] upstream response status=%d", upstream_response.status_code)
        
        if upstream_response.status_code >= 500:
            response.status_code = 503
            logger.warning("[ollama-service][ready] upstream degraded status=%d", upstream_response.status_code)
        else:
            logger.debug("[ollama-service][ready] upstream ok")
            
        result = {
            "status": "ok" if upstream_response.status_code < 500 else "degraded",
            "service": "ollama-service",
            "upstream": settings.ollama_base_url,
            "upstream_status_code": upstream_response.status_code,
        }
        logger.info("[ollama-service][ready] response sent status=%s elapsed_ms=%.2f", result["status"], _ms(request_start))
        return result
    except Exception as exc:
        response.status_code = 503
        logger.error("[ollama-service][ready] upstream error elapsed_ms=%.2f: %s", _ms(request_start), exc)
        return {
            "status": "degraded",
            "service": "ollama-service",
            "upstream": settings.ollama_base_url,
            "error": str(exc),
        }


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][models] request received")
    
    logger.debug("[ollama-service][models] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "models")
    logger.debug("[ollama-service][models] step=rate_limit_ok elapsed_ms=%.2f", _ms(request_start))
    
    logger.debug("[ollama-service][models] step=get_upstream_models")
    models = await get_ollama("/api/tags", timeout_seconds=30.0)
    logger.info("[ollama-service][models] response sent elapsed_ms=%.2f", _ms(request_start))
    
    return {
        "configured_models": {
            "chat": settings.chat_model,
            "orchestrator": settings.orchestrator_model,
            "embedding": settings.embedding_model,
        },
        "ollama_models": models,
    }


@app.get("/api/tags")
async def native_tags(api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][api_tags] request received")
    
    logger.debug("[ollama-service][api_tags] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "api_tags")
    logger.debug("[ollama-service][api_tags] step=rate_limit_ok elapsed_ms=%.2f", _ms(request_start))
    
    logger.debug("[ollama-service][api_tags] step=get_upstream_tags")
    result = await get_ollama("/api/tags", timeout_seconds=30.0)
    logger.info("[ollama-service][api_tags] response sent elapsed_ms=%.2f", _ms(request_start))
    return result


@app.post("/v1/chat")
async def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][chat] request received messages=%d", len(req.messages))
    
    logger.debug("[ollama-service][chat] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "chat")
    logger.debug("[ollama-service][chat] step=rate_limit_ok elapsed_ms=%.2f", _ms(request_start))
    
    logger.debug("[ollama-service][chat] step=validate_messages")
    _validate_messages(req.messages)
    logger.debug("[ollama-service][chat] step=validate_ok messages=%d elapsed_ms=%.2f", len(req.messages), _ms(request_start))

    logger.debug("[ollama-service][chat] step=construct_payload model=%s", settings.chat_model)
    payload = {
        "model": settings.chat_model,
        "messages": [model_dump(item) for item in req.messages],
        "options": req.options or {},
    }
    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    logger.debug("[ollama-service][chat] step=payload_ready num_messages=%d elapsed_ms=%.2f", len(req.messages), _ms(request_start))
    
    logger.debug("[ollama-service][chat] step=call_upstream timeout_seconds=%d", settings.ollama_chat_timeout_seconds)
    result = await post_ollama(
        "/api/chat",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )
    logger.info("[ollama-service][chat] response sent model=%s elapsed_ms=%.2f", settings.chat_model, _ms(request_start))
    return result


@app.post("/v1/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][generate] request received prompt_chars=%d", len(req.prompt))
    
    logger.debug("[ollama-service][generate] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "generate")
    logger.debug("[ollama-service][generate] step=rate_limit_ok elapsed_ms=%.2f", _ms(request_start))
    
    logger.debug("[ollama-service][generate] step=validate_prompt max_chars=%d", settings.max_chat_chars)
    validate_text_length(req.prompt, settings.max_chat_chars, "prompt")
    if req.system:
        validate_text_length(req.system, settings.max_chat_chars, "system")
    logger.debug("[ollama-service][generate] step=validate_ok prompt_chars=%d elapsed_ms=%.2f", len(req.prompt), _ms(request_start))

    logger.debug("[ollama-service][generate] step=construct_payload model=%s", settings.chat_model)
    model_name = (req.model or "").strip() or settings.chat_model
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": req.prompt,
        "options": req.options or {},
    }
    if req.system:
        payload["system"] = req.system

    payload = enforce_num_predict(payload, settings.max_chat_num_predict)
    logger.debug("[ollama-service][generate] step=payload_ready model=%s elapsed_ms=%.2f", model_name, _ms(request_start))
    
    logger.debug("[ollama-service][generate] step=call_upstream timeout_seconds=%d", settings.ollama_chat_timeout_seconds)
    result = await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )
    logger.info("[ollama-service][generate] response sent model=%s elapsed_ms=%.2f", model_name, _ms(request_start))
    return result


@app.post("/v1/orchestrator/classify")
async def orchestrator_classify(
    req: OrchestratorClassifyRequest,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][orchestrator] request received query_chars=%d", len(req.query))

    logger.debug("[ollama-service][orchestrator] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "orchestrator")
    logger.debug("[ollama-service][orchestrator] step=rate_limit_ok elapsed_ms=%.2f", _ms(request_start))

    logger.debug("[ollama-service][orchestrator] step=validate_query max_chars=%d", settings.max_chat_chars)
    validate_text_length(req.query, settings.max_chat_chars, "query")
    logger.debug("[ollama-service][orchestrator] step=validate_ok elapsed_ms=%.2f", _ms(request_start))

    logger.debug(
        "[ollama-service][orchestrator] step=construct_payload model=%s",
        settings.orchestrator_model,
    )
    payload: dict[str, Any] = {
        "model": settings.orchestrator_model,
        "messages": [
            {"role": "system", "content": _ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": req.query},
        ],
        "format": "json",
        "options": {
            "temperature": settings.orchestrator_temperature,
            "num_predict": settings.orchestrator_num_predict,
        },
        "stream": False,
    }
    payload = enforce_num_predict(payload, settings.orchestrator_num_predict)
    logger.debug("[ollama-service][orchestrator] step=payload_ready elapsed_ms=%.2f", _ms(request_start))

    logger.debug(
        "[ollama-service][orchestrator] step=call_upstream timeout_seconds=%.1f",
        settings.ollama_orchestrator_timeout_seconds,
    )
    result = await post_ollama(
        "/api/chat",
        payload,
        timeout_seconds=settings.ollama_orchestrator_timeout_seconds,
    )
    content = (result.get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        logger.error("[ollama-service][orchestrator] empty upstream content")
        raise HTTPException(status_code=502, detail="Orchestrator model returned empty content.")

    parsed = _parse_orchestrator_json(content)
    if parsed is None:
        logger.error("[ollama-service][orchestrator] invalid JSON content=%s", content[:300])
        raise HTTPException(status_code=502, detail="Orchestrator model returned invalid JSON.")

    response = _normalize_orchestrator_result(parsed)
    logger.info(
        "[ollama-service][orchestrator] response sent model=%s type=%s requires_retrieval=%s elapsed_ms=%.2f",
        settings.orchestrator_model,
        response["query_type"],
        response["requires_retrieval"],
        _ms(request_start),
    )
    return response


@app.post("/v1/indexing/batch")
async def indexing_batch(req: IndexingBatchRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    request_start = time.perf_counter()
    logger.info("[ollama-service][indexing_batch] request received items=%d", len(req.items) if hasattr(req, 'items') else 0)
    logger.warning("[ollama-service][indexing_batch] endpoint disabled status_code=410 elapsed_ms=%.2f", _ms(request_start))
    del req, api_key
    raise HTTPException(
        status_code=410,
        detail=(
            "LLM metadata indexing is disabled. Re-index documents through the "
            "section_parent_child pipeline."
        ),
    )


@app.post("/v1/embed")
async def embed(req: EmbedRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    started = time.perf_counter()
    input_batch = req.input if isinstance(req.input, list) else [req.input]
    logger.info("[ollama-service][embed] request received batch_size=%d", len(input_batch))
    
    logger.debug("[ollama-service][embed] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "embed")
    logger.debug("[ollama-service][embed] step=rate_limit_ok elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][embed] step=validate_batch batch_size=%d max_chars=%d", len(input_batch), settings.max_embedding_chars)
    validate_text_batch(
        req.input,
        max_items=100,
        max_chars=settings.max_embedding_chars,
        field_name="input",
    )
    logger.debug("[ollama-service][embed] step=validate_ok elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][embed] step=construct_payload model=%s", settings.embedding_model)
    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "input": req.input,
    }
    if req.options:
        payload["options"] = req.options
    logger.debug("[ollama-service][embed] step=payload_ready elapsed_ms=%.2f", _ms(started))

    input_batch = req.input if isinstance(req.input, list) else [req.input]
    input_chars = sum(len(str(item or "")) for item in input_batch)
    logger.info(
        "[ollama-service][timing] step=generate_embedding status=start model=%s batch_size=%d input_chars=%d",
        settings.embedding_model,
        len(input_batch),
        input_chars,
    )
    
    logger.debug("[ollama-service][embed] step=call_upstream timeout_seconds=%d", settings.ollama_embedding_timeout_seconds)
    result = await post_ollama(
        "/api/embed",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )
    
    embeddings = result.get("embeddings")
    output_vectors = len(embeddings) if isinstance(embeddings, list) else 0
    logger.info(
        "[ollama-service][timing] step=generate_embedding status=ok model=%s output_vectors=%d elapsed_ms=%.2f",
        settings.embedding_model,
        output_vectors,
        _ms(started),
    )
    return result


@app.post("/api/embed")
async def native_embed(req: OllamaNativeEmbedRequest, api_key: str = Depends(verify_api_key)) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info("[ollama-service][api_embed] request received")
    
    logger.debug("[ollama-service][api_embed] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "api_embed")
    logger.debug("[ollama-service][api_embed] step=rate_limit_ok elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][api_embed] step=extract_input")
    input_text = req.input if req.input is not None else req.prompt
    if input_text is None:
        logger.error("[ollama-service][api_embed] missing input/prompt")
        raise HTTPException(status_code=422, detail="Thiếu input hoặc prompt.")
    logger.debug("[ollama-service][api_embed] step=validate_batch max_chars=%d", settings.max_embedding_chars)

    validate_text_batch(
        input_text,
        max_items=100,
        max_chars=settings.max_embedding_chars,
        field_name="input",
    )
    logger.debug("[ollama-service][api_embed] step=validate_ok elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][api_embed] step=construct_payload model=%s", settings.embedding_model)
    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "input": input_text,
    }
    for key in ("truncate", "options", "keep_alive", "dimensions"):
        value = getattr(req, key)
        if value is not None:
            payload[key] = value
    logger.debug("[ollama-service][api_embed] step=payload_ready elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][api_embed] step=call_upstream timeout_seconds=%d", settings.ollama_embedding_timeout_seconds)
    result = await post_ollama(
        "/api/embed",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )
    logger.info("[ollama-service][api_embed] response sent elapsed_ms=%.2f", _ms(started))
    return result


@app.post("/api/embeddings")
async def native_embeddings(
    req: OllamaNativeEmbeddingsRequest,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    started = time.perf_counter()
    logger.info("[ollama-service][api_embeddings] request received")
    
    logger.debug("[ollama-service][api_embeddings] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "api_embeddings")
    logger.debug("[ollama-service][api_embeddings] step=rate_limit_ok elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][api_embeddings] step=validate_prompt max_chars=%d", settings.max_embedding_chars)
    validate_text_length(req.prompt, settings.max_embedding_chars, "prompt")
    logger.debug("[ollama-service][api_embeddings] step=validate_ok elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][api_embeddings] step=construct_payload model=%s", settings.embedding_model)
    payload: dict[str, Any] = {
        "model": settings.embedding_model,
        "prompt": req.prompt,
    }
    if req.options:
        payload["options"] = req.options
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive
    logger.debug("[ollama-service][api_embeddings] step=payload_ready elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][api_embeddings] step=call_upstream timeout_seconds=%d", settings.ollama_embedding_timeout_seconds)
    result = await post_ollama(
        "/api/embeddings",
        payload,
        timeout_seconds=settings.ollama_embedding_timeout_seconds,
    )
    logger.info("[ollama-service][api_embeddings] response sent elapsed_ms=%.2f", _ms(started))
    return result


@app.post("/api/chat", response_model=None)
async def native_chat(
    req: OllamaNativeChatRequest,
    api_key: str = Depends(verify_api_key),
) -> Any:
    started = time.perf_counter()
    logger.info("[ollama-service][api_chat] request received messages=%d", len(req.messages))
    
    logger.debug("[ollama-service][api_chat] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "api_chat")
    logger.debug("[ollama-service][api_chat] step=rate_limit_ok elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][api_chat] step=validate_messages")
    _validate_messages(req.messages)
    logger.debug("[ollama-service][api_chat] step=validate_ok messages=%d elapsed_ms=%.2f", len(req.messages), _ms(started))

    logger.debug("[ollama-service][api_chat] step=construct_payload model=%s", settings.chat_model)
    payload: dict[str, Any] = {
        "model": settings.chat_model,
        "messages": [model_dump(item) for item in req.messages],
        "options": req.options or {},
    }
    if req.format:
        payload["format"] = req.format
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive

    payload = enforce_num_predict(payload, settings.max_chat_num_predict, stream=req.stream)
    logger.debug("[ollama-service][api_chat] step=payload_ready num_messages=%d elapsed_ms=%.2f", len(req.messages), _ms(started))
    
    logger.debug("[ollama-service][api_chat] step=call_upstream timeout_seconds=%d", settings.ollama_chat_timeout_seconds)
    if req.stream:
        client, response = await open_ollama_stream(
            "/api/chat",
            payload,
            timeout_seconds=settings.ollama_chat_timeout_seconds,
        )
        logger.info("[ollama-service][api_chat] stream response started elapsed_ms=%.2f", _ms(started))
        return StreamingResponse(
            iter_ollama_stream(client, response, path="/api/chat", started=started),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await post_ollama(
        "/api/chat",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )
    logger.info("[ollama-service][api_chat] response sent elapsed_ms=%.2f", _ms(started))
    return result


@app.post("/api/generate", response_model=None)
async def native_generate(
    req: OllamaNativeGenerateRequest,
    api_key: str = Depends(verify_api_key),
) -> Any:
    started = time.perf_counter()
    logger.info("[ollama-service][api_generate] request received prompt_chars=%d", len(req.prompt))
    
    logger.debug("[ollama-service][api_generate] step=enforce_rate_limit")
    enforce_rate_limit(api_key, "api_generate")
    logger.debug("[ollama-service][api_generate] step=rate_limit_ok elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][api_generate] step=validate_prompt max_chars=%d", settings.max_chat_chars)
    validate_text_length(req.prompt, settings.max_chat_chars, "prompt")
    logger.debug("[ollama-service][api_generate] step=validate_ok elapsed_ms=%.2f", _ms(started))

    logger.debug("[ollama-service][api_generate] step=construct_payload model=%s", settings.chat_model)
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

    payload = enforce_num_predict(payload, settings.max_chat_num_predict, stream=req.stream)
    logger.debug("[ollama-service][api_generate] step=payload_ready elapsed_ms=%.2f", _ms(started))
    
    logger.debug("[ollama-service][api_generate] step=call_upstream timeout_seconds=%d", settings.ollama_chat_timeout_seconds)
    if req.stream:
        client, response = await open_ollama_stream(
            "/api/generate",
            payload,
            timeout_seconds=settings.ollama_chat_timeout_seconds,
        )
        logger.info("[ollama-service][api_generate] stream response started elapsed_ms=%.2f", _ms(started))
        return StreamingResponse(
            iter_ollama_stream(client, response, path="/api/generate", started=started),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await post_ollama(
        "/api/generate",
        payload,
        timeout_seconds=settings.ollama_chat_timeout_seconds,
    )
    logger.info("[ollama-service][api_generate] response sent elapsed_ms=%.2f", _ms(started))
    return result


def _parse_orchestrator_json(content: str) -> dict[str, Any] | None:
    text = _strip_json_response(content)
    candidates = [text]

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    return None


def _strip_json_response(content: str) -> str:
    text = str(content or "").strip()
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end < 0:
            break
        text = (text[:start] + text[end + len("</think>"):]).strip()
    return text.replace("<think>", "").replace("</think>", "").strip()


def _normalize_orchestrator_result(parsed: dict[str, Any]) -> dict[str, Any]:
    requires_retrieval = bool(parsed.get("requires_retrieval", True))
    query_type = str(parsed.get("query_type") or "").strip()
    if query_type not in {"greeting", "chitchat", "needs_retrieval"}:
        query_type = "needs_retrieval" if requires_retrieval else "chitchat"

    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0

    signals = parsed.get("signals") or []
    if not isinstance(signals, list):
        signals = [str(signals)]

    return {
        "query_type": query_type,
        "requires_retrieval": requires_retrieval,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or ""),
        "signals": [str(item) for item in signals if item],
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
