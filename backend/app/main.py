from __future__ import annotations

import json
import logging
import os
import re
from urllib import error, request

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .core.settings import settings
from .db import init_db
from .services.chunk_metadata import warmup_metadata_model
from .services.rag_runtime import warmup_chat_model, warmup_embedding_model


logger = logging.getLogger(__name__)


_QUANTIZED_TAG_PATTERN = re.compile(r"(?:^|:|[-_])q[2-8](?:[_-]?[a-z0-9]+)?$", re.IGNORECASE)


def _fetch_ollama_model_sizes() -> dict[str, int]:
    """Fetch installed Ollama model sizes (bytes) via /api/tags."""

    base_url = settings.ollama_base_url.rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with request.urlopen(url, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return {}

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return {}

    sizes: dict[str, int] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        size = item.get("size")
        if not name:
            continue
        try:
            sizes[name] = int(size)
        except (TypeError, ValueError):
            continue
    return sizes


def _is_likely_quantized_by_size(model_name: str, size_bytes: int) -> bool:
    """Best-effort size heuristic for common model families when tag has no explicit q-suffix."""

    lower_name = model_name.lower()
    # Heuristic thresholds for common quantized pulls in Ollama.
    if "8b" in lower_name:
        return size_bytes <= 7 * 1024 * 1024 * 1024
    if "3b" in lower_name:
        return size_bytes <= 4 * 1024 * 1024 * 1024
    return False


def _is_probably_quantized(model_name: str) -> bool:
    name = str(model_name or "").strip()
    if not name:
        return False
    if _QUANTIZED_TAG_PATTERN.search(name):
        return True
    # Ollama official tags may omit q-suffix but remain quantized; size check must be done via `ollama list`.
    return False


def _log_vram_tuning_hints() -> None:
    ollama_sizes = _fetch_ollama_model_sizes()

    check_models = [
        ("LLM_MODEL", settings.llm_model),
        ("METADATA_MODEL", settings.metadata_model or settings.hyq_model or settings.llm_model),
    ]

    for setting_name, model_name in check_models:
        if not model_name:
            continue

        if _is_probably_quantized(model_name):
            continue

        size = ollama_sizes.get(model_name)
        if size is None and ":" not in model_name:
            size = ollama_sizes.get(f"{model_name}:latest")

        if size is not None and _is_likely_quantized_by_size(model_name, size):
            logger.info(
                "[%s] model='%s' looks quantized by size (%d bytes) even without explicit q4/q5 tag.",
                setting_name,
                model_name,
                size,
            )
            continue

        if model_name:
            logger.warning(
                "[%s] model='%s' has no explicit q4/q5 tag. For 12GB VRAM, prefer quantized tags (e.g. q4 or q5).",
                setting_name,
                model_name,
            )

    kv_cache_type = os.getenv("OLLAMA_KV_CACHE_TYPE", "").strip()
    if not kv_cache_type:
        logger.warning(
            "OLLAMA_KV_CACHE_TYPE is not set. Consider q8_0 or q4_0 to lower KV-cache memory usage."
        )
    else:
        logger.info("OLLAMA_KV_CACHE_TYPE=%s", kv_cache_type)

    flash_attention = os.getenv("OLLAMA_FLASH_ATTENTION", "").strip()
    if flash_attention in {"1", "true", "TRUE", "on", "ON"}:
        logger.info("OLLAMA_FLASH_ATTENTION is enabled.")
    else:
        logger.warning(
            "OLLAMA_FLASH_ATTENTION is not enabled. Enable it if your Ollama build supports Flash Attention."
        )

    num_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "").strip()
    if not num_parallel:
        logger.warning(
            "OLLAMA_NUM_PARALLEL is not set. For orchestration with shared VRAM, start with OLLAMA_NUM_PARALLEL=2."
        )
    else:
        logger.info("OLLAMA_NUM_PARALLEL=%s", num_parallel)

    max_loaded = os.getenv("OLLAMA_MAX_LOADED_MODELS", "").strip()
    if not max_loaded:
        logger.warning(
            "OLLAMA_MAX_LOADED_MODELS is not set. Set it to at least 2-3 to keep metadata and embedding models resident."
        )
    else:
        logger.info("OLLAMA_MAX_LOADED_MODELS=%s", max_loaded)

    logger.info(
        "Embedding backend=sentence-transformers model='%s' max_length=%d fp16=%s batch_size=%d device=%s",
        settings.embedding_model_name,
        settings.embedding_max_length,
        settings.embedding_use_fp16,
        settings.embedding_batch_size,
        settings.embedding_device,
    )


def _warmup_orchestrated_models() -> None:
    if not settings.model_warmup_on_startup:
        return

    if settings.model_warmup_embedding:
        try:
            warmup_embedding_model()
            logger.info("Embedding model warmup completed.")
        except Exception as exc:
            logger.warning("Embedding model warmup failed: %s", exc)

    if settings.model_warmup_metadata:
        try:
            warmup_metadata_model()
            logger.info("Metadata model warmup completed.")
        except Exception as exc:
            logger.warning("Metadata model warmup failed: %s", exc)

    if settings.model_warmup_chat:
        try:
            warmup_chat_model()
            logger.info("Chat model warmup completed.")
        except Exception as exc:
            logger.warning("Chat model warmup failed: %s", exc)


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.20.2.60:5173",
        "http://10.20.2.60:3000",
        "*"
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10\.20\.2\.\d+)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables and seed default admin at app startup."""

    init_db()
    _log_vram_tuning_hints()
    _warmup_orchestrated_models()


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    return {"status": "ok"}


app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
