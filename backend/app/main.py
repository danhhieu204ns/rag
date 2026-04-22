from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .core.settings import settings
from .db import init_db


logger = logging.getLogger(__name__)


_QUANTIZED_TAG_PATTERN = re.compile(r"(?:^|:|[-_])q[2-8](?:[_-]?[a-z0-9]+)?$", re.IGNORECASE)


def _is_probably_quantized(model_name: str) -> bool:
    name = str(model_name or "").strip()
    if not name:
        return False
    if _QUANTIZED_TAG_PATTERN.search(name):
        return True
    # Ollama official tags may omit q-suffix but remain quantized; size check must be done via `ollama list`.
    return False


def _log_vram_tuning_hints() -> None:
    check_models = [
        ("LLM_MODEL", settings.llm_model),
        ("METADATA_MODEL", settings.metadata_model or settings.hyq_model or settings.llm_model),
    ]

    for setting_name, model_name in check_models:
        if model_name and not _is_probably_quantized(model_name):
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


@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    return {"status": "ok"}


app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
