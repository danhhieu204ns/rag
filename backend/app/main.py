from __future__ import annotations

import logging
import time
import contextlib
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_BACKEND_ROOT / ".env", override=True)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.documents import router as documents_router
from .api.users import router as users_router
from .core.settings import settings
from .db import engine, init_db



logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


@contextlib.contextmanager
def _timed_step(name: str, logger: logging.Logger, **context):
    params = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info("[backend][timing] step=%s status=start %s", name, params)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = _ms(start)
        logger.exception("[backend][timing] step=%s status=error elapsed_ms=%.2f %s", name, elapsed, params)
        raise
    else:
        elapsed = _ms(start)
        logger.info("[backend][timing] step=%s status=ok elapsed_ms=%.2f %s", name, elapsed, params)


def _configure_backend_file_logging() -> logging.Logger:
    root = logging.getLogger("app.main")
    root.setLevel(logging.INFO)
    log_dir = Path(__file__).resolve().parents[1] / "storage" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / ("backend_service_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".log")
    from logging.handlers import RotatingFileHandler

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(str(log_path), maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path) for h in root.handlers):
        root.addHandler(fh)
    return root


# configure backend logger
logger = _configure_backend_file_logging()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize database tables and seed default admin at app startup."""

    start = time.perf_counter()
    logger.info("[backend][startup] initializing database")
    with _timed_step("startup.init_db", logger):
        init_db()
    logger.info("[backend][startup] initialization complete elapsed_ms=%.2f", _ms(start))


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    """Basic health endpoint for backend service."""

    start = time.perf_counter()
    logger.info("[backend][health] request received")
    elapsed = _ms(start)
    logger.info("[backend][health] response sent elapsed_ms=%.2f", elapsed)
    return {
        "status": "ok",
        "service": "api-gateway",
        "version": "1.0.0",
    }


@app.get("/ready")
@app.get("/api/ready")
def ready(response: Response) -> dict[str, Any]:
    """Readiness endpoint covering local DB and configured internal services."""

    start = time.perf_counter()
    logger.info("[backend][ready] request received")
    checks: dict[str, Any] = {
        "database": _database_ready(),
        "ollama_service": _http_service_ready(settings.ollama_base_url, settings.ollama_api_key),
    }

    if settings.ingestion_service_url:
        checks["ingestion_service"] = _http_service_ready(settings.ingestion_service_url)
    if settings.retrieval_service_url:
        checks["retrieval_service"] = _http_service_ready(
            settings.retrieval_service_url,
            settings.ollama_api_key,
        )

    status = "ok" if all(item["ready"] for item in checks.values()) else "degraded"
    if status != "ok":
        response.status_code = 503
    elapsed = _ms(start)
    logger.info("[backend][ready] response sent elapsed_ms=%.2f status=%s", elapsed, status)
    return {
        "status": status,
        "service": "api-gateway",
        "checks": checks,
    }


def _database_ready() -> dict[str, Any]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception as exc:  # pragma: no cover - depends on runtime filesystem
        logger.exception("Database readiness check failed")
        return {"ready": False, "error": str(exc)}


def _http_service_ready(url: str, api_key: str = "") -> dict[str, Any]:
    headers = {"x-api-key": api_key} if api_key else {}
    target = url.rstrip("/")
    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            response = client.get(f"{target}/ready")
            if response.status_code == 404:
                response = client.get(f"{target}/health")
        ready = response.status_code < 500
        return {
            "ready": ready,
            "status_code": response.status_code,
            "url": target,
        }
    except Exception as exc:
        return {
            "ready": False,
            "url": target,
            "error": str(exc),
        }


app.include_router(auth_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(users_router, prefix="/api")
