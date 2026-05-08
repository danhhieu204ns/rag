from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    storage_dir: Path
    pdf_parser_mode: str
    retrieval_service_url: str
    retrieval_timeout_seconds: float
    api_key: str


def _string_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or default


def _pdf_parser_mode_env(default: str = "legacy") -> str:
    value = _string_env("PDF_PARSER_MODE", default).lower()
    if value not in {"legacy", "marker"}:
        return default
    return value


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    storage_dir = Path(_string_env("INGESTION_STORAGE_DIR", str(root / "storage")))
    storage_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        app_name=_string_env("INGESTION_APP_NAME", "RAG Ingestion Service"),
        storage_dir=storage_dir,
        pdf_parser_mode=_pdf_parser_mode_env(),
        retrieval_service_url=_string_env("RETRIEVAL_SERVICE_URL", "").rstrip("/"),
        retrieval_timeout_seconds=float(_string_env("RETRIEVAL_TIMEOUT_SECONDS", "180")),
        api_key=_string_env("OLLAMA_API_KEY", ""),
    )


settings = get_settings()
