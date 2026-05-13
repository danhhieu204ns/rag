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
    indexing_timeout_seconds: float
    indexing_batch_size: int
    indexing_num_predict: int
    llm_min_chunk_chars: int
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


def _float_env(name: str, default: float) -> float:
    try:
        return float(_string_env(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_string_env(name, str(default)))
    except ValueError:
        return default


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    storage_dir = Path(_string_env("INGESTION_STORAGE_DIR", str(root / "storage")))
    storage_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        app_name=_string_env("INGESTION_APP_NAME", "RAG Ingestion Service"),
        storage_dir=storage_dir,
        pdf_parser_mode=_pdf_parser_mode_env(),
        retrieval_service_url=_string_env("RETRIEVAL_SERVICE_URL", "").rstrip("/"),
        retrieval_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
        indexing_timeout_seconds=_float_env("INDEXING_TIMEOUT_SECONDS", _float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0)),
        indexing_batch_size=max(1, min(100, _int_env("INDEXING_BATCH_SIZE", 16))),
        indexing_num_predict=max(64, min(2048, _int_env("INDEXING_NUM_PREDICT", 384))),
        llm_min_chunk_chars=max(0, _int_env("LLM_MIN_CHUNK_CHARS", 300)),
        api_key=_string_env("OLLAMA_API_KEY", ""),
    )


settings = get_settings()
