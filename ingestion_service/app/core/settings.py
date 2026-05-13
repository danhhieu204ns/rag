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
    chunking_strategy: str
    parent_max_tokens: int
    child_chunk_size: int
    child_chunk_overlap: int
    prepend_heading_path: bool
    indexing_index_type: str
    api_key: str


def _string_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or default


def _resolve_service_path(raw_path: str, service_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    parts = path.parts
    if parts and parts[0] == service_root.name:
        return service_root.parent / path
    return service_root / path


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


def _bool_env(name: str, default: bool) -> bool:
    value = _string_env(name, "true" if default else "false").strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    storage_dir = _resolve_service_path(_string_env("INGESTION_STORAGE_DIR", "storage"), root)
    storage_dir.mkdir(parents=True, exist_ok=True)
    child_chunk_size = max(1, _int_env("CHILD_CHUNK_SIZE", _int_env("CHUNK_SIZE", 500)))
    return Settings(
        app_name=_string_env("INGESTION_APP_NAME", "RAG Ingestion Service"),
        storage_dir=storage_dir,
        pdf_parser_mode=_pdf_parser_mode_env(),
        retrieval_service_url=_string_env("RETRIEVAL_SERVICE_URL", "").rstrip("/"),
        retrieval_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
        chunking_strategy=_string_env("CHUNKING_STRATEGY", "section_parent_child"),
        parent_max_tokens=max(1, _int_env("PARENT_MAX_TOKENS", 2500)),
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=max(0, min(_int_env("CHILD_CHUNK_OVERLAP", _int_env("CHUNK_OVERLAP", 100)), child_chunk_size - 1)),
        prepend_heading_path=_bool_env("PREPEND_HEADING_PATH", True),
        indexing_index_type=_string_env("INDEXING_INDEX_TYPE", "section_parent_child"),
        api_key=_string_env("OLLAMA_API_KEY", ""),
    )


settings = get_settings()
