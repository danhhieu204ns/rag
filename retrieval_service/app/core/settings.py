from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    storage_dir: Path
    database_path: Path
    qdrant_path: Path
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    ollama_service_url: str
    ollama_api_key: str
    vector_batch_size: int
    default_top_k: int
    request_timeout_seconds: float


def _string_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _path_env(name: str, default: Path, *, base_dir: Path) -> Path:
    raw = _string_env(name, str(default))
    path = Path(raw)
    if path.is_absolute():
        return path
    return base_dir / path


def get_settings() -> Settings:
    service_root = Path(__file__).resolve().parents[2]
    repo_root = service_root.parent
    storage_dir = _path_env(
        "RETRIEVAL_STORAGE_DIR",
        service_root / "storage",
        base_dir=repo_root,
    )
    database_path = _path_env(
        "RETRIEVAL_DATABASE_PATH",
        repo_root / "backend" / "storage" / "app.db",
        base_dir=repo_root,
    )
    qdrant_path = _path_env(
        "QDRANT_PATH",
        repo_root / "backend" / "storage" / "indexes" / "global_qdrant",
        base_dir=repo_root,
    )

    storage_dir.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=_string_env("RETRIEVAL_APP_NAME", "RAG Retrieval Service"),
        storage_dir=storage_dir,
        database_path=database_path,
        qdrant_path=qdrant_path,
        qdrant_url=_string_env("QDRANT_URL", "").rstrip("/"),
        qdrant_api_key=_string_env("QDRANT_API_KEY", ""),
        qdrant_collection_name=_string_env("QDRANT_COLLECTION_NAME", "global_child_chunks"),
        ollama_service_url=_string_env("OLLAMA_SERVICE_URL", "http://localhost:8200").rstrip("/"),
        ollama_api_key=_string_env("OLLAMA_API_KEY", ""),
        vector_batch_size=max(1, _int_env("VECTOR_BATCH_SIZE", 64)),
        default_top_k=max(1, _int_env("RETRIEVAL_TOP_K", 5)),
        request_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
    )


settings = get_settings()
