from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings resolved from environment variables."""

    app_name: str
    cors_allow_origins: list[str]
    cors_allow_origin_regex: str
    storage_dir: Path
    uploads_dir: Path
    database_path: Path
    ollama_base_url: str
    ollama_api_key: str
    ollama_chat_model: str
    ollama_embedding_model: str
    orchestrator_enabled: bool
    orchestrator_timeout_seconds: float
    ingestion_service_url: str
    ingestion_timeout_seconds: float
    retrieval_service_url: str
    retrieval_timeout_seconds: float
    # Auth
    secret_key: str
    access_token_expire_minutes: int
    admin_default_username: str
    admin_default_password: str



def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _string_env(name: str, default: str) -> str:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    cleaned = raw_value.strip().strip('"').strip("'")
    return cleaned or default


def _list_env(name: str, default: list[str]) -> list[str]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    values = [
        item.strip().strip('"').strip("'")
        for item in raw_value.split(",")
    ]
    return [item for item in values if item] or default


def get_settings() -> Settings:
    """Create immutable settings object from environment values."""

    base_dir = Path(__file__).resolve().parents[2]
    storage_dir = base_dir / "storage"
    uploads_dir = storage_dir / "uploads"

    storage_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    ollama_api_key = os.getenv("OLLAMA_API_KEY", "").strip()

    return Settings(
        app_name=_string_env("APP_NAME", "RAG App Backend"),
        cors_allow_origins=_list_env(
            "CORS_ALLOW_ORIGINS",
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ],
        ),
        cors_allow_origin_regex=_string_env("CORS_ALLOW_ORIGIN_REGEX", ""),
        storage_dir=storage_dir,
        uploads_dir=uploads_dir,
        database_path=storage_dir / "app.db",
        ollama_chat_model=_string_env("OLLAMA_CHAT_MODEL", "default"),
        ollama_embedding_model=_string_env("OLLAMA_EMBEDDING_MODEL", "default"),
        ollama_base_url=ollama_base_url,
        ollama_api_key=ollama_api_key,
        orchestrator_enabled=_bool_env("ORCHESTRATOR_ENABLED", True),
        orchestrator_timeout_seconds=_float_env("ORCHESTRATOR_TIMEOUT_SECONDS", 30.0),
        ingestion_service_url=_string_env("INGESTION_SERVICE_URL", "").rstrip("/"),
        ingestion_timeout_seconds=_float_env("INGESTION_TIMEOUT_SECONDS", 180.0),
        retrieval_service_url=_string_env("RETRIEVAL_SERVICE_URL", "").rstrip("/"),
        retrieval_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
        secret_key=os.getenv("SECRET_KEY", "change-this-secret-key-in-production"),
        access_token_expire_minutes=_int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 1440),
        admin_default_username=os.getenv("ADMIN_DEFAULT_USERNAME", "admin"),
        admin_default_password=os.getenv("ADMIN_DEFAULT_PASSWORD", "Admin@123"),
    )


settings = get_settings()
