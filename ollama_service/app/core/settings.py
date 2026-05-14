from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    storage_dir: Path
    ollama_base_url: str
    shield_api_key: str
    chat_model: str
    orchestrator_model: str
    embedding_model: str
    max_chat_chars: int
    max_embedding_chars: int
    max_messages: int
    max_chat_num_predict: int
    orchestrator_num_predict: int
    orchestrator_temperature: float
    rate_limit_per_minute: int
    ollama_connect_timeout_seconds: float
    ollama_chat_timeout_seconds: float
    ollama_orchestrator_timeout_seconds: float
    ollama_embedding_timeout_seconds: float


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


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    storage_dir = Path(_string_env("OLLAMA_STORAGE_DIR", str(root / "storage")))
    storage_dir.mkdir(parents=True, exist_ok=True)
    ollama_base_url = _string_env(
        "UPSTREAM_OLLAMA_BASE_URL",
        _string_env("OLLAMA_UPSTREAM_BASE_URL", _string_env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
    ).rstrip("/")
    return Settings(
        app_name=_string_env("OLLAMA_SHIELD_APP_NAME", "Ollama FastAPI Shield"),
        storage_dir=storage_dir,
        ollama_base_url=ollama_base_url,
        shield_api_key=_string_env("SHIELD_API_KEY", ""),
        chat_model=_string_env("CHAT_MODEL", "qwen3:30b-a3b-instruct-2507-q4_K_M"),
        orchestrator_model=_string_env("ORCHESTRATOR_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
        embedding_model=_string_env("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        max_chat_chars=_int_env("MAX_CHAT_CHARS", 24000),
        max_embedding_chars=_int_env("MAX_EMBEDDING_CHARS", 12000),
        max_messages=_int_env("MAX_MESSAGES", 20),
        max_chat_num_predict=_int_env("MAX_CHAT_NUM_PREDICT", 2048),
        orchestrator_num_predict=_int_env("ORCHESTRATOR_NUM_PREDICT", 500),
        orchestrator_temperature=_float_env("ORCHESTRATOR_TEMPERATURE", 0.0),
        rate_limit_per_minute=max(1, _int_env("RATE_LIMIT_PER_MINUTE", 30)),
        ollama_connect_timeout_seconds=_float_env("OLLAMA_CONNECT_TIMEOUT_SECONDS", 10.0),
        ollama_chat_timeout_seconds=_float_env("OLLAMA_CHAT_TIMEOUT_SECONDS", 240.0),
        ollama_orchestrator_timeout_seconds=_float_env("OLLAMA_ORCHESTRATOR_TIMEOUT_SECONDS", 30.0),
        ollama_embedding_timeout_seconds=_float_env("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", 180.0),
    )


settings = get_settings()
