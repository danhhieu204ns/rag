from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings resolved from environment variables."""

    app_name: str
    app_env: str
    base_dir: Path
    storage_dir: Path
    uploads_dir: Path
    qdrant_path: Path
    qdrant_collection_name: str
    qdrant_url: str
    qdrant_api_key: str
    database_path: Path
    ollama_base_url: str
    embedding_model_name: str
    pdf_parser_mode: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    llm_model: str
    llm_temperature: float
    llm_num_ctx: int
    llm_keep_alive: str
    ollama_num_thread: int
    hyq_enabled: bool
    hyq_use_llm: bool
    metadata_use_llm: bool
    metadata_model: str
    metadata_ollama_num_thread: int
    metadata_ollama_num_predict: int
    metadata_llm_batch_size: int
    model_warmup_on_startup: bool
    model_warmup_chat: bool
    model_warmup_metadata: bool
    model_warmup_embedding: bool
    query_rewrite_enabled: bool
    # Reranking
    reranker_enabled: bool
    reranker_model: str
    reranker_candidate_pool: int
    # Query Rewriting
    query_rewrite_model: str
    query_rewrite_min_words: int
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


def _pdf_parser_mode_env(default: str = "legacy") -> str:
    raw_value = _string_env("PDF_PARSER_MODE", default).lower()
    allowed = {"legacy", "marker"}
    if raw_value not in allowed:
        return default
    return raw_value



def get_settings() -> Settings:
    """Create immutable settings object from environment values."""

    base_dir = Path(__file__).resolve().parents[2]
    storage_dir = base_dir / "storage"
    uploads_dir = storage_dir / "uploads"
    qdrant_path = storage_dir / "indexes" / "global_qdrant"

    storage_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")

    hyq_use_llm = _bool_env("HYQ_USE_LLM", True)
    ollama_num_thread = max(1, _int_env("OLLAMA_NUM_THREAD", 8))
    metadata_ollama_num_thread = max(1, _int_env("METADATA_OLLAMA_NUM_THREAD", ollama_num_thread))
    metadata_ollama_num_predict = max(64, _int_env("METADATA_OLLAMA_NUM_PREDICT", 256))

    return Settings(
        app_name=os.getenv("APP_NAME", "RAG App Backend"),
        app_env=os.getenv("APP_ENV", "development"),
        base_dir=base_dir,
        storage_dir=storage_dir,
        uploads_dir=uploads_dir,
        qdrant_path=qdrant_path,
        qdrant_collection_name=_string_env("QDRANT_COLLECTION_NAME", "global_child_chunks"),
        qdrant_url=_string_env("QDRANT_URL", ""),
        qdrant_api_key=_string_env("QDRANT_API_KEY", ""),
        database_path=storage_dir / "app.db",
        ollama_base_url=ollama_base_url,
        embedding_model_name=_string_env("EMBEDDING_MODEL_NAME", "bge-m3:latest"),
        pdf_parser_mode=_pdf_parser_mode_env(),
        chunk_size=_int_env("CHUNK_SIZE", 500),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 50),
        retriever_k=4,
        llm_model=_string_env("LLM_MODEL", "llama3.1:8b"),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.0),
        llm_num_ctx=max(512, _int_env("LLM_NUM_CTX", 2048)),
        llm_keep_alive=_string_env("LLM_KEEP_ALIVE", "10m"),
        ollama_num_thread=ollama_num_thread,
        hyq_enabled=_bool_env("HYQ_ENABLED", True),
        hyq_use_llm=hyq_use_llm,
        metadata_use_llm=_bool_env("METADATA_USE_LLM", hyq_use_llm),
        metadata_model=_string_env("METADATA_MODEL", "gemma3:1b"),
        metadata_ollama_num_thread=metadata_ollama_num_thread,
        metadata_ollama_num_predict=metadata_ollama_num_predict,
        metadata_llm_batch_size=max(1, _int_env("METADATA_LLM_BATCH_SIZE", 2)),
        model_warmup_on_startup=_bool_env("MODEL_WARMUP_ON_STARTUP", False),
        model_warmup_chat=_bool_env("MODEL_WARMUP_CHAT", False),
        model_warmup_metadata=_bool_env("MODEL_WARMUP_METADATA", True),
        model_warmup_embedding=_bool_env("MODEL_WARMUP_EMBEDDING", True),
        query_rewrite_enabled=_bool_env("QUERY_REWRITE_ENABLED", False),
        # Reranking
        reranker_enabled=_bool_env("RERANKER_ENABLED", False),
        reranker_model=_string_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_candidate_pool=_int_env("RERANKER_CANDIDATE_POOL", 20),
        query_rewrite_model=_string_env("QUERY_REWRITE_MODEL", ""),
        query_rewrite_min_words=max(1, _int_env("QUERY_REWRITE_MIN_WORDS", 5)),
        secret_key=os.getenv("SECRET_KEY", "change-this-secret-key-in-production"),
        access_token_expire_minutes=_int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 1440),
        admin_default_username=os.getenv("ADMIN_DEFAULT_USERNAME", "admin"),
        admin_default_password=os.getenv("ADMIN_DEFAULT_PASSWORD", "Admin@123"),
    )


settings = get_settings()
