from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings resolved from environment variables."""

    app_name: str
    storage_dir: Path
    uploads_dir: Path
    qdrant_path: Path
    qdrant_collection_name: str
    qdrant_url: str
    qdrant_api_key: str
    database_path: Path
    ollama_base_url: str
    ollama_api_key: str
    ollama_chat_model: str
    ollama_embedding_model: str
    pdf_parser_mode: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    query_rewrite_enabled: bool
    query_rewrite_min_terms: int
    query_rewrite_max_terms: int
    orchestrator_enabled: bool
    # Hybrid retrieval
    hybrid_probe_multiplier: int
    hybrid_rrf_k: int
    hybrid_vector_rrf_weight: float
    hybrid_keyword_rrf_weight: float
    # Reranking
    reranker_enabled: bool
    reranker_model: str
    reranker_candidate_pool: int
    metadata_llm_batch_size: int
    indexing_concurrency: int
    embedding_batch_size: int
    ingestion_service_url: str
    ingestion_timeout_seconds: float
    retrieval_service_url: str
    retrieval_timeout_seconds: float
    vector_batch_size: int
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


def _int_env_any(names: tuple[str, ...], default: int) -> int:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        return int(raw_value)
    return default



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

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    ollama_api_key = os.getenv("OLLAMA_API_KEY", "").strip()

    return Settings(
        app_name=_string_env("APP_NAME", "RAG App Backend"),
        storage_dir=storage_dir,
        uploads_dir=uploads_dir,
        qdrant_path=qdrant_path,
        qdrant_collection_name=_string_env("QDRANT_COLLECTION_NAME", "global_child_chunks"),
        qdrant_url=_string_env("QDRANT_URL", ""),
        qdrant_api_key=_string_env("QDRANT_API_KEY", ""),
        database_path=storage_dir / "app.db",
        ollama_chat_model=_string_env("OLLAMA_CHAT_MODEL", "default"),
        ollama_embedding_model=_string_env("OLLAMA_EMBEDDING_MODEL", "default"),
        ollama_base_url=ollama_base_url,
        ollama_api_key=ollama_api_key,
        pdf_parser_mode=_pdf_parser_mode_env(),
        chunk_size=_int_env("CHUNK_SIZE", 500),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 50),
        retriever_k=4,
        query_rewrite_enabled=_bool_env("QUERY_REWRITE_ENABLED", False),
        query_rewrite_min_terms=max(
            1,
            _int_env_any(("QUERY_REWRITE_MIN_TERMS", "QUERY_REWRITE_MIN_WORDS"), 5),
        ),
        query_rewrite_max_terms=max(1, _int_env("QUERY_REWRITE_MAX_TERMS", 12)),
        orchestrator_enabled=_bool_env("ORCHESTRATOR_ENABLED", True),
        hybrid_probe_multiplier=max(1, _int_env("HYBRID_PROBE_MULTIPLIER", 4)),
        hybrid_rrf_k=max(1, _int_env("HYBRID_RRF_K", 60)),
        hybrid_vector_rrf_weight=_float_env("HYBRID_VECTOR_RRF_WEIGHT", 1.0),
        hybrid_keyword_rrf_weight=_float_env("HYBRID_KEYWORD_RRF_WEIGHT", 1.0),
        reranker_enabled=_bool_env("RERANKER_ENABLED", True),
        reranker_model=_string_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_candidate_pool=_int_env("RERANKER_CANDIDATE_POOL", 20),
        metadata_llm_batch_size=_int_env("METADATA_LLM_BATCH_SIZE", 12),
        indexing_concurrency=_int_env("INDEXING_CONCURRENCY", 12),
        embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", 128),
        ingestion_service_url=_string_env("INGESTION_SERVICE_URL", "").rstrip("/"),
        ingestion_timeout_seconds=_float_env("INGESTION_TIMEOUT_SECONDS", 180.0),
        retrieval_service_url=_string_env("RETRIEVAL_SERVICE_URL", "").rstrip("/"),
        retrieval_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
        vector_batch_size=_int_env("VECTOR_BATCH_SIZE", 64),
        secret_key=os.getenv("SECRET_KEY", "change-this-secret-key-in-production"),
        access_token_expire_minutes=_int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 1440),
        admin_default_username=os.getenv("ADMIN_DEFAULT_USERNAME", "admin"),
        admin_default_password=os.getenv("ADMIN_DEFAULT_PASSWORD", "Admin@123"),
    )


settings = get_settings()
