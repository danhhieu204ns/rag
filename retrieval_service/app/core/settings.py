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
    embedding_batch_size: int
    default_top_k: int
    search_child_chunks: bool
    expand_to_parent: bool
    deduplicate_parents: bool
    top_k_children: int
    request_timeout_seconds: float
    hybrid_probe_multiplier: int
    hybrid_rrf_k: int
    hybrid_vector_rrf_weight: float
    hybrid_keyword_rrf_weight: float
    reranker_enabled: bool
    reranker_model: str
    reranker_candidate_pool: int
    reranker_preserve_rrf_top_n: int
    bm25_enabled: bool
    bm25_bilingual_expansion: bool
    bm25_candidate_limit_multiplier: int
    bm25_extra_synonyms: str
    query_rewrite_enabled: bool
    query_rewrite_min_terms: int
    query_rewrite_max_terms: int


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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
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
        embedding_batch_size=max(1, _int_env("EMBEDDING_BATCH_SIZE", 128)),
        default_top_k=max(1, _int_env("RETRIEVAL_TOP_K", 5)),
        search_child_chunks=_bool_env("RETRIEVAL_SEARCH_CHILD_CHUNKS", True),
        expand_to_parent=_bool_env("RETRIEVAL_EXPAND_TO_PARENT", True),
        deduplicate_parents=_bool_env("RETRIEVAL_DEDUPLICATE_PARENTS", True),
        top_k_children=max(1, _int_env("RETRIEVAL_TOP_K_CHILDREN", 20)),
        request_timeout_seconds=_float_env("RETRIEVAL_TIMEOUT_SECONDS", 180.0),
        hybrid_probe_multiplier=max(1, _int_env("HYBRID_PROBE_MULTIPLIER", 4)),
        hybrid_rrf_k=max(1, _int_env("HYBRID_RRF_K", 60)),
        hybrid_vector_rrf_weight=_float_env("HYBRID_VECTOR_RRF_WEIGHT", 1.0),
        hybrid_keyword_rrf_weight=_float_env("HYBRID_KEYWORD_RRF_WEIGHT", 1.0),
        reranker_enabled=_bool_env("RERANKER_ENABLED", True),
        reranker_model=_string_env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_candidate_pool=max(1, _int_env("RERANKER_CANDIDATE_POOL", 20)),
        reranker_preserve_rrf_top_n=max(0, _int_env("RERANKER_PRESERVE_RRF_TOP_N", 5)),
        bm25_enabled=_bool_env("BM25_ENABLED", True),
        bm25_bilingual_expansion=_bool_env("BM25_BILINGUAL_EXPANSION", True),
        bm25_candidate_limit_multiplier=max(1, _int_env("BM25_CANDIDATE_LIMIT_MULTIPLIER", 10)),
        bm25_extra_synonyms=_string_env("BM25_EXTRA_SYNONYMS", ""),
        query_rewrite_enabled=_bool_env("QUERY_REWRITE_ENABLED", False),
        query_rewrite_min_terms=max(1, _int_env("QUERY_REWRITE_MIN_TERMS", 5)),
        query_rewrite_max_terms=max(1, _int_env("QUERY_REWRITE_MAX_TERMS", 12)),
    )


settings = get_settings()
