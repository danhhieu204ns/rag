from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ...core.settings import settings

logger = logging.getLogger(__name__)

_embeddings: Embeddings | None = None
_qdrant_client = None  # defined here so it won't conflict if moved
_llm: ChatOllama | None = None
_variant_llm: ChatOllama | None = None
_reranker: Any = None
_embeddings_lock = threading.Lock()
_llm_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """Create or return cached Ollama embedding instance."""
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = OllamaEmbeddings(
                    model=settings.embedding_model_name,
                    base_url=settings.ollama_base_url,
                )
                logger.info(
                    "[embedding] Using Ollama embedding model '%s' at %s",
                    settings.embedding_model_name,
                    settings.ollama_base_url,
                )

    return _embeddings


def get_llm() -> ChatOllama:
    """Create or return cached Ollama chat model instance."""
    global _llm

    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = ChatOllama(
                    model=settings.llm_model,
                    base_url=settings.ollama_base_url,
                    temperature=settings.llm_temperature,
                    num_thread=settings.ollama_num_thread,
                    num_ctx=settings.llm_num_ctx,
                    keep_alive=settings.llm_keep_alive,
                )
    return _llm


def warmup_embedding_model() -> None:
    """Warm embedding model once to reduce first-request cold start."""
    get_embeddings().embed_documents(["warmup embedding model"])


def warmup_chat_model() -> None:
    """Warm chat model with minimal output to reduce first-request cold start."""
    get_llm().invoke("Trả về đúng 1 từ: OK")


def _get_variant_llm() -> ChatOllama:
    """Return cached LLM instance for multi-query variant generation."""
    global _variant_llm

    if _variant_llm is None:
        model = settings.multi_query_model or settings.llm_model
        _variant_llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_thread=settings.ollama_num_thread,
            format="json",
        )
    return _variant_llm


def get_reranker() -> Any:
    """Lazy-load CrossEncoder reranker."""
    global _reranker
    if _reranker is None and settings.reranker_enabled:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(
                settings.reranker_model,
                max_length=512,
            )
            logger.info(
                "[reranker] Loaded CrossEncoder: model=%s",
                settings.reranker_model,
            )
        except ImportError:
            logger.error(
                "[reranker] sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error("[reranker] Failed to load reranker: %s", str(e))
            return None
    return _reranker
