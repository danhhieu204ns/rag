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
                headers = {}
                if settings.ollama_api_key:
                    headers["x-api-key"] = settings.ollama_api_key

                _embeddings = OllamaEmbeddings(
                    model="default",
                    base_url=settings.ollama_base_url,
                    headers=headers,
                )
                logger.info(
                    "[embedding] Using Ollama embedding model at %s",
                    settings.ollama_base_url,
                )

    return _embeddings


def get_llm() -> ChatOllama:
    """Create or return cached Ollama chat model instance."""
    global _llm

    if _llm is None:
        with _llm_lock:
            if _llm is None:
                headers = {}
                if settings.ollama_api_key:
                    headers["x-api-key"] = settings.ollama_api_key

                _llm = ChatOllama(
                    model="default",
                    base_url=settings.ollama_base_url,
                    headers=headers,
                    temperature=0.0,
                )
    return _llm


def _get_variant_llm() -> ChatOllama:
    """Return cached LLM instance for multi-query variant generation."""
    global _variant_llm

    if _variant_llm is None:
        headers = {}
        if settings.ollama_api_key:
            headers["x-api-key"] = settings.ollama_api_key

        _variant_llm = ChatOllama(
            model="default",
            base_url=settings.ollama_base_url,
            headers=headers,
            temperature=0.0,
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
