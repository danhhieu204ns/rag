from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ...core.settings import settings
from ..ollama_service_client import ollama_service_headers, require_ollama_service_url

logger = logging.getLogger(__name__)

_embeddings: Embeddings | None = None
_qdrant_client = None  # defined here so it won't conflict if moved
_llm: ChatOllama | None = None
_reranker: Any = None
_embeddings_lock = threading.Lock()
_llm_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """Create or return cached Ollama embedding instance."""
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                # Nếu dùng Ollama qua proxy/service khác yêu cầu xác thực API key
                base_url = require_ollama_service_url()
                headers = ollama_service_headers()

                # LangChain Ollama yêu cầu truyền headers qua client_kwargs
                # Note: OllamaEmbeddings uses /api/embeddings by default.
                # If using a proxy that maps /v1/chat but not /api/embeddings, this may fail with 404.
                _embeddings = OllamaEmbeddings(
                    model=settings.ollama_embedding_model,
                    base_url=base_url,
                    client_kwargs={"headers": headers},
                )
                logger.info(
                    "[embedding] Using remote Ollama embedding model (with auth) at %s, model=%s",
                    base_url,
                    settings.ollama_embedding_model,
                )

    return _embeddings


def get_llm() -> ChatOllama:
    """Create or return cached Ollama chat model instance."""
    global _llm

    if _llm is None:
        with _llm_lock:
            if _llm is None:
                base_url = require_ollama_service_url()
                headers = ollama_service_headers()

                _llm = ChatOllama(
                    model=settings.ollama_chat_model,
                    base_url=base_url,
                    client_kwargs={"headers": headers},
                    temperature=0.0,
                    reasoning=False,
                )
    return _llm


def _get_variant_llm() -> ChatOllama:
    """Return cached LLM instance for multi-query variant generation."""
    global _variant_llm

    if _variant_llm is None:
        base_url = require_ollama_service_url()
        headers = ollama_service_headers()

        _variant_llm = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=base_url,
            client_kwargs={"headers": headers},
            temperature=0.0,
            format="json",
        )
    return _variant_llm

def get_reranker() -> Any:
    """Backward-compatible stub; reranking is owned by retrieval_service."""
    logger.debug("[reranker] Backend reranker disabled; retrieval_service owns reranking.")
    return None
