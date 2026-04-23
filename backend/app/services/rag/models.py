from __future__ import annotations

import logging
import threading
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama

from ...core.settings import settings

logger = logging.getLogger(__name__)

_embeddings: Embeddings | None = None
_qdrant_client = None  # defined here so it won't conflict if moved
_llm: ChatOllama | None = None
_variant_llm: ChatOllama | None = None
_reranker: Any = None
_embeddings_lock = threading.Lock()
_llm_lock = threading.Lock()


class SentenceTransformerEmbeddings(Embeddings):
    """SentenceTransformer-based embeddings with fixed max_length and optional fp16."""

    def __init__(
        self,
        *,
        model_name: str,
        max_length: int,
        use_fp16: bool,
        batch_size: int,
        device: str,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        resolved_device = device.strip().lower() if device else "auto"
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs: dict[str, Any] = {}
        if use_fp16 and resolved_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_name,
            device=resolved_device,
            model_kwargs=model_kwargs,
            local_files_only=settings.embedding_local_files_only,
        )
        self.model.max_seq_length = max_length
        self.batch_size = max(1, batch_size)
        self.use_fp16 = bool(use_fp16)
        self.device = resolved_device

        if self.use_fp16 and resolved_device.startswith("cuda"):
            self.model.half()
        elif self.use_fp16:
            logger.warning(
                "EMBEDDING_USE_FP16=true but embedding device is '%s'. fp16 is skipped.",
                resolved_device,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.encode(
            [str(item or "") for item in texts],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


def get_embeddings() -> Embeddings:
    """Create or return cached local sentence-transformers embedding instance."""
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                try:
                    _embeddings = SentenceTransformerEmbeddings(
                        model_name=settings.embedding_model_name,
                        max_length=settings.embedding_max_length,
                        use_fp16=settings.embedding_use_fp16,
                        batch_size=settings.embedding_batch_size,
                        device=settings.embedding_device,
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "Embedding requires package 'sentence-transformers'."
                    ) from exc

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
