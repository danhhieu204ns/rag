from __future__ import annotations

# Facade for the rag package to maintain backward compatibility
from .rag.models import (
    get_embeddings,
    get_llm,
    get_reranker,
    warmup_embedding_model,
    warmup_chat_model,
)
from .rag.qdrant import (
    delete_vectors_by_document_id,
    upsert_child_documents,
    rebuild_index_from_chunks,
    load_index_if_available,
)
from .rag.retrieval import (
    similarity_search,
    rerank_documents,
)
from .rag.generation import (
    generate_answer,
    build_sources,
    parse_sources,
)

__all__ = [
    "get_embeddings",
    "get_llm",
    "get_reranker",
    "warmup_embedding_model",
    "warmup_chat_model",
    "delete_vectors_by_document_id",
    "upsert_child_documents",
    "rebuild_index_from_chunks",
    "load_index_if_available",
    "similarity_search",
    "rerank_documents",
    "generate_answer",
    "build_sources",
    "parse_sources",
]
