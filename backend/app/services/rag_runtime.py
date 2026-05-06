from __future__ import annotations

# Facade for the rag package to maintain backward compatibility
from .rag.models import (
    get_embeddings,
    get_llm,
    get_reranker,
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
    generate_answer_stream,
    build_sources,
    parse_sources,
)

__all__ = [
    "get_embeddings",
    "get_llm",
    "get_reranker",
    "delete_vectors_by_document_id",
    "upsert_child_documents",
    "rebuild_index_from_chunks",
    "load_index_if_available",
    "similarity_search",
    "rerank_documents",
    "generate_answer",
    "generate_answer_stream",
    "build_sources",
    "parse_sources",
]
