from __future__ import annotations

# Facade for the rag package to maintain backward compatibility
from langchain_core.documents import Document
from sqlalchemy.orm import Session

from . import retrieval_client
from .rag.models import (
    get_embeddings,
    get_llm,
    get_reranker,
)
from .rag.orchestrator import OrchestrationPlan
from .rag.qdrant import rebuild_index_from_chunks
from .rag.retrieval import (
    rerank_documents,
)
from .rag.generation import (
    generate_answer,
    generate_answer_stream,
    build_sources,
    parse_sources,
)


def delete_vectors_by_document_id(document_id: int) -> None:
    retrieval_client.delete_vectors_by_document_id(document_id)


def upsert_child_documents(
    documents: list[Document],
    *,
    purge_document_ids: list[int] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
) -> int:
    return retrieval_client.upsert_child_documents(
        documents,
        purge_document_ids=purge_document_ids,
        precomputed_vectors=precomputed_vectors,
    )


def load_index_if_available() -> bool:
    return retrieval_client.health_ready()


def retrieval_service_enabled() -> bool:
    return retrieval_client.enabled()


def similarity_search(
    query: str,
    top_k: int,
    db: Session | None = None,
    document_ids: list[int] | None = None,
    plan: OrchestrationPlan | None = None,
) -> list[Document]:
    return retrieval_client.similarity_search(
        query,
        top_k=top_k,
        document_ids=document_ids,
        plan=plan,
    )


__all__ = [
    "get_embeddings",
    "get_llm",
    "get_reranker",
    "delete_vectors_by_document_id",
    "upsert_child_documents",
    "rebuild_index_from_chunks",
    "load_index_if_available",
    "retrieval_service_enabled",
    "similarity_search",
    "rerank_documents",
    "generate_answer",
    "generate_answer_stream",
    "build_sources",
    "parse_sources",
]
