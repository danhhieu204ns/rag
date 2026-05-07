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
from .rag.qdrant import (
    delete_vectors_by_document_id as _delete_vectors_by_document_id_local,
    upsert_child_documents as _upsert_child_documents_local,
    rebuild_index_from_chunks,
    load_index_if_available as _load_index_if_available_local,
)
from .rag.retrieval import (
    similarity_search as _similarity_search_local,
    rerank_documents,
)
from .rag.generation import (
    generate_answer,
    generate_answer_stream,
    build_sources,
    parse_sources,
)


def delete_vectors_by_document_id(document_id: int) -> None:
    if retrieval_client.enabled():
        retrieval_client.delete_vectors_by_document_id(document_id)
        return
    _delete_vectors_by_document_id_local(document_id)


def upsert_child_documents(
    documents: list[Document],
    *,
    purge_document_ids: list[int] | None = None,
    precomputed_vectors: list[list[float]] | None = None,
) -> int:
    if retrieval_client.enabled():
        return retrieval_client.upsert_child_documents(
            documents,
            purge_document_ids=purge_document_ids,
            precomputed_vectors=precomputed_vectors,
        )
    return _upsert_child_documents_local(
        documents,
        purge_document_ids=purge_document_ids,
        precomputed_vectors=precomputed_vectors,
    )


def load_index_if_available() -> bool:
    if retrieval_client.enabled():
        return retrieval_client.health_ready()
    return _load_index_if_available_local()


def similarity_search(
    query: str,
    top_k: int,
    db: Session | None = None,
    document_ids: list[int] | None = None,
) -> list[Document]:
    if retrieval_client.enabled():
        return retrieval_client.similarity_search(
            query,
            top_k=top_k,
            document_ids=document_ids,
        )
    return _similarity_search_local(
        query,
        top_k=top_k,
        db=db,
        document_ids=document_ids,
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
