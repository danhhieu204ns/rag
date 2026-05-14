from __future__ import annotations

# Facade for the rag package to maintain backward compatibility
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from ..models import DocumentChunk

from . import retrieval_client
from .rag.models import (
    get_embeddings,
    get_llm,
    get_reranker,
)
from .rag.orchestrator import OrchestrationPlan
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
    top_k: int | None,
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


def rebuild_index_from_chunks(chunks: list[DocumentChunk]) -> int:
    """Convert parent `DocumentChunk` rows to child Documents and delegate
    indexing to the retrieval service via `retrieval_client.upsert_child_documents`.
    """
    if not chunks:
        return 0

    from json import loads as _json_loads

    docs: list[Document] = []
    for chunk in chunks:
        try:
            source_metadata = _json_loads(chunk.source_metadata_json) if chunk.source_metadata_json else {}
        except Exception:
            source_metadata = {}

        metadata = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "parent_chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "source_page": chunk.source_page,
            "source_kind": chunk.source_kind,
            "source_metadata": source_metadata,
        }
        docs.append(Document(page_content=chunk.content, metadata=metadata))

    purge_ids = sorted({int(c.document_id) for c in chunks})
    return retrieval_client.upsert_child_documents(docs, purge_document_ids=purge_ids)


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
    "generate_answer",
    "generate_answer_stream",
    "build_sources",
    "parse_sources",
]
