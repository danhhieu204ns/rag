from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections import defaultdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sqlalchemy.orm import Session

from ..core.settings import settings
from ..models import ChatMessage, DocumentChunk
from .chunk_metadata import build_hyq_children, build_keyword_blob, extract_document_codes

logger = logging.getLogger(__name__)
_EMBED_BATCH_SIZE = 16

_embeddings: Embeddings | None = None
_qdrant_client: QdrantClient | None = None
_llm: ChatOllama | None = None


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_lookup_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.lower().split())


def _lookup_terms(query: str) -> list[str]:
    normalized = _normalize_lookup_text(query)
    terms = re.findall(r"[a-z0-9/-]+", normalized)
    return [term for term in terms if len(term) >= 2]


def _parse_chunk_source_metadata(raw_json: str | None) -> dict[str, object]:
    if not raw_json:
        return {}
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        return {}


def _compact_source_metadata(raw_json: str | None) -> dict[str, object]:
    raw_metadata = _parse_chunk_source_metadata(raw_json)
    if not raw_metadata:
        return {}

    wanted_keys = {
        "chunk_id",
        "source_info",
        "context",
        "search_optimization",
        "admin_tags",
        "hyq",
    }
    compact = {
        key: value
        for key, value in raw_metadata.items()
        if key in wanted_keys and value is not None
    }
    if compact:
        return compact
    return raw_metadata


def _emit_reindex_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    logger.info(text)
    print(text, flush=True)


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    if settings.qdrant_url:
        kwargs: dict[str, str] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _qdrant_client = QdrantClient(**kwargs)
    else:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(settings.qdrant_path))

    return _qdrant_client


def _qdrant_collection_exists(client: QdrantClient) -> bool:
    collections = client.get_collections().collections
    return any(item.name == settings.qdrant_collection_name for item in collections)


def _clear_qdrant_collection(client: QdrantClient) -> None:
    if _qdrant_collection_exists(client):
        client.delete_collection(collection_name=settings.qdrant_collection_name)


def _serialize_qdrant_payload(metadata: dict[str, object], child_text: str) -> dict[str, object]:
    payload = dict(metadata)
    payload["child_text"] = child_text
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _search_qdrant_children(query: str, limit: int) -> list[Document]:
    if limit <= 0:
        return []

    client = _get_qdrant_client()
    if not _qdrant_collection_exists(client):
        return []

    query_vector = get_embeddings().embed_query(query)
    points = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
    )

    documents: list[Document] = []
    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        child_text = str(payload.get("child_text") or "")
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "child_text"
        }
        documents.append(Document(page_content=child_text, metadata=metadata))

    return documents


def _rebuild_qdrant_collection_with_batch_embeddings(documents: list[Document]) -> int:
    texts = [item.page_content for item in documents]
    metadatas = [dict(item.metadata or {}) for item in documents]
    embeddings_client = get_embeddings()

    total = len(texts)
    batch_size = max(1, _EMBED_BATCH_SIZE)
    batch_count = (total + batch_size - 1) // batch_size

    all_vectors: list[list[float]] = []
    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        batch_started_at = time.perf_counter()
        batch_vectors = embeddings_client.embed_documents(texts[start:end])
        all_vectors.extend(batch_vectors)

        elapsed = time.perf_counter() - batch_started_at
        _emit_reindex_progress(
            "[reindex] Embedded child docs %d/%d (batch %d/%d, %.2fs)",
            end,
            total,
            batch_index,
            batch_count,
            elapsed,
        )

    if not all_vectors:
        return 0

    client = _get_qdrant_client()
    _clear_qdrant_collection(client)

    vector_size = len(all_vectors[0])
    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    upsert_batch_size = max(1, _EMBED_BATCH_SIZE)
    upsert_batch_count = (total + upsert_batch_size - 1) // upsert_batch_size
    for batch_index, start in enumerate(range(0, total, upsert_batch_size), start=1):
        end = min(start + upsert_batch_size, total)
        points: list[PointStruct] = []

        for idx in range(start, end):
            points.append(
                PointStruct(
                    id=idx + 1,
                    vector=all_vectors[idx],
                    payload=_serialize_qdrant_payload(metadatas[idx], texts[idx]),
                )
            )

        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        _emit_reindex_progress(
            "[reindex] Upserted child docs %d/%d (batch %d/%d)",
            end,
            total,
            batch_index,
            upsert_batch_count,
        )

    return total


def _load_chunks_by_ids(db: Session, chunk_ids: list[int]) -> list[DocumentChunk]:
    if not chunk_ids:
        return []

    rows = db.query(DocumentChunk).filter(DocumentChunk.id.in_(chunk_ids)).all()
    row_by_id = {row.id: row for row in rows}
    return [row_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in row_by_id]


def _chunk_to_context_document(
    chunk: DocumentChunk,
    *,
    retrieval_mode: str | None = None,
    retrieval_score: float | None = None,
    child_type: str | None = None,
) -> Document:
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "source_page": chunk.source_page,
        "source_kind": chunk.source_kind,
    }

    source_metadata = _compact_source_metadata(chunk.source_metadata_json)
    if source_metadata:
        metadata["source_metadata"] = source_metadata

    if retrieval_mode:
        metadata["retrieval_mode"] = retrieval_mode
    if retrieval_score is not None:
        metadata["retrieval_score"] = retrieval_score
    if child_type:
        metadata["child_type"] = child_type

    return Document(page_content=chunk.content, metadata=metadata)


def _vector_parent_candidates(
    query: str,
    top_k: int,
    document_ids: list[int] | None,
) -> tuple[list[int], dict[int, str]]:
    if not load_index_if_available():
        return [], {}

    probe_multiplier = max(1, settings.hybrid_probe_multiplier)
    probe_k = max(top_k * probe_multiplier, top_k)
    children = _search_qdrant_children(query, limit=probe_k)

    wanted_docs = {int(doc_id) for doc_id in document_ids} if document_ids else None
    parent_ids: list[int] = []
    parent_child_type: dict[int, str] = {}
    seen_parent_ids: set[int] = set()

    for child in children:
        parent_chunk_id = _to_int(child.metadata.get("parent_chunk_id"))
        if parent_chunk_id is None:
            parent_chunk_id = _to_int(child.metadata.get("chunk_id"))
        if parent_chunk_id is None:
            continue

        if parent_chunk_id in seen_parent_ids:
            continue

        if wanted_docs is not None:
            child_document_id = _to_int(child.metadata.get("document_id"))
            if child_document_id is None or child_document_id not in wanted_docs:
                continue

        seen_parent_ids.add(parent_chunk_id)
        parent_ids.append(parent_chunk_id)
        parent_child_type[parent_chunk_id] = str(child.metadata.get("child_type") or "summary")

        if len(parent_ids) >= probe_k:
            break

    return parent_ids, parent_child_type


def _metadata_document_codes(metadata: dict[str, object]) -> set[str]:
    search_optimization = metadata.get("search_optimization")
    if not isinstance(search_optimization, dict):
        return set()

    raw_codes = search_optimization.get("document_codes")
    if not isinstance(raw_codes, list):
        return set()

    return {
        str(item).upper()
        for item in raw_codes
        if str(item).strip()
    }


def _keyword_match_score(
    *,
    query_terms: list[str],
    query_codes: list[str],
    metadata: dict[str, object],
    content: str,
) -> float:
    if not query_terms and not query_codes:
        return 0.0

    keyword_blob = build_keyword_blob(metadata, content)
    normalized_blob = _normalize_lookup_text(keyword_blob)
    metadata_codes = _metadata_document_codes(metadata)

    score = 0.0
    for code in query_codes:
        if code in metadata_codes:
            score += 12.0
        elif _normalize_lookup_text(code) in normalized_blob:
            score += 6.0

    for term in query_terms:
        if term in normalized_blob:
            score += 1.0

    return score


def _keyword_parent_candidates(
    query: str,
    top_k: int,
    db: Session,
    document_ids: list[int] | None,
) -> list[int]:
    query_terms = _lookup_terms(query)
    query_codes = [code.upper() for code in extract_document_codes(query)]
    if not query_terms and not query_codes:
        return []

    chunk_query = db.query(DocumentChunk)
    if document_ids:
        chunk_query = chunk_query.filter(DocumentChunk.document_id.in_([int(item) for item in document_ids]))
    candidates = chunk_query.order_by(DocumentChunk.id.asc()).all()

    scored: list[tuple[int, float]] = []
    for candidate in candidates:
        metadata = _compact_source_metadata(candidate.source_metadata_json)
        score = _keyword_match_score(
            query_terms=query_terms,
            query_codes=query_codes,
            metadata=metadata,
            content=candidate.content,
        )
        if score <= 0:
            continue

        scored.append((candidate.id, score))

    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)

    probe_multiplier = max(1, settings.hybrid_probe_multiplier)
    probe_k = max(top_k * probe_multiplier, top_k)
    return [chunk_id for chunk_id, _ in scored[:probe_k]]


def _rrf_merge(
    vector_ids: list[int],
    keyword_ids: list[int],
    top_k: int,
) -> tuple[list[int], dict[int, float]]:
    if not vector_ids and not keyword_ids:
        return [], {}

    rrf_k = max(1, settings.hybrid_rrf_k)
    scores: dict[int, float] = defaultdict(float)

    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] += settings.hybrid_vector_rrf_weight / (rrf_k + rank)

    for rank, chunk_id in enumerate(keyword_ids, start=1):
        scores[chunk_id] += settings.hybrid_keyword_rrf_weight / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    merged_ids = [chunk_id for chunk_id, _ in ranked[:top_k]]
    return merged_ids, scores



def get_embeddings() -> Embeddings:
    """Create or return cached Ollama embedding model instance."""

    global _embeddings

    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": 180},
        )

    return _embeddings



def get_llm() -> ChatOllama:
    """Create or return cached Ollama chat model instance."""

    global _llm

    if _llm is None:
        _llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )
    return _llm



def load_index_if_available() -> bool:
    """Check whether Qdrant child collection exists."""

    client = _get_qdrant_client()
    return _qdrant_collection_exists(client)



def rebuild_index_from_chunks(chunks: list[DocumentChunk]) -> int:
    """Rebuild global Qdrant collection from all chunks available in database."""

    started_at = time.perf_counter()
    client = _get_qdrant_client()

    if not chunks:
        _emit_reindex_progress("[reindex] No chunks found. Clearing Qdrant collection.")
        _clear_qdrant_collection(client)
        return 0

    _emit_reindex_progress("[reindex] Start rebuilding Qdrant from %d parent chunks.", len(chunks))

    documents: list[Document] = []
    for chunk in chunks:
        metadata: dict[str, object] = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "parent_chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "source_page": chunk.source_page,
            "source_kind": chunk.source_kind,
        }

        source_metadata = _compact_source_metadata(chunk.source_metadata_json)
        if source_metadata:
            metadata["source_metadata"] = source_metadata

        child_chunks = build_hyq_children(source_metadata, chunk.content)
        for child_index, (child_type, child_text) in enumerate(child_chunks):
            child_metadata = dict(metadata)
            child_metadata["child_type"] = child_type
            child_metadata["child_index"] = child_index

            documents.append(
                Document(
                    page_content=child_text,
                    metadata=child_metadata,
                )
            )

    if not documents:
        _emit_reindex_progress("[reindex] No child documents generated. Clearing Qdrant collection.")
        _clear_qdrant_collection(client)
        return 0

    _emit_reindex_progress(
        "[reindex] Generated %d child documents. Creating embeddings with model '%s'.",
        len(documents),
        settings.embedding_model_name,
    )

    indexed_count = _rebuild_qdrant_collection_with_batch_embeddings(documents)

    elapsed = time.perf_counter() - started_at
    _emit_reindex_progress(
        "[reindex] Completed. Indexed %d child documents in %.2f seconds.",
        indexed_count,
        elapsed,
    )

    return indexed_count



def similarity_search(
    query: str,
    top_k: int,
    db: Session | None = None,
    document_ids: list[int] | None = None,
) -> list[Document]:
    """Search relevant parent chunks using hybrid (vector + keyword) retrieval."""

    if not load_index_if_available():
        return []

    if db is None:
        return _search_qdrant_children(query, limit=top_k)

    vector_parent_ids, parent_child_type = _vector_parent_candidates(
        query,
        top_k,
        document_ids,
    )
    keyword_parent_ids = _keyword_parent_candidates(
        query,
        top_k,
        db,
        document_ids,
    )

    merged_parent_ids, scores = _rrf_merge(vector_parent_ids, keyword_parent_ids, top_k)
    if not merged_parent_ids:
        return []

    chunks = _load_chunks_by_ids(db, merged_parent_ids)
    vector_set = set(vector_parent_ids)
    keyword_set = set(keyword_parent_ids)

    results: list[Document] = []
    for chunk in chunks:
        if chunk.id in vector_set and chunk.id in keyword_set:
            retrieval_mode = "hybrid"
        elif chunk.id in keyword_set:
            retrieval_mode = "keyword"
        else:
            retrieval_mode = "vector"

        results.append(
            _chunk_to_context_document(
                chunk,
                retrieval_mode=retrieval_mode,
                retrieval_score=scores.get(chunk.id),
                child_type=parent_child_type.get(chunk.id),
            )
        )

    return results[:top_k]



def build_sources(context_docs: list[Document]) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Extract compact source payload from retrieved chunks."""

    sources: list[dict[str, int | str | float | dict[str, object] | None]] = []
    for doc in context_docs:
        text = doc.page_content.strip().replace("\n", " ")

        source_metadata = doc.metadata.get("source_metadata")
        if not isinstance(source_metadata, dict):
            source_metadata = None

        sources.append(
            {
                "document_id": int(doc.metadata.get("document_id")) if doc.metadata.get("document_id") is not None else None,
                "chunk_id": _to_int(doc.metadata.get("chunk_id")),
                "chunk_index": _to_int(doc.metadata.get("chunk_index")),
                "page": _to_int(doc.metadata.get("source_page")),
                "source_kind": str(doc.metadata.get("source_kind")) if doc.metadata.get("source_kind") is not None else None,
                "source_metadata": source_metadata,
                "retrieval_mode": str(doc.metadata.get("retrieval_mode")) if doc.metadata.get("retrieval_mode") is not None else None,
                "retrieval_score": _to_float(doc.metadata.get("retrieval_score")),
                "excerpt": text[:280],
            }
        )
    return sources



def generate_answer(
    question: str,
    context_docs: list[Document],
    history_messages: list[ChatMessage],
) -> str:
    """Generate answer from question, retrieval context, and chat history."""

    llm = get_llm()

    history_block = "\n".join(
        f"{message.role.upper()}: {message.content}" for message in history_messages[-8:]
    )

    context_lines: list[str] = []
    for index, doc in enumerate(context_docs, start=1):
        source_metadata = doc.metadata.get("source_metadata")
        source_info = source_metadata.get("source_info") if isinstance(source_metadata, dict) else None
        context = source_metadata.get("context") if isinstance(source_metadata, dict) else None
        search_optimization = source_metadata.get("search_optimization") if isinstance(source_metadata, dict) else None

        meta_parts: list[str] = []
        if isinstance(source_info, dict):
            if source_info.get("file_name"):
                meta_parts.append(f"file={source_info.get('file_name')}")
            if source_info.get("page_number"):
                meta_parts.append(f"page={source_info.get('page_number')}")
            if source_info.get("doc_type"):
                meta_parts.append(f"doc_type={source_info.get('doc_type')}")

        if isinstance(context, dict):
            if context.get("h2"):
                meta_parts.append(f"h2={context.get('h2')}")
            if context.get("h3"):
                meta_parts.append(f"h3={context.get('h3')}")

        if isinstance(search_optimization, dict):
            document_codes = search_optimization.get("document_codes")
            if isinstance(document_codes, list) and document_codes:
                meta_parts.append(f"document_codes={', '.join(str(item) for item in document_codes[:3])}")

        retrieval_mode = doc.metadata.get("retrieval_mode")
        if retrieval_mode is not None:
            meta_parts.append(f"retrieval={retrieval_mode}")

        prefix = f"[Chunk {index}]"
        if meta_parts:
            prefix += " " + " | ".join(meta_parts)

        context_lines.append(prefix)
        context_lines.append(doc.page_content)

    context_block = "\n\n".join(context_lines)

    if not context_block:
        context_block = "No retrieved context available."

    prompt = (
        "Bạn là Tử Kỳ, một người bạn đồng hành cùng học hỏi về kiến thức công nghệ.\n"
        "Sử dụng đại từ 'mình' và 'bạn' khi trả lời để tạo cảm giác thân thiện và gần gũi.\n"
        "Use the context to answer accurately and avoid hallucination.\n"
        "Luôn bắt đầu bằng một lời dẫn dắt liên quan đến câu hỏi.\n"
        "Nếu không có đủ thông tin trong ngữ cảnh, hãy nói rõ điều đó một cách khéo léo.\n\n"
        f"Lịch sử trò chuyện:\n{history_block or 'No previous messages.'}\n\n"
        f"Ngữ cảnh:\n{context_block}\n\n"
        f"Câu hỏi: {question}\n"
        "Trả lời:"
    )

    response = llm.invoke(prompt)
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)



def parse_sources(raw_json: str | None) -> list[dict[str, int | str | float | dict[str, object] | None]]:
    """Parse serialized sources from chat message payload."""

    if not raw_json:
        return []
    try:
        data = json.loads(raw_json)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return []
