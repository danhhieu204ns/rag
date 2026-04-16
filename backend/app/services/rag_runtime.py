from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ..core.settings import settings
from ..models import ChatMessage, DocumentChunk

_embeddings: Embeddings | None = None
_vectorstore: FAISS | None = None
_llm: ChatOllama | None = None


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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
        "source",
        "source_parser",
        "source_type",
        "marker_page_id",
        "marker_text_extraction_method",
        "marker_block_counts",
    }
    compact = {
        key: value
        for key, value in raw_metadata.items()
        if key in wanted_keys and value is not None
    }
    return compact


def _index_files_exist(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()



def get_embeddings() -> Embeddings:
    """Create or return cached Ollama embedding model instance."""

    global _embeddings

    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=settings.embedding_model_name,
            base_url=settings.ollama_base_url,
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



def load_index_if_available() -> FAISS | None:
    """Load FAISS index from disk once and cache it in memory."""

    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    if _index_files_exist(settings.index_dir):
        _vectorstore = FAISS.load_local(
            str(settings.index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    return _vectorstore



def rebuild_index_from_chunks(chunks: list[DocumentChunk]) -> int:
    """Rebuild global FAISS index from all chunks available in database."""

    global _vectorstore

    if not chunks:
        if settings.index_dir.exists():
            shutil.rmtree(settings.index_dir)
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        _vectorstore = None
        return 0

    documents: list[Document] = []
    for chunk in chunks:
        metadata: dict[str, object] = {
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "source_page": chunk.source_page,
            "source_kind": chunk.source_kind,
        }

        source_metadata = _compact_source_metadata(chunk.source_metadata_json)
        if source_metadata:
            metadata["source_metadata"] = source_metadata

        documents.append(
            Document(
                page_content=chunk.content,
                metadata=metadata,
            )
        )

    vectorstore = FAISS.from_documents(documents, get_embeddings())
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(settings.index_dir))
    _vectorstore = vectorstore

    return len(documents)



def similarity_search(
    query: str,
    top_k: int,
    document_ids: list[int] | None = None,
) -> list[Document]:
    """Search relevant chunks from global FAISS index."""

    vectorstore = load_index_if_available()
    if vectorstore is None:
        return []

    probe_k = max(top_k * 3, top_k)
    documents = vectorstore.similarity_search(query, k=probe_k)

    if document_ids:
        wanted = {int(doc_id) for doc_id in document_ids}
        documents = [
            item
            for item in documents
            if int(item.metadata.get("document_id", -1)) in wanted
        ]

    return documents[:top_k]



def build_sources(context_docs: list[Document]) -> list[dict[str, int | str | dict[str, object] | None]]:
    """Extract compact source payload from retrieved chunks."""

    sources: list[dict[str, int | str | dict[str, object] | None]] = []
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

    context_block = "\n\n".join(
        f"[Chunk {index}] {doc.page_content}" for index, doc in enumerate(context_docs, start=1)
    )

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



def parse_sources(raw_json: str | None) -> list[dict[str, int | str | dict[str, object] | None]]:
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
