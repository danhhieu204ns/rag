from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _recursive_split(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def _semantic_split(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    embeddings: Embeddings,
) -> list[Document]:
    semantic_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
    )
    semantic_chunks = semantic_splitter.split_documents(documents)

    if not semantic_chunks:
        return _recursive_split(documents, chunk_size, chunk_overlap)

    # Optional post-split cap keeps chunks bounded for vector indexing.
    return _recursive_split(semantic_chunks, chunk_size, chunk_overlap)


def load_source_documents(file_path: Path) -> list[Document]:
    """Load documents from local file path based on extension."""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(file_path), encoding="utf-8").load()
    raise ValueError(
        f"Unsupported file extension: {suffix}. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )



def split_source_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    chunking_method: Literal["recursive", "semantic"] = "recursive",
    embeddings: Embeddings | None = None,
) -> list[Document]:
    """
        Split loaded documents into chunks for embedding.
    """
    if chunking_method == "semantic":
        if embeddings is None:
            raise ValueError("Semantic chunking requires an embeddings instance.")
        return _semantic_split(documents, chunk_size, chunk_overlap, embeddings)

    return _recursive_split(documents, chunk_size, chunk_overlap)
