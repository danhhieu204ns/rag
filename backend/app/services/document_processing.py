from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


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
) -> list[Document]:
    """
        Split loaded documents into chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)
