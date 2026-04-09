from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, Language, RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
ChunkingMethod = Literal["recursive", "character", "markdown", "python"]

MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n- ",
    "\n* ",
    "\n",
    " ",
    "",
]



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
    chunking_method: ChunkingMethod = "recursive",
) -> list[Document]:
    """Split loaded documents into chunks for embedding.

    Supported methods: recursive, character, markdown, python.
    """

    if chunking_method == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif chunking_method == "character":
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif chunking_method == "markdown":
        splitter = RecursiveCharacterTextSplitter(
            separators=MARKDOWN_SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif chunking_method == "python":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(
            f"Unsupported chunking method: {chunking_method}. "
            "Supported methods: recursive, character, markdown, python"
        )

    return splitter.split_documents(documents)
