from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document

from ..core.settings import settings
from .document_processing import (
    load_documents_from_parsed_markdown as _local_load_documents_from_parsed_markdown,
)
from .document_processing import parse_source_to_markdown as _local_parse_source_to_markdown
from .document_processing import split_source_documents as _local_split_source_documents


def _service_enabled() -> bool:
    return bool(settings.ingestion_service_url.strip())


def _service_url(path: str) -> str:
    return f"{settings.ingestion_service_url}{path}"


def parse_source_to_markdown(file_path: Path) -> tuple[str, str, str]:
    if not _service_enabled():
        return _local_parse_source_to_markdown(file_path)

    with file_path.open("rb") as source:
        files = {"file": (file_path.name, source, "application/octet-stream")}
        with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
            response = client.post(_service_url("/v1/parse"), files=files)

    response.raise_for_status()
    payload = response.json()
    return (
        str(payload.get("markdown") or "").strip(),
        str(payload.get("source_parser") or "legacy").strip().lower(),
        str(payload.get("source_type") or "text").strip().lower(),
    )


def load_documents_from_parsed_markdown(
    markdown_path: Path,
    *,
    source_file_path: Path,
    source_parser: str,
    source_type: str,
) -> list[Document]:
    if not _service_enabled():
        return _local_load_documents_from_parsed_markdown(
            markdown_path,
            source_file_path=source_file_path,
            source_parser=source_parser,
            source_type=source_type,
        )

    markdown = markdown_path.read_text(encoding="utf-8").strip()
    if not markdown:
        return []

    return [
        Document(
            page_content=markdown,
            metadata={
                "source": str(source_file_path),
                "source_parser": source_parser.strip().lower() or "legacy",
                "source_type": source_type.strip().lower() or "text",
            },
        )
    ]


def split_source_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if not _service_enabled():
        return _local_split_source_documents(
            documents,
            chunk_size,
            chunk_overlap,
        )

    if not documents:
        return []

    markdown = "\n\n".join(str(item.page_content or "") for item in documents).strip()
    if not markdown:
        return []

    first_metadata: dict[str, Any] = dict(documents[0].metadata or {})
    payload = {
        "markdown": markdown,
        "source_file_path": str(first_metadata.get("source") or "unknown"),
        "source_parser": str(first_metadata.get("source_parser") or "legacy"),
        "source_type": str(first_metadata.get("source_type") or "text"),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
    }

    with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
        response = client.post(_service_url("/v1/split"), json=payload)
    response.raise_for_status()

    result = response.json()
    chunks = result.get("chunks") or []
    output: list[Document] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        content = str(item.get("page_content") or "").strip()
        if not content:
            continue
        metadata = item.get("metadata")
        output.append(
            Document(
                page_content=content,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return output
