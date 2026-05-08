from __future__ import annotations

from pathlib import Path
from typing import Any

import logging
import httpx
from langchain_core.documents import Document

from ..core.settings import settings


logger = logging.getLogger(__name__)


def _require_service_url() -> str:
    service_url = settings.ingestion_service_url.strip()
    if not service_url:
        raise RuntimeError(
            "INGESTION_SERVICE_URL is not configured. Backend requires the ingestion service for document parse/split."
        )
    return service_url


def _service_url(path: str) -> str:
    return f"{_require_service_url()}{path}"


def _raise_service_error(response: httpx.Response) -> None:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    detail = payload.get("detail") if isinstance(payload, dict) else None
    message = str(detail or response.text or response.reason_phrase).strip()
    raise RuntimeError(
        f"Ingestion service returned HTTP {response.status_code}: {message}"
    )


def parse_source_to_markdown(file_path: Path) -> tuple[str, str, str]:
    _require_service_url()

    try:
        url = _service_url("/v1/parse")
        logger.info("[ingestion-client] POST %s file=%s", url, file_path)
        with file_path.open("rb") as source:
            files = {"file": (file_path.name, source, "application/octet-stream")}
            with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
                response = client.post(url, files=files)
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ingestion service request timed out while parsing document.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to connect to ingestion service: {exc}") from exc

    if response.status_code >= 400:
        _raise_service_error(response)
    payload = response.json()
    logger.info(
        "[ingestion-client] parse response status=%s parser=%s type=%s",
        response.status_code,
        payload.get("source_parser"),
        payload.get("source_type"),
    )
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
    _require_service_url()

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
                "source_page": 1,
            },
        )
    ]


def split_source_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    _require_service_url()

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

    try:
        url = _service_url("/v1/split")
        logger.info(
            "[ingestion-client] POST %s markdown_chars=%d chunk_size=%d overlap=%d",
            url,
            len(markdown),
            chunk_size,
            chunk_overlap,
        )
        with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
            response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ingestion service request timed out while splitting markdown.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to connect to ingestion service: {exc}") from exc

    if response.status_code >= 400:
        _raise_service_error(response)

    result = response.json()
    chunks = result.get("chunks") or []
    logger.info("[ingestion-client] split response status=%s chunks=%d", response.status_code, len(chunks))
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
