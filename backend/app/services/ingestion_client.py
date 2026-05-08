from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_core.documents import Document

from ..core.settings import settings

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
    return (
        str(payload.get("markdown") or "").strip(),
        str(payload.get("source_parser") or "legacy").strip().lower(),
        str(payload.get("source_type") or "text").strip().lower(),
    )


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


def build_index_bundle(file_path: Path) -> dict[str, Any]:
    _require_service_url()
    try:
        url = _service_url("/v1/index/build")
        with file_path.open("rb") as source:
            files = {"file": (file_path.name, source, "application/octet-stream")}
            with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
                response = client.post(url, files=files)
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ingestion service request timed out while building index bundle.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to connect to ingestion service: {exc}") from exc

    if response.status_code >= 400:
        _raise_service_error(response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Ingestion service returned invalid index bundle payload.")
    return payload


def upsert_index_bundle(*, document_id: int, child_rows: list[dict[str, Any]]) -> int:
    _require_service_url()
    payload = {
        "document_id": int(document_id),
        "child_rows": child_rows,
    }
    try:
        url = _service_url("/v1/index/upsert")
        with httpx.Client(timeout=settings.ingestion_timeout_seconds) as client:
            response = client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ingestion service request timed out while upserting indexed chunks.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Failed to connect to ingestion service: {exc}") from exc

    if response.status_code >= 400:
        _raise_service_error(response)
    result = response.json()
    return int(result.get("indexed_chunks") or 0)
