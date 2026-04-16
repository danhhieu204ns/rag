from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..core.settings import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
_MARKER_PAGE_BREAK_PATTERN = re.compile(r"\n\n\{(?P<page_id>\d+)\}-+\n\n")
_MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
    ("######", "h6"),
]
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]
_marker_models: dict[str, Any] | None = None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_source_page(metadata: dict[str, Any]) -> int | None:
    source_page = _to_int(metadata.get("source_page"))
    if source_page is not None and source_page > 0:
        return source_page

    page_number = _to_int(metadata.get("page_number"))
    if page_number is not None and page_number > 0:
        return page_number

    zero_based_page = _to_int(metadata.get("page"))
    if zero_based_page is not None and zero_based_page >= 0:
        return zero_based_page + 1

    marker_page_id = _to_int(metadata.get("marker_page_id"))
    if marker_page_id is not None and marker_page_id >= 0:
        return marker_page_id + 1

    return None


def _attach_common_metadata(
    documents: list[Document],
    *,
    source_parser: str,
    source_type: str,
) -> list[Document]:
    for item in documents:
        metadata = dict(item.metadata or {})
        metadata["source_parser"] = source_parser
        metadata["source_type"] = source_type
        metadata.setdefault("source", str(metadata.get("source", "")))

        page_number = _extract_source_page(metadata)
        if page_number is not None:
            metadata["source_page"] = page_number

        item.metadata = metadata

    return documents


def _get_marker_models() -> dict[str, Any]:
    global _marker_models

    if _marker_models is None:
        from marker.models import create_model_dict

        _marker_models = create_model_dict()

    return _marker_models


def _marker_page_stats(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    page_stats = metadata.get("page_stats")
    if not isinstance(page_stats, list):
        return {}

    result: dict[int, dict[str, Any]] = {}
    for item in page_stats:
        if not isinstance(item, dict):
            continue
        page_id = _to_int(item.get("page_id"))
        if page_id is None:
            continue
        result[page_id] = item
    return result


def _split_marker_pages(markdown: str) -> list[tuple[int, str]]:
    matches = list(_MARKER_PAGE_BREAK_PATTERN.finditer(markdown))
    if not matches:
        return []

    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        marker_page_id = int(match.group("page_id"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        page_text = markdown[start:end].strip()
        if not page_text:
            continue
        pages.append((marker_page_id, page_text))

    return pages


def _write_markdown_output_log(
    file_path: Path,
    markdown: str,
    parser_name: str,
    page_count: int | None,
) -> str | None:
    """Persist parsed markdown output to disk for manual review/debugging."""

    log_dir = settings.storage_dir / "markdown_logs" / parser_name
    log_file = log_dir / f"{file_path.stem}.md"

    header_lines = [
        "<!-- parsed markdown output -->",
        f"source_file: {file_path}",
        f"parser: {parser_name}",
        f"generated_at_utc: {datetime.now(UTC).isoformat()}",
    ]
    if page_count is not None:
        header_lines.append(f"page_count: {page_count}")

    payload = "\n".join(header_lines) + "\n\n" + markdown.strip() + "\n"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file.write_text(payload, encoding="utf-8")
        return str(log_file)
    except OSError:
        return None


def _load_pdf_with_marker(file_path: Path) -> list[Document]:
    try:
        from marker.converters.pdf import PdfConverter
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ValueError(
            "PDF_PARSER_MODE=marker requires marker-pdf to be installed in backend venv. "
            "Run: pip install marker-pdf"
        ) from exc

    converter = PdfConverter(
        artifact_dict=_get_marker_models(),
        config={
            "paginate_output": True,
            "extract_images": False,
        },
    )

    rendered = converter(str(file_path))
    markdown = str(getattr(rendered, "markdown", "") or "").strip()
    if not markdown:
        return []

    marker_metadata = getattr(rendered, "metadata", {}) or {}
    page_stats = _marker_page_stats(marker_metadata)
    pages = _split_marker_pages(markdown)
    log_path = _write_markdown_output_log(
        file_path=file_path,
        markdown=markdown,
        parser_name="marker",
        page_count=len(pages) if pages else None,
    )

    if not pages:
        metadata: dict[str, Any] = {
            "source": str(file_path),
            "source_parser": "marker",
            "source_type": "pdf",
        }
        if log_path is not None:
            metadata["markdown_log_path"] = log_path

        return [
            Document(
                page_content=markdown,
                metadata=metadata,
            )
        ]

    documents: list[Document] = []
    for marker_page_id, page_text in pages:
        page_stat = page_stats.get(marker_page_id, {})

        page_metadata: dict[str, Any] = {
            "source": str(file_path),
            "source_parser": "marker",
            "source_type": "pdf",
            "source_page": marker_page_id + 1,
            "marker_page_id": marker_page_id,
        }
        if log_path is not None:
            page_metadata["markdown_log_path"] = log_path

        text_extraction_method = page_stat.get("text_extraction_method") if isinstance(page_stat, dict) else None
        if text_extraction_method is not None:
            page_metadata["marker_text_extraction_method"] = str(text_extraction_method)

        if isinstance(page_stat, dict) and page_stat.get("block_counts") is not None:
            page_metadata["marker_block_counts"] = page_stat.get("block_counts")

        documents.append(Document(page_content=page_text, metadata=page_metadata))

    return documents


def _load_pdf_with_legacy_parser(file_path: Path) -> list[Document]:
    loaded = PyPDFLoader(str(file_path)).load()
    return _attach_common_metadata(loaded, source_parser="legacy", source_type="pdf")


def _load_text_or_markdown(file_path: Path) -> list[Document]:
    loaded = TextLoader(str(file_path), encoding="utf-8").load()
    for item in loaded:
        metadata = dict(item.metadata or {})
        metadata.setdefault("source", str(file_path))
        metadata["source_parser"] = "legacy"
        metadata["source_type"] = "text"
        metadata.setdefault("source_page", 1)
        item.metadata = metadata
    return loaded


def _recursive_split(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_RECURSIVE_SEPARATORS,
    )
    return splitter.split_documents(documents)


def _markdown_header_split(documents: list[Document]) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    split_documents: list[Document] = []
    for item in documents:
        base_metadata = dict(item.metadata or {})
        text = str(item.page_content or "")
        if not text.strip():
            continue

        header_documents = splitter.split_text(text)
        if not header_documents:
            split_documents.append(Document(page_content=text, metadata=base_metadata))
            continue

        for section in header_documents:
            section_text = str(section.page_content or "").strip()
            if not section_text:
                continue

            section_metadata = dict(section.metadata or {})
            merged_metadata = dict(base_metadata)
            if section_metadata:
                merged_metadata["markdown_headers"] = section_metadata

            split_documents.append(
                Document(
                    page_content=section_text,
                    metadata=merged_metadata,
                )
            )

    return split_documents


def load_source_documents(file_path: Path) -> list[Document]:
    """Load documents from local file path based on extension."""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        if settings.pdf_parser_mode == "marker":
            return _load_pdf_with_marker(file_path)
        return _load_pdf_with_legacy_parser(file_path)
    if suffix in {".txt", ".md"}:
        return _load_text_or_markdown(file_path)
    raise ValueError(
        f"Unsupported file extension: {suffix}. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )



def split_source_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split loaded documents into chunks for embedding."""

    header_split_documents = _markdown_header_split(documents)
    if not header_split_documents:
        return []

    return _recursive_split(header_split_documents, chunk_size, chunk_overlap)
