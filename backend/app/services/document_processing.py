from __future__ import annotations

import json
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
_HEADER_SPLIT_SPANS_KEY = "_marker_page_spans"
_HEADER_SPLIT_START_KEY = "_header_split_start_index"
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


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _is_marker_pdf_documents(documents: list[Document]) -> bool:
    if not documents:
        return False

    for item in documents:
        metadata = item.metadata or {}
        if str(metadata.get("source_parser") or "").lower() != "marker":
            return False
        if str(metadata.get("source_type") or "").lower() != "pdf":
            return False

    return True


def _clean_marker_page_text(raw_text: str) -> str:
    cleaned = re.sub(r"\n?\{\d+\}-+\n?", "\n", str(raw_text or ""))
    return cleaned.strip()


def _page_join_separator(previous_text: str, next_text: str) -> str:
    previous_trimmed = previous_text.rstrip()
    next_trimmed = next_text.lstrip()

    if not previous_trimmed or not next_trimmed:
        return "\n\n"

    if previous_trimmed.endswith("-"):
        return ""

    previous_last_char = previous_trimmed[-1]
    next_first_char = next_trimmed[0]
    if previous_last_char.isalnum() and next_first_char.isalnum():
        return " "

    if previous_last_char not in ".!?;:\n" and next_first_char.isalpha():
        return " "

    return "\n\n"


def _merge_marker_pdf_documents_for_header_split(documents: list[Document]) -> list[Document]:
    if not documents:
        return []

    merged_parts: list[str] = []
    page_spans: list[dict[str, Any]] = []
    current_length = 0
    previous_page_text = ""
    first_metadata: dict[str, Any] = {}

    for item in documents:
        metadata = dict(item.metadata or {})
        page_text = _clean_marker_page_text(str(item.page_content or ""))
        if not page_text:
            continue

        if not first_metadata:
            first_metadata = dict(metadata)

        separator = ""
        if merged_parts:
            separator = _page_join_separator(previous_page_text, page_text)
            merged_parts.append(separator)
            current_length += len(separator)

        page_start = current_length
        merged_parts.append(page_text)
        current_length += len(page_text)
        page_end = current_length

        page_spans.append(
            {
                "start": page_start,
                "end": page_end,
                "source_page": _extract_source_page(metadata),
                "marker_page_id": _to_int(metadata.get("marker_page_id")),
                "marker_text_extraction_method": metadata.get("marker_text_extraction_method"),
                "marker_block_counts": metadata.get("marker_block_counts"),
            }
        )
        previous_page_text = page_text

    merged_text = "".join(merged_parts).strip()
    if not merged_text:
        return []

    merged_metadata = dict(first_metadata)
    merged_metadata.pop("source_page", None)
    merged_metadata.pop("marker_page_id", None)
    merged_metadata.pop("marker_text_extraction_method", None)
    merged_metadata.pop("marker_block_counts", None)
    merged_metadata[_HEADER_SPLIT_SPANS_KEY] = page_spans
    merged_metadata["merged_marker_page_count"] = len(page_spans)

    return [Document(page_content=merged_text, metadata=merged_metadata)]


def _find_section_offset(text: str, section_text: str, cursor: int) -> int:
    if not section_text:
        return cursor

    found_at = text.find(section_text, cursor)
    if found_at >= 0:
        return found_at

    stripped = section_text.strip()
    if stripped:
        found_at = text.find(stripped, cursor)
        if found_at >= 0:
            return found_at

        found_at = text.find(stripped)
        if found_at >= 0:
            return found_at

    return cursor


def _resolve_page_span_for_offset(spans: Any, offset: int) -> dict[str, Any] | None:
    if not isinstance(spans, list) or not spans:
        return None

    for item in spans:
        if not isinstance(item, dict):
            continue

        start = _to_int(item.get("start"))
        end = _to_int(item.get("end"))
        if start is None or end is None:
            continue

        if start <= offset < end:
            return item

    last_item = spans[-1]
    if isinstance(last_item, dict):
        return last_item
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


def _render_pdf_with_marker(file_path: Path) -> tuple[str, dict[str, Any]]:
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
    marker_metadata = getattr(rendered, "metadata", {}) or {}
    return markdown, marker_metadata


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


def _resolve_source_from_documents(documents: list[Document]) -> str | None:
    for item in documents:
        metadata = item.metadata or {}
        source = metadata.get("source")
        if source is None:
            continue
        source_text = str(source).strip()
        if source_text:
            return source_text
    return None


def _build_header_split_log_stem(source: str | None) -> str:
    if source:
        candidate = Path(source).stem.strip()
        if candidate:
            normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
            if normalized:
                return normalized
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"header_split_{timestamp}"


def _write_header_split_output_log(
    input_documents: list[Document],
    header_split_documents: list[Document],
) -> str | None:
    """Persist markdown-header split output to disk for manual review/debugging."""

    if not header_split_documents:
        return None

    source = _resolve_source_from_documents(input_documents)
    log_dir = settings.storage_dir / "markdown_logs" / "header_split"
    log_file = log_dir / f"{_build_header_split_log_stem(source)}.md"

    header_lines = [
        "<!-- markdown header split output -->",
        f"source_file: {source or 'unknown'}",
        "split_stage: markdown_header_split",
        f"generated_at_utc: {datetime.now(UTC).isoformat()}",
        f"input_document_count: {len(input_documents)}",
        f"section_count: {len(header_split_documents)}",
    ]

    section_lines: list[str] = []
    for index, item in enumerate(header_split_documents, start=1):
        metadata = dict(item.metadata or {})
        visible_metadata = {
            key: value
            for key, value in metadata.items()
            if not str(key).startswith("_")
        }
        safe_metadata = _json_safe_value(visible_metadata)
        metadata_json = json.dumps(safe_metadata, ensure_ascii=False, indent=2)
        section_lines.extend(
            [
                f"## Section {index}",
                "",
                "```json",
                metadata_json,
                "```",
                "",
                item.page_content.strip(),
                "",
                "---",
                "",
            ]
        )

    payload = "\n".join(header_lines) + "\n\n" + "\n".join(section_lines)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file.write_text(payload, encoding="utf-8")
        return str(log_file)
    except OSError:
        return None


def _load_pdf_with_marker(file_path: Path) -> list[Document]:
    markdown, marker_metadata = _render_pdf_with_marker(file_path)
    if not markdown:
        return []

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


def _markdown_documents_for_pdf(
    markdown: str,
    *,
    source: str,
    source_parser: str,
) -> list[Document]:
    pages = _split_marker_pages(markdown)
    if not pages:
        return [
            Document(
                page_content=markdown,
                metadata={
                    "source": source,
                    "source_parser": source_parser,
                    "source_type": "pdf",
                },
            )
        ]

    return [
        Document(
            page_content=page_text,
            metadata={
                "source": source,
                "source_parser": source_parser,
                "source_type": "pdf",
                "source_page": marker_page_id + 1,
                "marker_page_id": marker_page_id,
            },
        )
        for marker_page_id, page_text in pages
        if page_text.strip()
    ]


def parse_source_to_markdown(file_path: Path) -> tuple[str, str, str]:
    """Parse one source file to markdown text and return parser/source metadata."""

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension: {suffix}. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        if settings.pdf_parser_mode == "marker":
            markdown, _ = _render_pdf_with_marker(file_path)
            return markdown.strip(), "marker", "pdf"

        pages = _load_pdf_with_legacy_parser(file_path)
        markdown = "\n\n".join(str(item.page_content or "").strip() for item in pages if str(item.page_content or "").strip())
        return markdown.strip(), "legacy", "pdf"

    if suffix == ".md":
        markdown = file_path.read_text(encoding="utf-8")
        return markdown.strip(), "legacy", "text"

    loaded = _load_text_or_markdown(file_path)
    markdown = "\n\n".join(str(item.page_content or "").strip() for item in loaded if str(item.page_content or "").strip())
    return markdown.strip(), "legacy", "text"


def load_documents_from_parsed_markdown(
    markdown_path: Path,
    *,
    source_file_path: Path,
    source_parser: str,
    source_type: str,
) -> list[Document]:
    """Load split-ready documents from parsed markdown generated in Step A."""

    markdown = markdown_path.read_text(encoding="utf-8").strip()
    if not markdown:
        return []

    source = str(source_file_path)
    normalized_source_type = source_type.strip().lower()
    normalized_parser = source_parser.strip().lower()

    if normalized_source_type == "pdf":
        return _markdown_documents_for_pdf(
            markdown,
            source=source,
            source_parser=normalized_parser or "marker",
        )

    return [
        Document(
            page_content=markdown,
            metadata={
                "source": source,
                "source_parser": normalized_parser or "legacy",
                "source_type": "text",
                "source_page": 1,
            },
        )
    ]


def _recursive_split(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_RECURSIVE_SEPARATORS,
        add_start_index=True,
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
        cursor = 0
        if not header_documents:
            fallback_metadata = dict(base_metadata)
            fallback_metadata[_HEADER_SPLIT_START_KEY] = 0
            split_documents.append(Document(page_content=text.strip(), metadata=fallback_metadata))
            continue

        for section in header_documents:
            section_raw_text = str(section.page_content or "")
            section_text = section_raw_text.strip()
            if not section_text:
                continue

            section_start = _find_section_offset(text, section_raw_text, cursor)
            cursor = max(cursor, section_start + len(section_raw_text))

            section_metadata = dict(section.metadata or {})
            merged_metadata = dict(base_metadata)
            if section_metadata:
                merged_metadata["markdown_headers"] = section_metadata
            merged_metadata[_HEADER_SPLIT_START_KEY] = section_start

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

    documents_for_header_split = documents
    if _is_marker_pdf_documents(documents):
        documents_for_header_split = _merge_marker_pdf_documents_for_header_split(documents)

    header_split_documents = _markdown_header_split(documents_for_header_split)
    if not header_split_documents:
        return []

    log_path = _write_header_split_output_log(documents_for_header_split, header_split_documents)
    if log_path is not None:
        for item in header_split_documents:
            metadata = dict(item.metadata or {})
            metadata["header_split_log_path"] = log_path
            item.metadata = metadata

    recursive_documents = _recursive_split(header_split_documents, chunk_size, chunk_overlap)
    for item in recursive_documents:
        metadata = dict(item.metadata or {})

        section_start = _to_int(metadata.get(_HEADER_SPLIT_START_KEY))
        chunk_start = _to_int(metadata.get("start_index"))
        if section_start is not None and chunk_start is not None:
            absolute_offset = section_start + chunk_start
            span = _resolve_page_span_for_offset(metadata.get(_HEADER_SPLIT_SPANS_KEY), absolute_offset)
            if span is not None:
                source_page = _to_int(span.get("source_page"))
                marker_page_id = _to_int(span.get("marker_page_id"))
                if source_page is not None:
                    metadata["source_page"] = source_page
                if marker_page_id is not None:
                    metadata["marker_page_id"] = marker_page_id

                extraction_method = span.get("marker_text_extraction_method")
                if extraction_method is not None:
                    metadata["marker_text_extraction_method"] = str(extraction_method)

                if span.get("marker_block_counts") is not None:
                    metadata["marker_block_counts"] = span.get("marker_block_counts")

        metadata.pop(_HEADER_SPLIT_SPANS_KEY, None)
        metadata.pop(_HEADER_SPLIT_START_KEY, None)
        item.metadata = metadata

    return recursive_documents
