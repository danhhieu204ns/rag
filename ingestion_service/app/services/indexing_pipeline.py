from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .markdown_sections import parse_markdown_sections
from .parent_child_chunker import INDEX_TYPE, ChildChunk, ParentChunk, build_parent_child_chunks


logger = logging.getLogger(__name__)

_DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
_DOC_CODE_PATTERN = re.compile(r"\b\d{1,6}[/-][A-Za-z]{1,12}(?:[/-][A-Za-z0-9]{1,16})+\b")


@dataclass(frozen=True, slots=True)
class SectionParentChildIndex:
    parents: list[ParentChunk]
    children: list[ChildChunk]
    parent_payloads: list[dict[str, Any]]
    child_payloads: list[dict[str, Any]]


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _source_kind(source_parser: str, source_type: str, suffix: str) -> str:
    normalized_parser = source_parser.strip().lower() or "legacy"
    normalized_type = source_type.strip().lower() or ("pdf" if suffix == ".pdf" else "text")
    if normalized_type == "pdf" and normalized_parser == "marker":
        return "pdf_marker_section"
    if normalized_type == "pdf":
        return "pdf_section"
    if normalized_type == "text":
        return "text_section"
    return normalized_type


def _dedupe(items: list[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = " ".join(str(item or "").split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _search_metadata(text: str, heading_path: list[str]) -> dict[str, list[str]]:
    doc_codes = _dedupe([item.upper() for item in _DOC_CODE_PATTERN.findall(text)], 15)
    dates = _dedupe(_DATE_PATTERN.findall(text), 15)
    heading_terms = _dedupe(heading_path, 10)
    keywords = _dedupe([*heading_terms, *doc_codes, *dates], 20)
    return {
        "keywords": keywords,
        "entities": [],
        "organizations": [],
        "dates": dates,
        "document_codes": doc_codes,
    }


def _context_metadata(heading_path: list[str]) -> dict[str, Any]:
    context: dict[str, Any] = {"heading_path": list(heading_path)}
    for index, title in enumerate(heading_path[:6], start=1):
        context[f"h{index}"] = title
    return context


def _base_metadata(
    *,
    file_name: str,
    source_parser: str,
    source_type: str,
    source_kind: str,
    parent: ParentChunk,
) -> dict[str, Any]:
    page_number = parent.page_start
    return {
        "index_type": INDEX_TYPE,
        "parent_id": parent.parent_id,
        "section_id": parent.section_id,
        "section_title": parent.title,
        "title": parent.title,
        "heading_path": list(parent.heading_path),
        "parent_section_id": parent.parent_section_id,
        "page_start": parent.page_start,
        "page_end": parent.page_end,
        "token_count": parent.token_count,
        "source_parser": source_parser,
        "source_type": source_type,
        "source_kind": source_kind,
        "source_info": {
            "file_name": file_name,
            "page_number": page_number,
            "doc_type": "document",
        },
        "context": _context_metadata(parent.heading_path),
        "search_optimization": _search_metadata(parent.text, parent.heading_path),
        "admin_tags": {
            "security_level": "internal",
            "department": "general",
        },
        "chunking": {
            "strategy": INDEX_TYPE,
            "role": "parent",
        },
    }


def build_section_parent_child_index(
    *,
    markdown: str,
    source_file_path: Path,
    source_parser: str,
    source_type: str,
    parent_max_tokens: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    prepend_heading_path: bool,
) -> SectionParentChildIndex:
    started = time.perf_counter()
    cleaned_markdown = str(markdown or "").strip()
    if not cleaned_markdown:
        raise ValueError("Parsed markdown is empty.")

    sections = parse_markdown_sections(cleaned_markdown, source_type=source_type)
    logger.info(
        "[ingestion][indexing] parsed_sections=%d source=%s",
        len(sections),
        source_file_path,
    )

    parents, children = build_parent_child_chunks(
        sections,
        parent_max_tokens=parent_max_tokens,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        prepend_heading_path=prepend_heading_path,
    )
    logger.info(
        "[ingestion][indexing] built_parent_child parents=%d child_chunks=%d parent_max_tokens=%d child_size=%d child_overlap=%d",
        len(parents),
        len(children),
        parent_max_tokens,
        child_chunk_size,
        child_chunk_overlap,
    )

    if not parents or not children:
        raise ValueError("Markdown did not produce indexable section chunks.")

    suffix = source_file_path.suffix.lower()
    source_kind = _source_kind(source_parser, source_type, suffix)
    file_name = source_file_path.name

    children_by_parent: dict[str, list[ChildChunk]] = {}
    for child in children:
        children_by_parent.setdefault(child.parent_id, []).append(child)

    parent_payloads: list[dict[str, Any]] = []
    for parent in parents:
        metadata = _base_metadata(
            file_name=file_name,
            source_parser=source_parser,
            source_type=source_type,
            source_kind=source_kind,
            parent=parent,
        )
        metadata["child_chunk_count"] = len(children_by_parent.get(parent.parent_id, []))
        parent_payloads.append(
            {
                "parent_id": parent.parent_id,
                "section_id": parent.section_id,
                "chunk_index": parent.parent_index,
                "content": parent.text,
                "title": parent.title,
                "heading_path": list(parent.heading_path),
                "token_count": parent.token_count,
                "page_start": parent.page_start,
                "page_end": parent.page_end,
                "source_page": parent.page_start,
                "source_kind": source_kind,
                "source_metadata": metadata,
            }
        )

    parent_by_id = {parent.parent_id: parent for parent in parents}
    child_payloads: list[dict[str, Any]] = []
    for child in children:
        parent = parent_by_id[child.parent_id]
        parent_metadata = _base_metadata(
            file_name=file_name,
            source_parser=source_parser,
            source_type=source_type,
            source_kind=source_kind,
            parent=parent,
        )
        child_metadata = dict(parent_metadata)
        child_metadata.update(
            {
                "chunk_id": child.chunk_id,
                "child_chunk_id": child.chunk_id,
                "child_index": child.chunk_index,
                "child_type": "section_child",
                "child_token_count": child.token_count,
                "token_count": child.token_count,
                "chunking": {
                    "strategy": INDEX_TYPE,
                    "role": "child",
                },
            }
        )
        child_payloads.append(
            {
                "chunk_index": parent.parent_index,
                "chunk_id": child.chunk_id,
                "parent_id": child.parent_id,
                "section_id": child.section_id,
                "child_type": "section_child",
                "child_index": child.chunk_index,
                "child_text": child.text,
                "embedding_text": child.embedding_text,
                "token_count": child.token_count,
                "title": child.title,
                "section_title": child.title,
                "heading_path": list(child.heading_path),
                "page_start": child.page_start,
                "page_end": child.page_end,
                "source_page": child.page_start,
                "source_kind": source_kind,
                "index_type": INDEX_TYPE,
                "source_metadata": child_metadata,
            }
        )

    logger.info(
        "[ingestion][timing] step=section_parent_child_index status=ok elapsed_ms=%.2f sections=%d parents=%d child_chunks=%d embedding_count=%d",
        _ms(started),
        len(sections),
        len(parent_payloads),
        len(child_payloads),
        len(child_payloads),
    )
    return SectionParentChildIndex(
        parents=parents,
        children=children,
        parent_payloads=parent_payloads,
        child_payloads=child_payloads,
    )
