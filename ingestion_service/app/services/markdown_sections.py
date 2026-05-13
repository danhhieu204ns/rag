from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


DEFAULT_SECTION_TITLE = "Document Introduction"

_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)(?:\s+#+\s*)?$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_PAGE_MARKER_RE = re.compile(r"(?:^|\n{2,})\{(?P<page_id>\d+)\}-+\n{2,}")


@dataclass(frozen=True, slots=True)
class PageSpan:
    start: int
    end: int
    page: int | None


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    section_id: str
    title: str
    level: int
    heading_path: list[str]
    content: str
    parent_section_id: str | None
    doc_id: int | str | None
    page_start: int | None
    page_end: int | None
    start_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class _OpenSection:
    section_id: str
    title: str
    level: int
    heading_path: list[str]
    parent_section_id: str | None
    start_offset: int
    content_start_offset: int


def _page_join_separator(previous_text: str, next_text: str) -> str:
    previous_trimmed = previous_text.rstrip()
    next_trimmed = next_text.lstrip()
    if not previous_trimmed or not next_trimmed:
        return "\n\n"
    if previous_trimmed.endswith("-"):
        return ""
    if previous_trimmed[-1].isalnum() and next_trimmed[0].isalnum():
        return " "
    if previous_trimmed[-1] not in ".!?;:\n" and next_trimmed[0].isalpha():
        return " "
    return "\n\n"


def _clean_title(raw_title: str) -> str:
    title = re.sub(r"\s+", " ", str(raw_title or "").strip())
    title = title.strip(" #")
    return title or DEFAULT_SECTION_TITLE


def normalize_markdown_pages(
    markdown: str,
    *,
    source_type: str = "text",
) -> tuple[str, list[PageSpan]]:
    """Remove marker page separators while preserving character spans per page."""

    raw = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return "", []

    matches = list(_PAGE_MARKER_RE.finditer(raw))
    if not matches:
        default_page = None if source_type.strip().lower() == "pdf" else 1
        return raw, [PageSpan(start=0, end=len(raw), page=default_page)]

    segments: list[tuple[int | None, str]] = []
    prefix = raw[: matches[0].start()].strip()
    if prefix:
        segments.append((1, prefix))

    for index, match in enumerate(matches):
        page_id = int(match.group("page_id"))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        page_text = raw[start:end].strip()
        if page_text:
            segments.append((page_id + 1, page_text))

    combined_parts: list[str] = []
    spans: list[PageSpan] = []
    current_length = 0
    previous_text = ""
    for page, text in segments:
        separator = ""
        if combined_parts:
            separator = _page_join_separator(previous_text, text)
            combined_parts.append(separator)
            current_length += len(separator)

        page_start = current_length
        combined_parts.append(text)
        current_length += len(text)
        spans.append(PageSpan(start=page_start, end=current_length, page=page))
        previous_text = text

    combined = "".join(combined_parts).strip()
    return combined, spans


def _page_range_for_offsets(
    page_spans: list[PageSpan],
    *,
    start_offset: int,
    end_offset: int,
) -> tuple[int | None, int | None]:
    pages = [
        span.page
        for span in page_spans
        if span.page is not None and span.end > start_offset and span.start < end_offset
    ]
    if not pages:
        return None, None
    return min(pages), max(pages)


def parse_markdown_sections(
    markdown: str,
    *,
    source_type: str = "text",
    doc_id: int | str | None = None,
) -> list[MarkdownSection]:
    """Parse Markdown into heading-aware sections.

    Text before the first heading is emitted as "Document Introduction".
    Fenced code blocks are ignored when detecting headings.
    """

    text, page_spans = normalize_markdown_pages(markdown, source_type=source_type)
    if not text.strip():
        return []

    sections: list[MarkdownSection] = []
    stack: list[_OpenSection] = []
    current: _OpenSection | None = None
    next_id = 1
    in_fence = False
    found_heading = False

    def new_section_id() -> str:
        nonlocal next_id
        section_id = f"sec-{next_id:04d}"
        next_id += 1
        return section_id

    def append_section(open_section: _OpenSection, end_offset: int) -> None:
        content = text[open_section.content_start_offset:end_offset].strip()
        page_start, page_end = _page_range_for_offsets(
            page_spans,
            start_offset=open_section.start_offset,
            end_offset=max(end_offset, open_section.start_offset + 1),
        )
        sections.append(
            MarkdownSection(
                section_id=open_section.section_id,
                title=open_section.title,
                level=open_section.level,
                heading_path=list(open_section.heading_path),
                content=content,
                parent_section_id=open_section.parent_section_id,
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                start_offset=open_section.start_offset,
                end_offset=end_offset,
            )
        )

    def append_intro(start_offset: int, end_offset: int) -> None:
        if not text[start_offset:end_offset].strip():
            return
        intro = _OpenSection(
            section_id=new_section_id(),
            title=DEFAULT_SECTION_TITLE,
            level=0,
            heading_path=[DEFAULT_SECTION_TITLE],
            parent_section_id=None,
            start_offset=start_offset,
            content_start_offset=start_offset,
        )
        append_section(intro, end_offset)

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence

        heading_match = None if in_fence else _HEADING_RE.match(line.rstrip("\n"))
        if heading_match is not None:
            found_heading = True
            if current is None:
                append_intro(0, offset)
            else:
                append_section(current, offset)

            level = len(heading_match.group("marks"))
            title = _clean_title(heading_match.group("title"))
            while stack and stack[-1].level >= level:
                stack.pop()

            parent_section_id = stack[-1].section_id if stack else None
            heading_path = [item.title for item in stack] + [title]
            opened = _OpenSection(
                section_id=new_section_id(),
                title=title,
                level=level,
                heading_path=heading_path,
                parent_section_id=parent_section_id,
                start_offset=offset,
                content_start_offset=offset + len(line),
            )
            stack.append(opened)
            current = opened

        offset += len(line)

    if current is not None:
        append_section(current, len(text))
    elif not found_heading:
        append_intro(0, len(text))

    return sections


def section_to_metadata(section: MarkdownSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "title": section.title,
        "level": section.level,
        "heading_path": list(section.heading_path),
        "parent_section_id": section.parent_section_id,
        "doc_id": section.doc_id,
        "page_start": section.page_start,
        "page_end": section.page_end,
    }
