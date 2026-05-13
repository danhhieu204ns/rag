from __future__ import annotations

import re
from dataclasses import dataclass

from .markdown_sections import MarkdownSection


INDEX_TYPE = "section_parent_child"

_TOKEN_RE = re.compile(r"\S+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True, slots=True)
class ParentChunk:
    parent_id: str
    section_id: str
    doc_id: int | str | None
    title: str
    heading_path: list[str]
    text: str
    body_text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    parent_section_id: str | None
    parent_index: int


@dataclass(frozen=True, slots=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    section_id: str
    doc_id: int | str | None
    title: str
    heading_path: list[str]
    text: str
    embedding_text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    chunk_index: int
    index_type: str = INDEX_TYPE


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(str(text or "")))


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()]
    return "\n".join(lines).strip()


def _tail_tokens(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""
    tokens = _TOKEN_RE.findall(str(text or ""))
    if not tokens:
        return ""
    return " ".join(tokens[-token_count:])


def _word_windows(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    tokens = _TOKEN_RE.findall(str(text or ""))
    if not tokens:
        return []
    if len(tokens) <= chunk_size:
        return [" ".join(tokens)]

    stride = max(1, chunk_size - max(0, min(overlap, chunk_size - 1)))
    chunks: list[str] = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def _split_units(text: str, *, chunk_size: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    units: list[str] = []
    for block in re.split(r"\n\s*\n", normalized):
        block = block.strip()
        if not block:
            continue
        if count_tokens(block) <= chunk_size:
            units.append(block)
            continue

        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(block) if item.strip()]
        if len(sentences) <= 1:
            units.extend(_word_windows(block, chunk_size=chunk_size, overlap=0))
            continue

        for sentence in sentences:
            if count_tokens(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_word_windows(sentence, chunk_size=chunk_size, overlap=0))
    return units


def split_text_to_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if count_tokens(normalized) <= chunk_size:
        return [normalized]

    units = _split_units(normalized, chunk_size=chunk_size)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    last_overlap = ""

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > chunk_size:
            emitted = "\n\n".join(item for item in current if item.strip()).strip()
            if emitted:
                chunks.append(emitted)
                last_overlap = _tail_tokens(emitted, overlap)
            current = [last_overlap] if last_overlap else []
            current_tokens = count_tokens(last_overlap)

        current.append(unit)
        current_tokens += unit_tokens

    final = "\n\n".join(item for item in current if item.strip()).strip()
    if final and final != last_overlap:
        chunks.append(final)

    return chunks


def _heading_prefix(heading_path: list[str]) -> str:
    return " > ".join(item.strip() for item in heading_path if item.strip())


def _parent_text(heading_path: list[str], body_text: str) -> str:
    prefix = _heading_prefix(heading_path)
    body = _normalize_text(body_text)
    if prefix and body:
        return f"{prefix}\n\n{body}"
    return body or prefix


def build_parent_child_chunks(
    sections: list[MarkdownSection],
    *,
    parent_max_tokens: int = 2500,
    child_chunk_size: int = 500,
    child_chunk_overlap: int = 100,
    prepend_heading_path: bool = True,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    parent_max_tokens = max(1, int(parent_max_tokens))
    child_chunk_size = max(1, int(child_chunk_size))
    child_chunk_overlap = max(0, min(int(child_chunk_overlap), child_chunk_size - 1))

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for section in sections:
        body = _normalize_text(section.content)
        if not body:
            continue

        heading_tokens = count_tokens(_heading_prefix(section.heading_path))
        body_parent_limit = max(child_chunk_size, parent_max_tokens - heading_tokens)
        parent_bodies = split_text_to_chunks(body, chunk_size=body_parent_limit, overlap=0)

        for part_index, parent_body in enumerate(parent_bodies):
            parent_id = section.section_id if len(parent_bodies) == 1 else f"{section.section_id}-parent-{part_index + 1:02d}"
            title = section.title if len(parent_bodies) == 1 else f"{section.title} Part {part_index + 1}"
            text = _parent_text(section.heading_path, parent_body)
            parent = ParentChunk(
                parent_id=parent_id,
                section_id=section.section_id,
                doc_id=section.doc_id,
                title=title,
                heading_path=list(section.heading_path),
                text=text,
                body_text=parent_body,
                token_count=count_tokens(text),
                page_start=section.page_start,
                page_end=section.page_end,
                parent_section_id=section.parent_section_id,
                parent_index=len(parents),
            )
            parents.append(parent)

            child_texts = split_text_to_chunks(
                parent_body,
                chunk_size=child_chunk_size,
                overlap=child_chunk_overlap,
            )
            for child_index, child_text in enumerate(child_texts):
                prefix = _heading_prefix(parent.heading_path)
                embedding_text = f"{prefix}\n\n{child_text}" if prepend_heading_path and prefix else child_text
                children.append(
                    ChildChunk(
                        chunk_id=f"{parent.parent_id}-child-{child_index:04d}",
                        parent_id=parent.parent_id,
                        section_id=parent.section_id,
                        doc_id=parent.doc_id,
                        title=parent.title,
                        heading_path=list(parent.heading_path),
                        text=child_text,
                        embedding_text=embedding_text,
                        token_count=count_tokens(child_text),
                        page_start=parent.page_start,
                        page_end=parent.page_end,
                        chunk_index=child_index,
                    )
                )

    return parents, children
