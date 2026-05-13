from __future__ import annotations

import json
import time
import unicodedata
import re
from typing import Any

def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    return str(value)

def _preview_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

def _preview_ids(values: list[int], limit: int = 8) -> list[int | str]:
    if len(values) <= limit:
        return values
    return [*values[:limit], f"+{len(values) - limit} more"]

def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)

def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def _to_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _normalize_lookup_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return " ".join(without_marks.lower().split())

# Vietnamese function words that carry no retrieval signal
_VI_STOP_WORDS: frozenset[str] = frozenset({
    "la", "gi", "cua", "va", "co", "do", "de", "ra", "da", "an",
    "hay", "se", "bi", "duoc", "voi", "thi", "ma", "khi", "neu",
    "tu", "sau", "trong", "ngoai", "tren", "duoi", "nhu", "the",
    "nay", "kia", "ay", "ho", "ta", "ban", "minh", "no", "chung",
    "cac", "nhung", "mot", "hai", "ba", "bon", "nam", "sau",
    "nhieu", "moi", "tat", "ca", "vi", "den", "len", "xuong",
})


def _lookup_terms(query: str) -> list[str]:
    normalized = _normalize_lookup_text(query)
    terms = re.findall(r"[a-z0-9/-]+", normalized)
    return [term for term in terms if len(term) >= 2 and term not in _VI_STOP_WORDS]

def _parse_chunk_source_metadata(raw_json: str | None) -> dict[str, object]:
    if not raw_json:
        return {}
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return {}
        return data
    except json.JSONDecodeError:
        return {}

def _compact_source_metadata(raw_json: str | None) -> dict[str, object]:
    raw_metadata = _parse_chunk_source_metadata(raw_json)
    if not raw_metadata:
        return {}

    wanted_keys = {
        "chunk_id",
        "child_chunk_id",
        "parent_id",
        "parent_chunk_id",
        "section_id",
        "section_title",
        "title",
        "heading_path",
        "page_start",
        "page_end",
        "index_type",
        "source_info",
        "context",
        "search_optimization",
        "admin_tags",
    }
    compact = {
        key: value
        for key, value in raw_metadata.items()
        if key in wanted_keys and value is not None
    }
    if compact:
        return compact
    return raw_metadata


def _segment_vietnamese(text: str) -> str:
    """Word-segment Vietnamese text using underthesea if installed, else return as-is."""
    try:
        from underthesea import word_tokenize  # type: ignore
        return word_tokenize(text, format="text")
    except ImportError:
        return text


def _build_full_text_search(source_metadata: dict[str, object], content: str) -> str:
    """Build augmented search text from section headings, metadata, and content."""
    parts: list[str] = []

    heading_path = source_metadata.get("heading_path")
    if isinstance(heading_path, list):
        parts.extend(str(item).strip() for item in heading_path if str(item).strip())

    section_title = source_metadata.get("section_title") or source_metadata.get("title")
    if isinstance(section_title, str) and section_title.strip():
        parts.append(section_title.strip())

    context = source_metadata.get("context") or {}
    if isinstance(context, dict):
        for value in context.values():
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
            elif isinstance(value, list):
                parts.extend(str(item).strip() for item in value if str(item).strip())

    if content:
        parts.append(content.strip())

    search_opt = source_metadata.get("search_optimization") or {}
    if isinstance(search_opt, dict):
        for key in ("keywords", "entities", "organizations"):
            vals = search_opt.get(key) or []
            if isinstance(vals, list):
                parts.append(" ".join(str(v) for v in vals if v))

    full_text = " ".join(p for p in parts if p)
    return _segment_vietnamese(full_text)
