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

def _lookup_terms(query: str) -> list[str]:
    normalized = _normalize_lookup_text(query)
    terms = re.findall(r"[a-z0-9/-]+", normalized)
    return [term for term in terms if len(term) >= 2]

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
        "source_info",
        "context",
        "search_optimization",
        "admin_tags",
        "hyq",
    }
    compact = {
        key: value
        for key, value in raw_metadata.items()
        if key in wanted_keys and value is not None
    }
    if compact:
        return compact
    return raw_metadata
