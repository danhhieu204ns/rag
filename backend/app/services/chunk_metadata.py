from __future__ import annotations

import re
from typing import Any

_DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
_DOCUMENT_CODE_PATTERN = re.compile(
    r"\b\d{1,6}[/-][A-Za-zĐđ]{1,12}(?:[/-][A-Za-z0-9Đđ]{1,16})+\b"
)


def _normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def _dedupe_keep_order(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for raw in items:
        cleaned = _normalize_spaces(raw)
        if not cleaned:
            continue

        key = cleaned.casefold()
        if key in seen:
            continue

        output.append(cleaned)
        seen.add(key)

        if len(output) >= limit:
            break

    return output


def extract_document_codes(text: str, limit: int = 20) -> list[str]:
    raw_codes = [match.group(0).upper() for match in _DOCUMENT_CODE_PATTERN.finditer(str(text or ""))]
    return _dedupe_keep_order(raw_codes, limit)


def _extract_dates(text: str, limit: int = 20) -> list[str]:
    raw_dates = [match.group(0) for match in _DATE_PATTERN.finditer(str(text or ""))]
    return _dedupe_keep_order(raw_dates, limit)


def _flatten_metadata_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_flatten_metadata_values(item))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_metadata_values(item))
        return values
    return [str(value)]


def build_keyword_blob(metadata: dict[str, Any], content: str) -> str:
    parts: list[str] = []

    for key in ("title", "section_title", "source_kind", "index_type"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))

    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list):
        parts.extend(str(item) for item in heading_path if str(item).strip())

    source_info = metadata.get("source_info")
    if isinstance(source_info, dict):
        parts.extend(_flatten_metadata_values(source_info))

    context = metadata.get("context")
    if isinstance(context, dict):
        parts.extend(_flatten_metadata_values(context))

    search_optimization = metadata.get("search_optimization")
    if isinstance(search_optimization, dict):
        for key in ("keywords", "entities", "organizations", "dates", "document_codes"):
            value = search_optimization.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value)

    admin_tags = metadata.get("admin_tags")
    if isinstance(admin_tags, dict):
        parts.extend(
            [
                str(admin_tags.get("security_level") or ""),
                str(admin_tags.get("department") or ""),
            ]
        )

    parts.extend(extract_document_codes(content))
    parts.extend(_extract_dates(content))
    parts.append(str(content or ""))

    return _normalize_spaces(" ".join(parts))
