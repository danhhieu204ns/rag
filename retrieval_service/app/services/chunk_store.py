from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any
import re

from ..core.settings import settings


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: int
    document_id: int
    content: str
    source_page: int | None
    source_kind: str | None
    source_metadata: dict[str, Any]


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_chunks_by_ids(chunk_ids: list[int]) -> dict[int, StoredChunk]:
    if not chunk_ids or not settings.database_path.exists():
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    query = (
        "SELECT id, document_id, content, source_page, source_kind, source_metadata_json "
        f"FROM document_chunks WHERE id IN ({placeholders})"
    )

    with sqlite3.connect(str(settings.database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, chunk_ids).fetchall()

    chunks: dict[int, StoredChunk] = {}
    for row in rows:
        chunk = StoredChunk(
            chunk_id=int(row["id"]),
            document_id=int(row["document_id"]),
            content=str(row["content"] or ""),
            source_page=row["source_page"],
            source_kind=row["source_kind"],
            source_metadata=_parse_json_object(row["source_metadata_json"]),
        )
        chunks[chunk.chunk_id] = chunk
    return chunks


def _normalize_lookup_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _lookup_terms(query: str) -> list[str]:
    normalized = _normalize_lookup_text(query)
    terms = re.findall(r"[\wÀ-ỹĐđ]+", normalized, flags=re.UNICODE)
    stopwords = {
        "là",
        "và",
        "của",
        "có",
        "cho",
        "các",
        "một",
        "những",
        "nào",
        "gì",
        "the",
        "and",
        "or",
    }
    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < 2 or term in stopwords or term in seen:
            continue
        seen.add(term)
        output.append(term)
    return output


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


def _metadata_matches(metadata: dict[str, Any], filters: Any) -> bool:
    requested = getattr(filters, "metadata", None) if filters is not None else None
    if not isinstance(requested, dict):
        requested = {}
    if "index_type" not in requested and metadata.get("index_type") != "section_parent_child":
        return False
    if not requested:
        return True

    flattened = _normalize_lookup_text(" ".join(_flatten_metadata_values(metadata)))
    for key, expected in requested.items():
        if expected is None:
            continue

        candidate_values = _flatten_metadata_values(metadata.get(key))
        if not candidate_values:
            candidate_values = _flatten_metadata_values(metadata)
        candidate_blob = _normalize_lookup_text(" ".join(candidate_values or [flattened]))

        expected_values = expected if isinstance(expected, list) else [expected]
        if not any(_normalize_lookup_text(str(item)) in candidate_blob for item in expected_values):
            return False

    return True


def _keyword_score(query_terms: list[str], content: str, metadata: dict[str, Any]) -> float:
    if not query_terms:
        return 0.0

    metadata_blob = " ".join(_flatten_metadata_values(metadata))
    haystack = _normalize_lookup_text(f"{metadata_blob} {content}")
    score = 0.0
    for term in query_terms:
        occurrences = haystack.count(term)
        if occurrences:
            score += 1.0 + min(occurrences, 5) * 0.25
    return score


def search_keyword_candidates(
    *,
    query: str,
    limit: int,
    filters: Any,
) -> list[tuple[StoredChunk, float]]:
    if limit <= 0 or not settings.database_path.exists():
        return []

    query_terms = _lookup_terms(query)
    if not query_terms:
        return []

    document_ids = getattr(filters, "document_ids", None) if filters is not None else None
    params: list[Any] = []
    where = ""
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        where = f" WHERE document_id IN ({placeholders})"
        params.extend(int(item) for item in document_ids)

    sql = (
        "SELECT id, document_id, content, source_page, source_kind, source_metadata_json "
        f"FROM document_chunks{where}"
    )

    scored: list[tuple[StoredChunk, float]] = []
    with sqlite3.connect(str(settings.database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(sql, params).fetchall()

    for row in rows:
        metadata = _parse_json_object(row["source_metadata_json"])
        if not _metadata_matches(metadata, filters):
            continue

        chunk = StoredChunk(
            chunk_id=int(row["id"]),
            document_id=int(row["document_id"]),
            content=str(row["content"] or ""),
            source_page=row["source_page"],
            source_kind=row["source_kind"],
            source_metadata=metadata,
        )
        score = _keyword_score(query_terms, chunk.content, metadata)
        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda item: (item[1], -item[0].chunk_id), reverse=True)
    return scored[:limit]
