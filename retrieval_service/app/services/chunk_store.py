from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

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
