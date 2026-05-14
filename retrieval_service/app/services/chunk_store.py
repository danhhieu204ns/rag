from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Any
import re
import unicodedata

from ..core.settings import settings

logger = logging.getLogger(__name__)

BM25_FTS_TABLE = "document_chunks_bm25_fts"
BM25_META_TABLE = "document_chunks_bm25_meta"
BM25_SIGNATURE_KEY = "document_chunks_signature"
BM25_COLUMNS = (
    "title",
    "heading_path",
    "keywords",
    "aliases",
    "document_codes",
    "dates",
    "source",
    "content",
)
BM25_FIELD_WEIGHTS = (8.0, 5.0, 6.0, 6.0, 10.0, 8.0, 2.0, 1.0)
_DOC_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9]*(?:[-_/][A-Z0-9]+)+\b", re.IGNORECASE)
_DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{4})\b"
)
_BUILTIN_BILINGUAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "nghỉ phép": ("annual leave", "paid leave", "vacation leave", "leave entitlement"),
    "nghỉ ốm": ("sick leave", "medical leave"),
    "nghỉ thai sản": ("maternity leave", "parental leave"),
    "bảo hiểm": ("insurance", "social insurance"),
    "bảo hiểm xã hội": ("social insurance", "social security"),
    "hợp đồng": ("contract", "agreement"),
    "lương": ("salary", "wage", "payroll"),
    "phúc lợi": ("benefits", "employee benefits"),
    "nhân viên": ("employee", "staff"),
    "khách hàng": ("customer", "client"),
    "hóa đơn": ("invoice", "bill"),
    "thanh toán": ("payment", "settlement"),
    "quy định": ("regulation", "rule", "requirement"),
    "quy trình": ("procedure", "process", "workflow"),
    "chính sách": ("policy",),
    "hướng dẫn": ("guideline", "instruction", "manual"),
    "báo cáo": ("report",),
    "phụ lục": ("appendix", "annex"),
    "đánh giá hiệu suất": ("performance review", "performance evaluation", "appraisal"),
    "đào tạo": ("training",),
    "bảo mật": ("security", "confidentiality"),
    "quyền riêng tư": ("privacy", "data privacy"),
    "dữ liệu cá nhân": ("personal data", "personally identifiable information", "pii"),
    "điều khoản": ("terms", "terms and conditions"),
    "điều kiện": ("condition", "requirement"),
    "yêu cầu": ("requirement", "request"),
    "trách nhiệm": ("responsibility", "liability"),
    "phê duyệt": ("approval", "approve"),
    "tuân thủ": ("compliance", "comply"),
}


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


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value or ""))
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def _normalize_glossary_text(value: str) -> str:
    value = _strip_accents(value).casefold()
    value = re.sub(r"[^\wÀ-ỹĐđ]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


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
        "về",
        "với",
        "trong",
        "được",
        "không",
        "tôi",
        "muốn",
        "hỏi",
        "nêu",
        "the",
        "and",
        "or",
        "is",
        "are",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "a",
        "an",
        "what",
        "how",
        "where",
        "when",
        "why",
    }
    output: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if len(term) < 2 or term in stopwords or term in seen:
            continue
        seen.add(term)
        output.append(term)
    return output


def _dedupe_texts(values: list[str], limit: int = 120) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _normalize_lookup_text(value)
        if not cleaned:
            continue
        key = _normalize_glossary_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _extra_synonym_groups() -> dict[str, tuple[str, ...]]:
    raw = settings.bm25_extra_synonyms.strip()
    if not raw:
        return {}

    groups: dict[str, tuple[str, ...]] = {}
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        separator = "=" if "=" in group else ":"
        if separator not in group:
            continue
        key, aliases_raw = group.split(separator, 1)
        aliases = tuple(item.strip() for item in aliases_raw.split(",") if item.strip())
        if key.strip() and aliases:
            groups[key.strip()] = aliases
    return groups


def _bilingual_synonym_groups() -> dict[str, tuple[str, ...]]:
    if not settings.bm25_bilingual_expansion:
        return {}
    groups = dict(_BUILTIN_BILINGUAL_SYNONYMS)
    groups.update(_extra_synonym_groups())
    return groups


def _expand_bilingual_aliases(text: str) -> list[str]:
    groups = _bilingual_synonym_groups()
    if not groups:
        return []

    haystack = _normalize_glossary_text(text)
    if not haystack:
        return []

    aliases: list[str] = []
    for key, values in groups.items():
        related_terms = (key, *values)
        if any(_normalize_glossary_text(term) in haystack for term in related_terms):
            aliases.extend(related_terms)
    return _dedupe_texts(aliases)


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


def _search_legacy_keyword_candidates(
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


def _connect_database() -> sqlite3.Connection:
    connection = sqlite3.connect(str(settings.database_path), timeout=settings.request_timeout_seconds)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def _document_chunks_signature(connection: sqlite3.Connection) -> tuple[str, int]:
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS chunk_count,
            COALESCE(MAX(id), 0) AS max_id,
            COALESCE(MAX(created_at), '') AS max_created_at
        FROM document_chunks
        """
    ).fetchone()
    if row is None:
        return "0:0:", 0
    count = int(row["chunk_count"] or 0)
    signature = f"{count}:{int(row['max_id'] or 0)}:{row['max_created_at'] or ''}"
    return signature, count


def _ensure_bm25_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {BM25_FTS_TABLE} USING fts5(
            title,
            heading_path,
            keywords,
            aliases,
            document_codes,
            dates,
            source,
            content,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BM25_META_TABLE} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def _metadata_text_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _search_optimization_values(metadata: dict[str, Any], key: str) -> list[str]:
    search_optimization = metadata.get("search_optimization")
    if not isinstance(search_optimization, dict):
        return []
    value = search_optimization.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def _bm25_index_fields(
    *,
    content: str,
    source_kind: str | None,
    metadata: dict[str, Any],
) -> dict[str, str]:
    title_values = _dedupe_texts(
        [
            *_metadata_text_list(metadata, "title"),
            *_metadata_text_list(metadata, "section_title"),
        ],
        limit=20,
    )
    heading_values = _dedupe_texts(
        [
            *_metadata_text_list(metadata, "heading_path"),
            *_flatten_metadata_values(metadata.get("context")),
        ],
        limit=60,
    )
    keyword_values = _dedupe_texts(
        [
            *_search_optimization_values(metadata, "keywords"),
            *_search_optimization_values(metadata, "entities"),
            *_search_optimization_values(metadata, "organizations"),
            *_flatten_metadata_values(metadata.get("admin_tags")),
        ],
        limit=80,
    )
    code_values = _dedupe_texts(
        [
            *_search_optimization_values(metadata, "document_codes"),
            *_DOC_CODE_PATTERN.findall(content),
            *_DOC_CODE_PATTERN.findall(" ".join(_flatten_metadata_values(metadata))),
        ],
        limit=50,
    )
    date_values = _dedupe_texts(
        [
            *_search_optimization_values(metadata, "dates"),
            *_DATE_PATTERN.findall(content),
        ],
        limit=50,
    )

    source_info = metadata.get("source_info")
    source_values = _dedupe_texts(
        [
            str(source_kind or ""),
            *_flatten_metadata_values(source_info),
            *_metadata_text_list(metadata, "source"),
            *_metadata_text_list(metadata, "source_parser"),
            *_metadata_text_list(metadata, "source_type"),
        ],
        limit=60,
    )

    alias_source = " ".join(
        [
            *title_values,
            *heading_values,
            *keyword_values,
            *code_values,
            *date_values,
            *source_values,
            content,
        ]
    )
    alias_values = _expand_bilingual_aliases(alias_source)

    return {
        "title": " ".join(title_values),
        "heading_path": " ".join(heading_values),
        "keywords": " ".join(keyword_values),
        "aliases": " ".join(alias_values),
        "document_codes": " ".join(code_values),
        "dates": " ".join(date_values),
        "source": " ".join(source_values),
        "content": content,
    }


def _rebuild_bm25_index(connection: sqlite3.Connection, signature: str) -> None:
    rows = connection.execute(
        """
        SELECT id, content, source_kind, source_metadata_json
        FROM document_chunks
        ORDER BY id
        """
    ).fetchall()

    connection.execute(f"DELETE FROM {BM25_FTS_TABLE}")
    insert_sql = (
        f"INSERT INTO {BM25_FTS_TABLE} "
        f"(rowid, {', '.join(BM25_COLUMNS)}) "
        f"VALUES (?, {', '.join('?' for _ in BM25_COLUMNS)})"
    )
    for row in rows:
        metadata = _parse_json_object(row["source_metadata_json"])
        fields = _bm25_index_fields(
            content=str(row["content"] or ""),
            source_kind=row["source_kind"],
            metadata=metadata,
        )
        connection.execute(
            insert_sql,
            [int(row["id"]), *[fields[column] for column in BM25_COLUMNS]],
        )

    connection.execute(
        f"INSERT OR REPLACE INTO {BM25_META_TABLE} (key, value) VALUES (?, ?)",
        (BM25_SIGNATURE_KEY, signature),
    )


def _ensure_bm25_index() -> bool:
    if not settings.database_path.exists():
        return False

    try:
        with closing(_connect_database()) as connection:
            with connection:
                _ensure_bm25_schema(connection)
                signature, chunk_count = _document_chunks_signature(connection)
                stored_signature = connection.execute(
                    f"SELECT value FROM {BM25_META_TABLE} WHERE key = ?",
                    (BM25_SIGNATURE_KEY,),
                ).fetchone()
                fts_count = int(
                    connection.execute(f"SELECT COUNT(*) FROM {BM25_FTS_TABLE}").fetchone()[0] or 0
                )
                if (
                    stored_signature is None
                    or stored_signature["value"] != signature
                    or fts_count != chunk_count
                ):
                    logger.info(
                        "[bm25] rebuilding FTS index chunks=%d previous_signature=%s current_signature=%s",
                        chunk_count,
                        stored_signature["value"] if stored_signature is not None else None,
                        signature,
                    )
                    _rebuild_bm25_index(connection, signature)
        return True
    except sqlite3.Error:
        logger.exception("[bm25] FTS index unavailable; falling back to legacy keyword scan")
        return False


def _fts_tokens(value: str) -> list[str]:
    normalized = _normalize_glossary_text(value)
    tokens = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _build_fts_match_query(query: str) -> str:
    phrases = _dedupe_texts([query, *_expand_bilingual_aliases(query)], limit=80)
    conditions: list[str] = []
    seen: set[str] = set()

    def add_condition(condition: str) -> None:
        if condition and condition not in seen:
            seen.add(condition)
            conditions.append(condition)

    for phrase in phrases:
        tokens = _fts_tokens(phrase)
        if not tokens:
            continue
        if len(tokens) > 1:
            add_condition('"' + " ".join(tokens) + '"')
        for token in tokens:
            if len(token) >= 4 and not token.isdigit():
                add_condition(f"{token}*")
            else:
                add_condition(token)
        if len(conditions) >= 100:
            break

    return " OR ".join(conditions)


def _search_bm25_candidates(
    *,
    query: str,
    limit: int,
    filters: Any,
) -> list[tuple[StoredChunk, float]]:
    if limit <= 0 or not _ensure_bm25_index():
        return _search_legacy_keyword_candidates(query=query, limit=limit, filters=filters)

    match_query = _build_fts_match_query(query)
    if not match_query:
        return []

    document_ids = getattr(filters, "document_ids", None) if filters is not None else None
    params: list[Any] = [match_query]
    document_filter = ""
    if document_ids:
        placeholders = ",".join("?" for _ in document_ids)
        document_filter = f" AND dc.document_id IN ({placeholders})"
        params.extend(int(item) for item in document_ids)

    candidate_limit = max(limit * settings.bm25_candidate_limit_multiplier, limit, 100)
    weights = ", ".join(str(weight) for weight in BM25_FIELD_WEIGHTS)
    sql = (
        "SELECT dc.id, dc.document_id, dc.content, dc.source_page, dc.source_kind, "
        "dc.source_metadata_json, "
        f"-bm25({BM25_FTS_TABLE}, {weights}) AS score "
        f"FROM {BM25_FTS_TABLE} "
        f"JOIN document_chunks dc ON dc.id = {BM25_FTS_TABLE}.rowid "
        f"WHERE {BM25_FTS_TABLE} MATCH ?{document_filter} "
        "ORDER BY score DESC "
        "LIMIT ?"
    )
    params.append(candidate_limit)

    selected: list[tuple[StoredChunk, float]] = []
    try:
        with closing(_connect_database()) as connection:
            rows = connection.execute(sql, params).fetchall()
    except sqlite3.Error:
        logger.exception("[bm25] search failed; falling back to legacy keyword scan")
        return _search_legacy_keyword_candidates(query=query, limit=limit, filters=filters)

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
        selected.append((chunk, float(row["score"] or 0.0)))
        if len(selected) >= limit:
            break

    return selected


def search_keyword_candidates(
    *,
    query: str,
    limit: int,
    filters: Any,
) -> list[tuple[StoredChunk, float]]:
    if settings.bm25_enabled:
        return _search_bm25_candidates(query=query, limit=limit, filters=filters)
    return _search_legacy_keyword_candidates(query=query, limit=limit, filters=filters)
