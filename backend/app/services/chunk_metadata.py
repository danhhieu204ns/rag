from __future__ import annotations

import atexit
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..core.settings import settings
from .ollama_service_client import ollama_service_headers, require_ollama_service_url

logger = logging.getLogger(__name__)

_CAPITALIZED_PHRASE_PATTERN = re.compile(
    r"\b[A-ZÀ-Ỵ][A-Za-zÀ-ỹĐđ'\-]*(?:\s+[A-ZÀ-Ỵ][A-Za-zÀ-ỹĐđ'\-]*){1,4}\b"
)
_DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
_DOCUMENT_CODE_PATTERN = re.compile(
    r"\b\d{1,6}[/-][A-Za-zĐđ]{1,12}(?:[/-][A-Za-z0-9Đđ]{1,16})+\b"
)
_ORG_HINTS = (
    "bộ ",
    "đội ",
    "ban ",
    "phòng ",
    "trung tâm",
    "cục ",
    "tập đoàn",
    "công ty",
    "học viện",
    "viện ",
    "quân khu",
    "viettel",
    "qutw",
)


@dataclass(slots=True)
class HyQResult:
    summary: str
    questions: list[str]


@dataclass(slots=True)
class ChunkEnrichmentResult:
    hyq: HyQResult | None
    entities: list[str]


_SEARCH_OPT_LIMITS: dict[str, int] = {
    "keywords": 20,
    "entities": 15,
    "organizations": 15,
    "dates": 15,
    "document_codes": 15,
}


def _normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def _word_limited_text(text: str, max_words: int) -> str:
    words = _normalize_spaces(text).split(" ")
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


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


def _payload_list(payload: dict[str, Any], key: str, limit: int) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return _dedupe_keep_order([str(item) for item in value if str(item).strip()], limit=limit)


def _build_fallback_keywords(
    *,
    entities: list[str],
    organizations: list[str],
    dates: list[str],
    document_codes: list[str],
) -> list[str]:
    combined = [
        *document_codes,
        *organizations,
        *entities,
        *dates,
    ]
    return _dedupe_keep_order(combined, limit=_SEARCH_OPT_LIMITS["keywords"])



def _parse_hyq_payload(
    payload: dict[str, Any],
    *,
    summary_words: int,
    question_count: int,
) -> HyQResult | None:
    summary = _word_limited_text(str(payload.get("summary") or ""), summary_words)
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        return None

    cleaned_questions = [str(item) for item in raw_questions if str(item).strip()]
    deduped = _dedupe_keep_order(cleaned_questions, limit=question_count)
    if not summary or not deduped:
        return None

    while len(deduped) < question_count:
        deduped.append(f"Thông tin chính của đoạn này là gì? ({len(deduped) + 1})")

    normalized_questions = [question if question.endswith("?") else f"{question}?" for question in deduped]
    return HyQResult(summary=summary, questions=normalized_questions)


def extract_document_codes(text: str, limit: int = 20) -> list[str]:
    raw_codes = [match.group(0).upper() for match in _DOCUMENT_CODE_PATTERN.finditer(str(text or ""))]
    return _dedupe_keep_order(raw_codes, limit)


def _extract_dates(text: str, limit: int = 20) -> list[str]:
    raw_dates = [match.group(0) for match in _DATE_PATTERN.finditer(str(text or ""))]
    return _dedupe_keep_order(raw_dates, limit)


def _split_named_phrases(text: str) -> tuple[list[str], list[str]]:
    entities: list[str] = []
    organizations: list[str] = []

    for match in _CAPITALIZED_PHRASE_PATTERN.finditer(str(text or "")):
        phrase = _normalize_spaces(match.group(0))
        if len(phrase) < 3:
            continue

        normalized = phrase.casefold()
        if any(hint in normalized for hint in _ORG_HINTS):
            organizations.append(phrase)
            continue

        if phrase.isupper() and len(phrase) <= 25:
            organizations.append(phrase)
            continue

        entities.append(phrase)

    return (
        _dedupe_keep_order(entities, limit=30),
        _dedupe_keep_order(organizations, limit=30),
    )


def _extract_context(raw_metadata: dict[str, Any]) -> dict[str, str | None]:
    headers = raw_metadata.get("markdown_headers")
    if not isinstance(headers, dict):
        return {"h2": None, "h3": None}

    h2 = headers.get("h2") or headers.get("h1")
    h3 = headers.get("h3") or headers.get("h4")

    return {
        "h2": _normalize_spaces(str(h2)) if h2 else None,
        "h3": _normalize_spaces(str(h3)) if h3 else None,
    }


def _infer_doc_type(file_name: str, raw_metadata: dict[str, Any]) -> str:
    existing = raw_metadata.get("doc_type")
    if existing is not None:
        cleaned = _normalize_spaces(str(existing))
        if cleaned:
            return cleaned

    text = _normalize_spaces(file_name).casefold()
    if "lịch sử" in text:
        return "Quy_trình_Lịch_sử"
    if "quy trình" in text:
        return "Quy_trình"
    if "hướng dẫn" in text:
        return "Hướng_dẫn"
    if "quy định" in text:
        return "Quy_định"
    return "Tài_liệu_nội_bộ"


def _normalize_security_level(level: str | None) -> str | None:
    if not level:
        return None

    normalized = _normalize_spaces(level).casefold()
    if normalized in {"public", "cong khai", "công khai"}:
        return "Công_khai"
    if normalized in {"confidential", "mat", "mật", "bi mat", "bí mật"}:
        return "Mật"
    if normalized in {"internal", "noi bo", "nội bộ"}:
        return "Nội_bộ"
    return None


def _infer_security_level(file_name: str, raw_metadata: dict[str, Any]) -> str:
    admin_tags = raw_metadata.get("admin_tags")
    if isinstance(admin_tags, dict):
        from_admin = _normalize_security_level(
            str(admin_tags.get("security_level")) if admin_tags.get("security_level") else None
        )
        if from_admin:
            return from_admin

    from_direct = _normalize_security_level(
        str(raw_metadata.get("security_level")) if raw_metadata.get("security_level") else None
    )
    if from_direct:
        return from_direct

    file_hint = _normalize_spaces(file_name).casefold()
    if any(keyword in file_hint for keyword in ("confidential", "mật", "mat", "bí_mật", "bi_mat", "tuyệt_mật", "tuyet_mat")):
        return "Mật"
    if any(keyword in file_hint for keyword in ("public", "công_khai", "cong_khai")):
        return "Công_khai"
    return "Nội_bộ"


def _infer_department(raw_metadata: dict[str, Any], context: dict[str, str | None]) -> str:
    admin_tags = raw_metadata.get("admin_tags")
    if isinstance(admin_tags, dict) and admin_tags.get("department"):
        return _normalize_spaces(str(admin_tags["department"]))

    if raw_metadata.get("department"):
        return _normalize_spaces(str(raw_metadata["department"]))

    text = _normalize_spaces(" ".join(filter(None, [context.get("h2"), context.get("h3")]))).casefold()
    if "chính trị" in text or "chinh tri" in text:
        return "Chính_trị"
    if "kỹ thuật" in text or "ky thuat" in text:
        return "Kỹ_thuật"
    if "nhân sự" in text or "nhan su" in text:
        return "Nhân_sự"
    if "truyền thông" in text or "truyen thong" in text:
        return "Truyền_thông"
    return "Tổng_hợp"


def _fallback_hyq(
    *,
    chunk_text: str,
    context: dict[str, str | None],
    search_optimization: dict[str, list[str]],
    summary_words: int,
    question_count: int,
) -> HyQResult:
    summary = _word_limited_text(chunk_text, max_words=max(20, summary_words))

    questions: list[str] = []
    document_codes = search_optimization.get("document_codes") or []
    organizations = search_optimization.get("organizations") or []
    entities = search_optimization.get("entities") or []
    dates = search_optimization.get("dates") or []

    if document_codes:
        questions.append(f"Văn bản {document_codes[0]} nói về nội dung gì?")

    if context.get("h3"):
        questions.append(f"{context['h3']} được trình bày như thế nào?")
    elif context.get("h2"):
        questions.append(f"Nội dung trong mục {context['h2']} là gì?")

    if organizations:
        questions.append(f"{organizations[0]} có vai trò gì trong đoạn này?")

    if entities:
        questions.append(f"{entities[0]} được nhắc đến trong bối cảnh nào?")

    if dates:
        questions.append(f"Sự kiện nào xảy ra vào mốc {dates[0]}?")

    questions.append("Đoạn này trả lời câu hỏi trong hoàn cảnh nào?")

    deduped = _dedupe_keep_order(questions, limit=question_count)
    while len(deduped) < question_count:
        deduped.append(f"Thông tin chính của đoạn này là gì? ({len(deduped) + 1})")

    normalized_questions = [question if question.endswith("?") else f"{question}?" for question in deduped]

    return HyQResult(summary=summary, questions=normalized_questions)


def _fallback_search_optimization(chunk_text: str) -> dict[str, list[str]]:
    entities, organizations = _split_named_phrases(chunk_text)
    dates = _extract_dates(chunk_text)
    document_codes = extract_document_codes(chunk_text)
    keywords = _build_fallback_keywords(
        entities=entities,
        organizations=organizations,
        dates=dates,
        document_codes=document_codes,
    )

    return {
        "keywords": keywords,
        "entities": entities[:15],
        "organizations": organizations[:15],
        "dates": dates[:15],
        "document_codes": document_codes[:15],
    }


class HyQResultModel(BaseModel):
    summary: str = Field(description="Tóm tắt nội dung chính ngắn gọn")
    questions: list[str] = Field(description="Danh sách câu hỏi tiếng Việt có dấu")
    entities: list[str] = Field(default_factory=list, description="Danh sách thực thể quan trọng trong đoạn")


class HyQBatchItemModel(BaseModel):
    index: int = Field(description="Index của đoạn trong batch")
    summary: str = Field(description="Tóm tắt nội dung chính ngắn gọn")
    questions: list[str] = Field(description="Danh sách câu hỏi tiếng Việt có dấu")
    entities: list[str] = Field(default_factory=list, description="Danh sách thực thể quan trọng trong đoạn")


class HyQBatchResultModel(BaseModel):
    items: list[HyQBatchItemModel] = Field(default_factory=list)


class SummaryBatchItemModel(BaseModel):
    index: int = Field(description="Index của đoạn trong batch")
    summary: str = Field(description="Tóm tắt nội dung chính ngắn gọn")


class SummaryBatchResultModel(BaseModel):
    items: list[SummaryBatchItemModel] = Field(default_factory=list)

class MetadataBundleGenerator:
    def __init__(self) -> None:
        self.summary_words = 50
        self.question_count = 3
        self.base_url = require_ollama_service_url()
        self._http_client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        atexit.register(self.close)

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is None:
            with self._client_lock:
                if self._http_client is None:
                    self._http_client = httpx.Client(timeout=180.0)
        return self._http_client

    def close(self) -> None:
        with self._client_lock:
            if self._http_client is None:
                return
            self._http_client.close()
            self._http_client = None

    def _generate_many_with_llm(
        self,
        *,
        chunk_texts: list[str],
        contexts: list[dict[str, str | None]],
    ) -> list[ChunkEnrichmentResult | None]:
        results: list[ChunkEnrichmentResult | None] = [None] * len(chunk_texts)
        headers = ollama_service_headers()

        payload = {"texts": chunk_texts}
        try:
            client = self._get_http_client()
            r = client.post(
                f"{self.base_url}/v1/indexing/batch",
                json=payload,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise RuntimeError(f"Ollama service request failed at /v1/indexing/batch: {exc}") from exc

        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("Ollama service returned an invalid response for /v1/indexing/batch.")

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f"Ollama service returned an invalid batch item at index {index}.")
            if item.get("error"):
                raise RuntimeError(f"Ollama service indexing failed for chunk {index}: {item.get('error')}")

            summary = item.get("summary") or ""
            hyq = item.get("hyq") or []
            metadata = item.get("metadata") or {}
            entities = metadata.get("keywords") or []

            parsed_hyq = _parse_hyq_payload(
                {
                    "summary": summary,
                    "questions": hyq,
                },
                summary_words=self.summary_words,
                question_count=self.question_count,
            )

            cleaned_entities = _dedupe_keep_order(
                [str(e) for e in entities if str(e).strip()],
                limit=8,
            )

            if parsed_hyq or cleaned_entities:
                results[index] = ChunkEnrichmentResult(
                    hyq=parsed_hyq,
                    entities=cleaned_entities,
                )

        return results

    def generate(
        self,
        *,
        chunk_text: str,
        context: dict[str, str | None],
        fallback_search: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], HyQResult | None]:
        _, hyq_results = self.generate_many(
            chunk_texts=[chunk_text],
            contexts=[context],
            fallback_searches=[fallback_search],
        )
        return fallback_search, hyq_results[0] if hyq_results else None

    def generate_many(
        self,
        *,
        chunk_texts: list[str],
        contexts: list[dict[str, str | None]],
        fallback_searches: list[dict[str, list[str]]],
    ) -> tuple[list[dict[str, list[str]]], list[HyQResult | None]]:
        if not chunk_texts:
            return fallback_searches, [None] * len(chunk_texts)

        llm_enrichments = self._generate_many_with_llm(
            chunk_texts=chunk_texts,
            contexts=contexts,
        )

        search_optimizations: list[dict[str, list[str]]] = []
        hyq_results: list[HyQResult | None] = [None] * len(chunk_texts)

        for idx, fallback in enumerate(fallback_searches):
            enrichment = llm_enrichments[idx]
            entities = enrichment.entities if enrichment is not None else []

            merged_entities = _dedupe_keep_order(
                [*entities, *fallback.get("entities", [])],
                limit=_SEARCH_OPT_LIMITS["entities"],
            )

            merged_keywords = _dedupe_keep_order(
                [*fallback.get("keywords", []), *entities],
                limit=_SEARCH_OPT_LIMITS["keywords"],
            )

            search_optimizations.append(
                {
                    "keywords": merged_keywords,
                    "entities": merged_entities,
                    "organizations": fallback.get("organizations", []),
                    "dates": fallback.get("dates", []),
                    "document_codes": fallback.get("document_codes", []),
                }
            )

            hyq_results[idx] = enrichment.hyq if enrichment is not None else None

        # Redundant summaries are handled by the same default model now
        return search_optimizations, hyq_results


_metadata_bundle_generator: MetadataBundleGenerator | None = None


def _get_metadata_bundle_generator() -> MetadataBundleGenerator:
    global _metadata_bundle_generator

    if _metadata_bundle_generator is None:
        _metadata_bundle_generator = MetadataBundleGenerator()

    return _metadata_bundle_generator


def build_structured_chunk_metadata(
    *,
    document_id: int,
    chunk_index: int,
    file_name: str,
    source_page: int | None,
    raw_metadata: dict[str, Any],
    chunk_text: str,
) -> dict[str, Any]:
    batch_result = build_structured_chunk_metadata_batch(
        document_id=document_id,
        file_name=file_name,
        chunks=[
            {
                "chunk_index": chunk_index,
                "source_page": source_page,
                "raw_metadata": raw_metadata,
                "chunk_text": chunk_text,
            }
        ],
    )
    return batch_result[0]


def build_structured_chunk_metadata_batch(
    *,
    document_id: int,
    file_name: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    contexts: list[dict[str, str | None]] = []
    fallback_searches: list[dict[str, list[str]]] = []
    chunk_texts: list[str] = []

    for item in chunks:
        raw_metadata = item.get("raw_metadata")
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        chunk_text = str(item.get("chunk_text") or "")
        contexts.append(_extract_context(raw_metadata))
        # Regex extraction as primary fallback
        fallback_searches.append(_fallback_search_optimization(chunk_text))
        chunk_texts.append(chunk_text)

    # LLM extraction for Summary, Questions, and Entities
    search_optimizations, llm_hyqs = _get_metadata_bundle_generator().generate_many(
        chunk_texts=chunk_texts,
        contexts=contexts,
        fallback_searches=fallback_searches,
    )

    structured_results: list[dict[str, Any]] = []
    for idx, item in enumerate(chunks):
        raw_metadata = item.get("raw_metadata")
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        raw_chunk_index = item.get("chunk_index")
        chunk_index = idx if raw_chunk_index is None else int(raw_chunk_index)
        source_page = item.get("source_page")
        # search_optimization is already combined (LLM entities + Regex fallback)
        search_optimization = search_optimizations[idx]

        hyq_result = llm_hyqs[idx]
        if hyq_result is None:
            raise RuntimeError(f"Ollama service did not return HyQ for chunk {chunk_index}.")

        structured_metadata: dict[str, Any] = {
            "chunk_id": f"doc_{document_id:02d}_chunk_{chunk_index:04d}",
            "source_info": {
                "file_name": file_name,
                "page_number": source_page,
                "doc_type": _infer_doc_type(file_name, raw_metadata),
            },
            "context": context,
            "search_optimization": search_optimization,
            "admin_tags": {
                "security_level": _infer_security_level(file_name, raw_metadata),
                "department": _infer_department(raw_metadata, context),
            },
            "hyq": {
                "summary": hyq_result.summary,
                "questions": hyq_result.questions,
            },
        }

        structured_results.append(structured_metadata)

    return structured_results


def build_hyq_children(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    hyq = metadata.get("hyq")
    if not isinstance(hyq, dict):
        hyq = {}

    summary = _normalize_spaces(str(hyq.get("summary") or ""))
    raw_questions = hyq.get("questions") if isinstance(hyq.get("questions"), list) else []
    questions = _dedupe_keep_order([str(item) for item in raw_questions], limit=12)

    children: list[tuple[str, str]] = []
    if summary:
        children.append(("summary", f"Tóm tắt: {summary}"))

    for question in questions:
        cleaned = question if question.endswith("?") else f"{question}?"
        children.append(("question", cleaned))

    if children:
        return children

    raise RuntimeError("Structured metadata is missing HyQ summary/questions from ollama_service.")

    fallback = _word_limited_text(fallback_text, max_words=120)
    if not fallback:
        fallback = "Không có nội dung khả dụng."
    return [("fallback", fallback)]


def build_keyword_blob(metadata: dict[str, Any], content: str) -> str:
    parts: list[str] = []

    source_info = metadata.get("source_info")
    if isinstance(source_info, dict):
        parts.extend(
            [
                str(source_info.get("file_name") or ""),
                str(source_info.get("doc_type") or ""),
                str(source_info.get("page_number") or ""),
            ]
        )

    context = metadata.get("context")
    if isinstance(context, dict):
        parts.extend([str(context.get("h2") or ""), str(context.get("h3") or "")])

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

    parts.append(str(content or ""))

    return _normalize_spaces(" ".join(parts))
