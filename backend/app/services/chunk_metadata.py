from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama

from ..core.settings import settings


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


def _extract_json_payload(raw_output: str) -> dict[str, Any] | None:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        payload = json.loads(raw_output[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    return payload


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


def _merge_search_optimization(
    primary: dict[str, list[str]],
    fallback: dict[str, list[str]],
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}

    for key, limit in _SEARCH_OPT_LIMITS.items():
        merged[key] = _dedupe_keep_order(
            [*primary.get(key, []), *fallback.get(key, [])],
            limit=limit,
        )

    return merged


def _parse_search_optimization_payload(payload: dict[str, Any]) -> dict[str, list[str]]:
    entities = _payload_list(payload, "entities", limit=_SEARCH_OPT_LIMITS["entities"])
    organizations = _payload_list(payload, "organizations", limit=_SEARCH_OPT_LIMITS["organizations"])
    dates = _payload_list(payload, "dates", limit=_SEARCH_OPT_LIMITS["dates"])

    raw_codes = _payload_list(payload, "document_codes", limit=30)
    parsed_codes: list[str] = []
    for item in raw_codes:
        parsed_codes.extend(extract_document_codes(item, limit=5))
    document_codes = _dedupe_keep_order(parsed_codes, limit=_SEARCH_OPT_LIMITS["document_codes"])

    keywords = _payload_list(payload, "keywords", limit=_SEARCH_OPT_LIMITS["keywords"])
    if not keywords:
        keywords = _build_fallback_keywords(
            entities=entities,
            organizations=organizations,
            dates=dates,
            document_codes=document_codes,
        )

    return {
        "keywords": keywords,
        "entities": entities,
        "organizations": organizations,
        "dates": dates,
        "document_codes": document_codes,
    }


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


class MetadataBundleGenerator:
    def __init__(self) -> None:
        self.summary_words = max(20, settings.hyq_summary_words)
        self.question_count = max(1, settings.hyq_questions_per_chunk)
        self.use_llm = bool(settings.metadata_use_llm or settings.hyq_use_llm)
        self.model_name = settings.metadata_model or settings.hyq_model or settings.llm_model
        self.base_url = settings.ollama_base_url
        self.num_thread = max(1, settings.metadata_ollama_num_thread)
        self.num_predict = max(64, settings.metadata_ollama_num_predict)
        self._thread_local = threading.local()
        self._llm_disabled = not self.use_llm

    def _get_llm(self) -> ChatOllama | None:
        if self._llm_disabled:
            return None

        llm = getattr(self._thread_local, "llm", None)
        if llm is None:
            if not self.model_name:
                self._llm_disabled = True
                return None

            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
                num_thread=self.num_thread,
                num_predict=self.num_predict,
            )
            self._thread_local.llm = llm

        return llm

    def _parse_llm_json(self, raw_output: str) -> tuple[dict[str, list[str]] | None, HyQResult | None]:
        payload = _extract_json_payload(raw_output)
        if payload is None:
            return None, None

        search_payload: dict[str, Any] | None = None
        hyq_payload: dict[str, Any] | None = None

        nested_search = payload.get("search_optimization")
        if isinstance(nested_search, dict):
            search_payload = nested_search
        elif any(key in payload for key in ("keywords", "entities", "organizations", "dates", "document_codes")):
            search_payload = payload

        nested_hyq = payload.get("hyq")
        if isinstance(nested_hyq, dict):
            hyq_payload = nested_hyq
        elif any(key in payload for key in ("summary", "questions")):
            hyq_payload = payload

        parsed_search = (
            _parse_search_optimization_payload(search_payload)
            if isinstance(search_payload, dict)
            else None
        )
        parsed_hyq = (
            _parse_hyq_payload(
                hyq_payload,
                summary_words=self.summary_words,
                question_count=self.question_count,
            )
            if isinstance(hyq_payload, dict)
            else None
        )

        return parsed_search, parsed_hyq

    def _generate_with_llm(
        self,
        *,
        chunk_text: str,
        context: dict[str, str | None],
        fallback_search: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, HyQResult | None]:
        llm = self._get_llm()
        if llm is None:
            return None, None

        try:
            prompt = (
                "Bạn đang chuẩn bị metadata cho hybrid RAG. "
                "Chỉ trả về DUY NHẤT một đối tượng JSON có 2 trường: "
                "search_optimization (object) và hyq (object). "
                "search_optimization phải gồm: keywords, entities, organizations, dates, document_codes (đều là array string). "
                f"hyq.summary phải <= {self.summary_words} từ. "
                f"hyq.questions phải chứa chính xác {self.question_count} câu hỏi tiếng Việt có dấu. "
                "Không dùng markdown fences.\n\n"
                f"h2: {context.get('h2') or ''}\n"
                f"h3: {context.get('h3') or ''}\n"
                f"seed_keywords: {', '.join(fallback_search.get('keywords') or [])}\n"
                f"seed_document_codes: {', '.join(fallback_search.get('document_codes') or [])}\n"
                f"seed_organizations: {', '.join(fallback_search.get('organizations') or [])}\n"
                f"seed_entities: {', '.join(fallback_search.get('entities') or [])}\n"
                f"seed_dates: {', '.join(fallback_search.get('dates') or [])}\n\n"
                f"chunk_text:\n{chunk_text}"
            )
            response = llm.invoke(prompt)
        except Exception:
            self._llm_disabled = True
            return None, None

        raw_output = str(getattr(response, "content", response) or "")
        return self._parse_llm_json(raw_output)

    def generate(
        self,
        *,
        chunk_text: str,
        context: dict[str, str | None],
        fallback_search: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]], HyQResult | None]:
        if not self.use_llm:
            return fallback_search, None

        llm_search, llm_hyq = self._generate_with_llm(
            chunk_text=chunk_text,
            context=context,
            fallback_search=fallback_search,
        )

        if llm_search is None:
            return fallback_search, llm_hyq

        merged_search = _merge_search_optimization(llm_search, fallback_search)
        return merged_search, llm_hyq


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
    context = _extract_context(raw_metadata)
    fallback_search = _fallback_search_optimization(chunk_text)
    search_optimization, llm_hyq = _get_metadata_bundle_generator().generate(
        chunk_text=chunk_text,
        context=context,
        fallback_search=fallback_search,
    )

    hyq_result = HyQResult(summary="", questions=[])
    if settings.hyq_enabled:
        if llm_hyq is not None:
            hyq_result = llm_hyq
        else:
            hyq_result = _fallback_hyq(
                chunk_text=chunk_text,
                context=context,
                search_optimization=search_optimization,
                summary_words=settings.hyq_summary_words,
                question_count=settings.hyq_questions_per_chunk,
            )

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

    return structured_metadata


def build_hyq_children(metadata: dict[str, Any], fallback_text: str) -> list[tuple[str, str]]:
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
