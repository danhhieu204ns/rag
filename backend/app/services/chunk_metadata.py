from __future__ import annotations

import json
import re
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


class HyQGenerator:
    def __init__(self) -> None:
        self.summary_words = max(20, settings.hyq_summary_words)
        self.question_count = max(1, settings.hyq_questions_per_chunk)
        self.use_llm = bool(settings.hyq_use_llm)
        self.model_name = settings.hyq_model or settings.llm_model
        self.base_url = settings.ollama_base_url
        self._llm: ChatOllama | None = None
        self._llm_disabled = not self.use_llm

    def _get_llm(self) -> ChatOllama | None:
        if self._llm_disabled:
            return None

        if self._llm is None:
            if not self.model_name:
                self._llm_disabled = True
                return None
            self._llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
            )

        return self._llm

    def _parse_llm_json(self, raw_output: str) -> HyQResult | None:
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

        summary = _word_limited_text(str(payload.get("summary") or ""), self.summary_words)
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            return None

        cleaned_questions = [str(item) for item in raw_questions if str(item).strip()]
        deduped = _dedupe_keep_order(cleaned_questions, limit=self.question_count)
        if not summary or not deduped:
            return None

        while len(deduped) < self.question_count:
            deduped.append(f"Thông tin chính của đoạn này là gì? ({len(deduped) + 1})")

        normalized_questions = [question if question.endswith("?") else f"{question}?" for question in deduped]
        return HyQResult(summary=summary, questions=normalized_questions)

    def _generate_with_llm(
        self,
        *,
        chunk_text: str,
        context: dict[str, str | None],
        search_optimization: dict[str, list[str]],
    ) -> HyQResult | None:
        llm = self._get_llm()
        if llm is None:
            return None

        try:
            prompt = (
                "Bạn đang chuẩn bị child chunks cho truy xuất hybrid RAG. "
                "Chỉ trả về DUY NHẤT một đối tượng JSON với các trường: summary (string) và questions (array). "
                f"summary phải <= {self.summary_words} từ. "
                f"questions phải chứa chính xác {self.question_count} câu hỏi tiếng Việt có dấu. "
                "Không dùng markdown fences.\n\n"
                f"h2: {context.get('h2') or ''}\n"
                f"h3: {context.get('h3') or ''}\n"
                f"document_codes: {', '.join(search_optimization.get('document_codes') or [])}\n"
                f"organizations: {', '.join(search_optimization.get('organizations') or [])}\n"
                f"entities: {', '.join(search_optimization.get('entities') or [])}\n"
                f"dates: {', '.join(search_optimization.get('dates') or [])}\n\n"
                f"chunk_text:\n{chunk_text}"
            )
            response = llm.invoke(prompt)
        except Exception:
            self._llm_disabled = True
            return None

        raw_output = str(getattr(response, "content", response) or "")
        parsed = self._parse_llm_json(raw_output)
        if parsed is None:
            return None
        return parsed

    def generate(
        self,
        *,
        chunk_text: str,
        context: dict[str, str | None],
        search_optimization: dict[str, list[str]],
    ) -> HyQResult:
        if self.use_llm:
            llm_result = self._generate_with_llm(
                chunk_text=chunk_text,
                context=context,
                search_optimization=search_optimization,
            )
            if llm_result is not None:
                return llm_result

        return _fallback_hyq(
            chunk_text=chunk_text,
            context=context,
            search_optimization=search_optimization,
            summary_words=self.summary_words,
            question_count=self.question_count,
        )


_hyq_generator: HyQGenerator | None = None


def _get_hyq_generator() -> HyQGenerator:
    global _hyq_generator

    if _hyq_generator is None:
        _hyq_generator = HyQGenerator()

    return _hyq_generator


def _extract_search_optimization(chunk_text: str) -> dict[str, list[str]]:
    entities, organizations = _split_named_phrases(chunk_text)
    dates = _extract_dates(chunk_text)
    document_codes = extract_document_codes(chunk_text)

    return {
        "entities": entities[:15],
        "organizations": organizations[:15],
        "dates": dates[:15],
        "document_codes": document_codes[:15],
    }


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
    search_optimization = _extract_search_optimization(chunk_text)

    hyq_result = HyQResult(summary="", questions=[])
    if settings.hyq_enabled:
        hyq_result = _get_hyq_generator().generate(
            chunk_text=chunk_text,
            context=context,
            search_optimization=search_optimization,
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
        for key in ("entities", "organizations", "dates", "document_codes"):
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
