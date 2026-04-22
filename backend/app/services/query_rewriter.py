from __future__ import annotations

import logging
import re

from langchain_ollama import ChatOllama

from ..core.settings import settings

logger = logging.getLogger(__name__)

# Patterns that identify precise identifiers which must survive rewriting unchanged.
# Order matters: more specific patterns first.
_PROTECTED_PATTERNS: list[str] = [
    r"\d{1,3}/[A-ZĐÂĂÊÔƠƯÀÁẢÃẠ][A-ZĐÂĂÊÔƠƯÀÁẢÃẠ\-]+",  # 123/QĐ-HĐQT
    r"\d{1,2}/\d{1,2}/\d{4}",                              # dd/mm/yyyy
    r"\b\d{4}\b",                                           # year (4 digits)
]

_REWRITE_PROMPT = (
    "Viết lại câu hỏi sau thành phiên bản đầy đủ hơn để tìm kiếm ngữ nghĩa.\n"
    "Quy tắc bắt buộc:\n"
    "- Giữ NGUYÊN VẸN các định danh sau (không đổi, không dịch): {protected}\n"
    "- Không thêm số liệu, ngày tháng, hoặc tên tổ chức mới nếu chúng không có trong câu hỏi gốc\n"
    "- Chỉ làm rõ ý định câu hỏi, không sáng tạo thêm nội dung\n"
    "- Trả về đúng 1 câu, không kèm giải thích\n\n"
    "Câu hỏi gốc: {query}\n"
    "Phiên bản đầy đủ:"
)

_rewrite_llm: ChatOllama | None = None


def _get_rewrite_llm() -> ChatOllama:
    global _rewrite_llm
    if _rewrite_llm is None:
        model = settings.query_rewrite_model or settings.llm_model
        _rewrite_llm = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_thread=settings.ollama_num_thread,
        )
    return _rewrite_llm


def extract_protected_entities(query: str) -> list[str]:
    """Return identifiers in query that must not be altered during rewriting."""
    seen: dict[str, None] = {}
    for pattern in _PROTECTED_PATTERNS:
        for match in re.finditer(pattern, query, flags=re.IGNORECASE):
            seen[match.group()] = None
    return list(seen)


def _should_rewrite(query: str, protected: list[str]) -> bool:
    """Return True when the query is short/vague enough to benefit from expansion."""
    word_count = len(query.split())
    # Long query with specific identifiers → already precise enough
    if word_count >= settings.query_rewrite_min_words and protected:
        return False
    # Very long query → already has enough context
    if word_count >= 12:
        return False
    return True


def rewrite_for_vector(query: str) -> str:
    """
    Return an expanded version of query suitable for the vector search branch.

    The keyword search branch must always receive the original query — this
    function is only called for the vector branch.  Returns the original query
    unchanged when rewriting is disabled, not needed, or the LLM fails.
    """
    if not settings.query_rewrite_enabled:
        return query

    protected = extract_protected_entities(query)

    if not _should_rewrite(query, protected):
        logger.debug(
            "[rewrite] skip: %d words, protected=%s",
            len(query.split()),
            protected,
        )
        return query

    try:
        llm = _get_rewrite_llm()
        protected_str = ", ".join(protected) if protected else "không có"
        prompt = _REWRITE_PROMPT.format(protected=protected_str, query=query)
        result = llm.invoke(prompt)
        expanded = result.content.strip()

        # Safety: if the LLM dropped any protected entity, fall back.
        for entity in protected:
            if entity not in expanded:
                logger.warning(
                    "[rewrite] entity '%s' lost — falling back to original query",
                    entity,
                )
                return query

        if not expanded or expanded == query:
            return query

        logger.info("[rewrite] '%s'  →  '%s'", query, expanded)
        return expanded

    except Exception as exc:
        logger.warning("[rewrite] LLM call failed, using original query: %s", exc)
        return query
