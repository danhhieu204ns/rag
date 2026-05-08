from __future__ import annotations

import time
from typing import Any

from ...core.settings import settings
from .logging import _emit_query_progress
from .models import get_llm
from .utils import _lookup_terms, _preview_text


def _should_rewrite_query(query: str) -> tuple[bool, int, str]:
    terms = _lookup_terms(query)
    term_count = len(terms)
    min_terms = max(1, settings.query_rewrite_min_terms)
    max_terms = max(min_terms, settings.query_rewrite_max_terms)
    if term_count < min_terms:
        return False, term_count, "too_short"
    if term_count >= max_terms:
        return False, term_count, "already_specific"
    return True, term_count, "rewrite_window"


def _rewrite_query(query: str) -> str:
    llm = get_llm()
    prompt = (
        "Bạn là bộ phận viết lại truy vấn cho hệ thống RAG. "
        "Hãy viết lại câu hỏi sau thành một câu truy vấn đầy đủ, rõ nghĩa và giàu ngữ cảnh hơn "
        "để tối ưu việc truy xuất thông tin từ mọi loại tài liệu như tài liệu chuyên môn, "
        "báo cáo, quy trình, hợp đồng, văn bản hành chính, giáo trình, tài liệu kỹ thuật hoặc tài liệu nội bộ. "
        "Bổ sung thuật ngữ chuyên ngành, đối tượng, phạm vi, tiêu chí, mốc thời gian hoặc bối cảnh liên quan nếu phù hợp. "
        "Giữ nguyên ý nghĩa gốc, không bịa thêm thông tin. "
        "Chỉ trả về đúng một câu đã viết lại, không giải thích.\n\n"
        f"Câu hỏi gốc: {query}\n"
        "Câu hỏi đã viết lại:"
    )

    started_at = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        rewritten = str(response.content or "").strip()
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        if not rewritten:
            _emit_query_progress(
                "[query][rewrite] Empty rewrite result, fallback to original query (%.2fms)",
                elapsed_ms,
                event="query_rewrite_empty",
                details={
                    "elapsed_ms": elapsed_ms,
                    "original_query_preview": _preview_text(query),
                },
            )
            return query
        rewritten_terms = len(_lookup_terms(rewritten))
        original_terms = len(_lookup_terms(query))
        if rewritten_terms < original_terms:
            _emit_query_progress(
                "[query][rewrite] Rewritten query has fewer terms (%d < %d), keep original (%.2fms)",
                rewritten_terms,
                original_terms,
                elapsed_ms,
                event="query_rewrite_rejected",
                details={
                    "elapsed_ms": elapsed_ms,
                    "original_term_count": original_terms,
                    "rewritten_term_count": rewritten_terms,
                    "original_query_preview": _preview_text(query),
                    "rewritten_query_preview": _preview_text(rewritten),
                },
            )
            return query
        _emit_query_progress(
            "[query][rewrite] Rewrite success in %.2fms: '%s' => '%s'",
            elapsed_ms,
            _preview_text(query),
            _preview_text(rewritten),
            event="query_rewrite_success",
            details={
                "elapsed_ms": elapsed_ms,
                "original_term_count": original_terms,
                "rewritten_term_count": rewritten_terms,
                "original_query_preview": _preview_text(query),
                "rewritten_query_preview": _preview_text(rewritten),
            },
        )
        return rewritten
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][rewrite] Rewrite failed after %.2fms: %s",
            elapsed_ms,
            str(exc),
            event="query_rewrite_error",
            details={
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
                "original_query_preview": _preview_text(query),
            },
        )
        return query


def _maybe_rewrite_query(query: str) -> tuple[str, dict[str, Any]]:
    should_rewrite, term_count, reason = _should_rewrite_query(query)
    details: dict[str, Any] = {
        "enabled": settings.query_rewrite_enabled,
        "term_count": term_count,
        "min_terms": settings.query_rewrite_min_terms,
        "max_terms": settings.query_rewrite_max_terms,
        "decision_reason": reason,
        "rewritten": False,
        "original_query_preview": _preview_text(query),
    }

    if not settings.query_rewrite_enabled:
        _emit_query_progress(
            "[query][rewrite] Disabled, skip rewrite (terms=%d)",
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    if not should_rewrite:
        _emit_query_progress(
            "[query][rewrite] Skip rewrite: reason=%s, terms=%d",
            reason,
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    rewritten = _rewrite_query(query)
    details["rewritten"] = rewritten != query
    details["effective_query_preview"] = _preview_text(rewritten)
    _emit_query_progress(
        "[query][rewrite] Rewrite decision done: rewritten=%s",
        details["rewritten"],
        event="query_rewrite_done",
        details=details,
    )
    return rewritten, details


