from __future__ import annotations

import json
import re
import time
from typing import Any

from ...core.settings import settings
from .logging import _emit_query_progress
from .models import get_llm, _get_variant_llm
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
        "Bạn đang hỗ trợ hệ thống tìm kiếm tài liệu khoa học tự nhiên. "
        "Hãy viết lại câu hỏi sau thành phiên bản đầy đủ hơn, bổ sung các thuật ngữ chuyên môn "
        "(định luật, công thức, khái niệm vật lý/hóa học/sinh học) nếu phù hợp, "
        "để tối ưu việc truy xuất tài liệu. "
        "Giữ nguyên ý nghĩa gốc. Chỉ trả về đúng một câu đã viết lại, không giải thích.\n\n"
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
    }

    if not settings.query_rewrite_enabled:
        _emit_query_progress(
            "[query][rewrite] Disabled, skip rewrite (terms=%d)",
            term_count,
            event="query_rewrite_skip",
            details=details,
        )
        return query, details

    if settings.multi_query_enabled:
        details["decision_reason"] = "skipped_multi_query_active"
        _emit_query_progress(
            "[query][rewrite] Skip rewrite: multi-query active, variants cover query diversity (terms=%d)",
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


def _extract_json_from_llm_text(raw_text: str) -> str:
    """Strip markdown code fences and leading prose that LLMs sometimes prepend."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    return text


def _generate_query_variants(query: str, variant_count: int) -> list[str]:
    llm = _get_variant_llm()
    prompt = (
        f"Bạn đang hỗ trợ hệ thống tìm kiếm tài liệu khoa học tự nhiên (vật lý, hóa học, sinh học, địa lý, ...). "
        f"Hãy sinh {variant_count} biến thể của câu hỏi dưới đây để tìm kiếm tài liệu hiệu quả hơn. "
        "Mỗi biến thể dùng từ vựng hoặc thuật ngữ chuyên môn khác nhau nhưng vẫn đúng ý định ban đầu "
        "(ví dụ: thay tên thông thường bằng thuật ngữ khoa học, hoặc diễn đạt theo hướng định luật/công thức). "
        "Trả về JSON hợp lệ dạng: {\"variants\":[\"...\",\"...\"]}.\n\n"
        f"Câu hỏi gốc: {query}"
    )

    _emit_query_progress(
        "[query][multi] Start generating %d variants for '%s'",
        max(1, variant_count),
        _preview_text(query),
        event="multi_query_variants_start",
        details={
            "variant_count": max(1, variant_count),
            "query_preview": _preview_text(query),
        },
    )

    started_at = time.perf_counter()
    raw_text = ""
    try:
        response = llm.invoke(prompt)
        raw_text = str(response.content or "").strip()
        clean_text = _extract_json_from_llm_text(raw_text)
        parsed = json.loads(clean_text)
        raw_variants = parsed.get("variants", []) if isinstance(parsed, dict) else []
        variants = [str(item).strip() for item in raw_variants if str(item).strip()]
    except json.JSONDecodeError as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][multi] JSON parse error after %.2fms — raw preview: %s",
            elapsed_ms,
            raw_text[:120],
            event="multi_query_variants_json_error",
            details={
                "elapsed_ms": elapsed_ms,
                "query_preview": _preview_text(query),
                "raw_preview": raw_text[:200],
                "error": str(exc),
            },
        )
        return []
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        _emit_query_progress(
            "[query][multi] Variant generation failed after %.2fms: %s",
            elapsed_ms,
            str(exc),
            event="multi_query_variants_error",
            details={
                "elapsed_ms": elapsed_ms,
                "query_preview": _preview_text(query),
                "error": str(exc),
            },
        )
        return []

    deduped: list[str] = []
    seen: set[str] = {query.strip().lower()}
    for variant in variants:
        normalized = variant.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(variant)
        if len(deduped) >= max(1, variant_count):
            break

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    _emit_query_progress(
        "[query][multi] Generated %d/%d variants in %.2fms",
        len(deduped),
        max(1, variant_count),
        elapsed_ms,
        event="multi_query_variants_done",
        details={
            "elapsed_ms": elapsed_ms,
            "query_preview": _preview_text(query),
            "variant_count_requested": max(1, variant_count),
            "variant_count_generated": len(deduped),
            "variants_preview": [_preview_text(item) for item in deduped],
        },
    )
    return deduped
