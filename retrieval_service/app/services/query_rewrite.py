"""
Query rewriting service for improving retrieval quality.

Enhances queries using LLM before embedding to improve semantic search.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..core.settings import settings
from .ollama_client import get_llm_client

logger = logging.getLogger(__name__)


def _should_rewrite_query(query: str) -> tuple[bool, int, str]:
    """Check if query meets criteria for rewriting."""
    # Simple term count heuristic
    terms = len(query.split())
    min_terms = max(1, getattr(settings, "query_rewrite_min_terms", 5))
    max_terms = max(min_terms, getattr(settings, "query_rewrite_max_terms", 12))
    
    if terms < min_terms:
        return False, terms, "too_short"
    if terms >= max_terms:
        return False, terms, "already_specific"
    return True, terms, "rewrite_window"


def rewrite_query(query: str) -> str:
    """Rewrite query using LLM to enhance semantic matching."""
    llm = get_llm_client()
    if llm is None:
        return query
    
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
            logger.info("[query_rewrite] Empty result, fallback (%.2fms)", elapsed_ms)
            return query
        
        # Sanity check: rewritten should have reasonable length
        if len(rewritten) < len(query) * 0.5:
            logger.info("[query_rewrite] Rewritten too short, fallback (%.2fms)", elapsed_ms)
            return query
        
        logger.info("[query_rewrite] Success (%.2fms): '%s' => '%s'", elapsed_ms, query[:80], rewritten[:80])
        return rewritten
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning("[query_rewrite] Failed after %.2fms: %s", elapsed_ms, str(exc))
        return query


def maybe_rewrite_query(query: str) -> tuple[str, dict[str, Any]]:
    """Conditionally rewrite query based on heuristics."""
    enabled = getattr(settings, "query_rewrite_enabled", False)
    should_rewrite, term_count, reason = _should_rewrite_query(query)
    
    details: dict[str, Any] = {
        "enabled": enabled,
        "term_count": term_count,
        "decision_reason": reason,
        "rewritten": False,
    }
    
    if not enabled:
        logger.debug("[query_rewrite] Disabled, skip (terms=%d)", term_count)
        return query, details
    
    if not should_rewrite:
        logger.debug("[query_rewrite] Skip (%s, terms=%d)", reason, term_count)
        return query, details
    
    rewritten = rewrite_query(query)
    details["rewritten"] = rewritten != query
    details["effective_query"] = rewritten
    return rewritten, details
