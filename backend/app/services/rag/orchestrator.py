"""Simple LLM orchestrator.

Use a single LLM classification pass to decide whether retrieval is required.
If not required (e.g. greeting/chit-chat), chat route can answer directly
without retrieval context.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from ...core.settings import settings
from ..ollama_service_client import ollama_service_headers, require_ollama_service_url
from .logging import _emit_query_progress
from .utils import _preview_text

QueryStrategy = Literal["keyword_heavy", "vector_heavy", "balanced", "broad"]
OutputMode = Literal["qa", "outline", "script", "quiz", "summary_doc"]

logger = logging.getLogger(__name__)

_RETRIEVAL_INTENT_PHRASES = (
    "phân biệt",
    "so sánh",
    "giải thích",
    "trình bày",
    "liệt kê",
    "tóm tắt",
    "định nghĩa",
    "là gì",
    "như thế nào",
    "hoạt động",
    "vì sao",
    "tại sao",
    "quy định",
    "chính sách",
    "hướng dẫn",
    "điều kiện",
    "yêu cầu",
    "bao nhiêu",
    "compare",
    "explain",
    "summarize",
    "define",
    "what is",
    "how does",
)
_ASSISTANT_IDENTITY_PHRASES = (
    "bạn là ai",
    "bạn là gì",
    "tên bạn",
    "who are you",
    "what are you",
)
_LOW_SIGNAL_TERMS = {
    "haha",
    "hihi",
    "xin",
    "chào",
    "hello",
    "hi",
    "cảm",
    "ơn",
    "thanks",
    "thank",
    "you",
    "hãy",
    "vui",
    "lòng",
    "giúp",
    "tôi",
    "mình",
    "ngắn",
    "gọn",
    "trả",
    "lời",
    "thật",
    "trong",
    "từ",
}


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class OrchestrationPlan:
    query_type: str
    output_mode: OutputMode
    strategy: QueryStrategy
    top_k: int
    max_iterations: int
    expand_query: bool
    signals: list[str] = field(default_factory=list)
    requires_retrieval: bool = True
    confidence: float = 1.0
    llm_reason: str | None = None
    orchestrator_source: str = "llm"


# ─── Public API ───────────────────────────────────────────────────────────────

def _call_orchestrator_service(query: str) -> dict[str, Any] | None:
    try:
        service_url = require_ollama_service_url()
        headers = ollama_service_headers()
    except RuntimeError:
        return None

    timeout = httpx.Timeout(settings.orchestrator_timeout_seconds, connect=5.0)
    payload = {"query": query}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.post(f"{service_url}/v1/orchestrator/classify", json=payload)
            resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except httpx.HTTPError:
        logger.exception("[orchestrator] service classify call failed")
        return None
    except ValueError as exc:
        logger.warning("[orchestrator] service classify response was not JSON: %s", exc)
        return None


def _looks_like_retrieval_query(query: str) -> bool:
    text = " ".join(str(query or "").casefold().split())
    if not text or any(phrase in text for phrase in _ASSISTANT_IDENTITY_PHRASES):
        return False
    if not any(phrase in text for phrase in _RETRIEVAL_INTENT_PHRASES):
        return False

    terms = re.findall(r"[\wÀ-ỹĐđ]+", text, flags=re.UNICODE)
    signal_terms = [
        term
        for term in terms
        if len(term) >= 2 and term not in _LOW_SIGNAL_TERMS and not term.isdigit()
    ]
    return bool(signal_terms)


def classify_query(
    query: str,
    base_top_k: int,
    output_mode_override: OutputMode | None = None,
) -> OrchestrationPlan:
    started_at = time.perf_counter()
    raw = _call_orchestrator_service(query) or {}
    requires_retrieval = bool(raw.get("requires_retrieval", True))
    query_type = str(raw.get("query_type") or ("needs_retrieval" if requires_retrieval else "chitchat"))
    signals = [str(s) for s in (raw.get("signals") or []) if s]
    retrieval_guard_applied = False
    if not requires_retrieval and _looks_like_retrieval_query(query):
        requires_retrieval = True
        query_type = "needs_retrieval"
        retrieval_guard_applied = True
        signals.append("retrieval_guard")
    confidence_raw = raw.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0

    plan = OrchestrationPlan(
        query_type=query_type,
        output_mode=output_mode_override or "qa",
        strategy="balanced",
        top_k=base_top_k,
        max_iterations=1,
        expand_query=False,
        signals=signals,
        requires_retrieval=requires_retrieval,
        confidence=confidence,
        llm_reason=_orchestrator_reason(raw, retrieval_guard_applied),
        orchestrator_source="llm",
    )
    return _emit_plan(query, started_at, plan)


def _orchestrator_reason(raw: dict[str, Any], retrieval_guard_applied: bool) -> str | None:
    reason = str(raw.get("reason") or "").strip()
    if retrieval_guard_applied:
        guard_reason = "retrieval_guard: query has factual/explanatory intent"
        return f"{reason} | {guard_reason}" if reason else guard_reason
    return reason or None


def _emit_plan(query: str, started_at: float, plan: OrchestrationPlan) -> OrchestrationPlan:
    elapsed = round((time.perf_counter() - started_at) * 1000, 2)
    src = getattr(plan, "orchestrator_source", "rule")
    confidence = getattr(plan, "confidence", None)
    requires_retrieval = getattr(plan, "requires_retrieval", None)
    llm_reason = getattr(plan, "llm_reason", None)

    _emit_query_progress(
        "[orchestrator] src=%s type=%s mode=%s strategy=%s top_k=%d "
        "retrieval_defaults=retrieval_service iters=%d signals=%s (%.2fms)",
        src,
        plan.query_type,
        plan.output_mode,
        plan.strategy,
        plan.top_k,
        plan.max_iterations,
        plan.signals,
        elapsed,
        event="orchestrator_plan",
        details={
            "query_preview": _preview_text(query),
            "query_type": plan.query_type,
            "output_mode": plan.output_mode,
            "strategy": plan.strategy,
            "top_k": plan.top_k,
            "retrieval_defaults_source": "retrieval_service",
            "max_iterations": plan.max_iterations,
            "expand_query": plan.expand_query,
            "signals": plan.signals,
            "orchestrator_source": src,
            "confidence": confidence,
            "requires_retrieval": requires_retrieval,
            "llm_reason": (llm_reason[:400] if isinstance(llm_reason, str) else None),
            "elapsed_ms": elapsed,
        },
    )
    return plan
