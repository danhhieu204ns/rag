"""Simple LLM orchestrator.

Use a single LLM classification pass to decide whether retrieval is required.
If not required (e.g. greeting/chit-chat), chat route can answer directly
without retrieval context.
"""

from __future__ import annotations

import logging
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


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class OrchestrationPlan:
    query_type: str
    output_mode: OutputMode
    strategy: QueryStrategy
    top_k: int
    vector_rrf_weight: float
    keyword_rrf_weight: float
    use_reranker: bool
    candidate_pool: int
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
        vector_rrf_weight=settings.hybrid_vector_rrf_weight,
        keyword_rrf_weight=settings.hybrid_keyword_rrf_weight,
        use_reranker=settings.reranker_enabled and requires_retrieval,
        candidate_pool=settings.reranker_candidate_pool,
        max_iterations=1,
        expand_query=False,
        signals=signals,
        requires_retrieval=requires_retrieval,
        confidence=confidence,
        llm_reason=str(raw.get("reason") or "") or None,
        orchestrator_source="llm",
    )
    return _emit_plan(query, started_at, plan)


def _emit_plan(query: str, started_at: float, plan: OrchestrationPlan) -> OrchestrationPlan:
    elapsed = round((time.perf_counter() - started_at) * 1000, 2)
    src = getattr(plan, "orchestrator_source", "rule")
    confidence = getattr(plan, "confidence", None)
    requires_retrieval = getattr(plan, "requires_retrieval", None)
    llm_reason = getattr(plan, "llm_reason", None)

    _emit_query_progress(
        "[orchestrator] src=%s type=%s mode=%s strategy=%s top_k=%d "
        "vw=%.1f kw=%.1f reranker=%s iters=%d signals=%s (%.2fms)",
        src,
        plan.query_type,
        plan.output_mode,
        plan.strategy,
        plan.top_k,
        plan.vector_rrf_weight,
        plan.keyword_rrf_weight,
        plan.use_reranker,
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
            "vector_rrf_weight": plan.vector_rrf_weight,
            "keyword_rrf_weight": plan.keyword_rrf_weight,
            "use_reranker": plan.use_reranker,
            "candidate_pool": plan.candidate_pool,
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
