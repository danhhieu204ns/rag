"""
Query Orchestrator — rule-based, zero-latency retrieval + output planner.

For each incoming query the orchestrator produces an OrchestrationPlan that
drives two orthogonal concerns:

  output_mode — what kind of structured content to generate
                ("qa" | "outline" | "script" | "quiz" | "summary_doc")

  query_type  — how to retrieve (which search arm to emphasise, how many
                candidates to gather, whether to retry)

Detection order inside classify_query():
  1. Output mode is resolved first (highest priority).
     Structured modes (outline / script / quiz / summary_doc) carry their
     own retrieval config so query-type classification is skipped for them.
  2. For plain Q&A mode the existing 9-bucket query-type logic runs.

No LLM calls are made — pure regex + heuristics, overhead < 1 ms.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from ...core.settings import settings
from .logging import _emit_query_progress
from .utils import _normalize_lookup_text, _lookup_terms, _preview_text

QueryStrategy = Literal["keyword_heavy", "vector_heavy", "balanced", "broad"]
OutputMode = Literal["qa", "outline", "script", "quiz", "summary_doc"]


# ─── Document-code patterns ────────────────────────────────────────────────────
_DOC_CODE_RE = re.compile(
    r"\b\d{1,6}[/-][A-ZĐĂÂÊÔƯ]{2,}(?:[/-][A-ZĐĂÂÊÔƯ]{2,})*\b"
    r"|\b[A-ZĐĂÂÊÔƯ]{2,}[/-]\d{1,6}(?:[/-][A-ZĐĂÂÊÔƯ]{2,})*\b",
    re.UNICODE,
)
_DOC_CODE_NORM_RE = re.compile(
    r"\b\d{1,6}[/-][A-Za-z]{2,}(?:[/-][A-Za-z]{2,})*\b"
    r"|\b[A-Za-z]{2,}[/-]\d{1,6}(?:[/-][A-Za-z]{2,})*\b"
    r"|\b(?:so|so\s+van\s+ban|cong\s+van|quyet\s+dinh|nghi\s+dinh"
    r"|thong\s+tu|chi\s+thi|nghi\s+quyet|van\s+ban)\s+so\s+\d{1,6}\b",
)

# ─── Output-mode detection (normalized text) ──────────────────────────────────
_OUTPUT_MODE_RE: dict[str, re.Pattern[str]] = {
    "outline": re.compile(
        r"\b(tao\s+de\s+cuong|lap\s+de\s+cuong|viet\s+de\s+cuong"
        r"|outline|dan\s+y|xay\s+dung\s+de\s+cuong|tao\s+outline"
        r"|de\s+cuong\s+bai|lap\s+dan\s+y|tao\s+dan\s+y)\b"
    ),
    "script": re.compile(
        r"\b(tao\s+script|viet\s+script|kich\s+ban\s+giang"
        r"|soan\s+bai\s+giang|viet\s+kich\s+ban|kich\s+ban\s+bai\s+giang"
        r"|soan\s+giao\s+an|viet\s+bai\s+giang|tao\s+bai\s+giang)\b"
    ),
    "quiz": re.compile(
        r"\b(tao\s+cau\s+hoi|tao\s+bai\s+tap|de\s+kiem\s+tra"
        r"|cau\s+hoi\s+on\s+tap|tao\s+de\s+thi|bo\s+cau\s+hoi"
        r"|bai\s+tap\s+thuc\s+hanh|cau\s+hoi\s+trac\s+nghiem"
        r"|tao\s+bai\s+kiem\s+tra)\b"
    ),
    "summary_doc": re.compile(
        r"\b(tom\s+tat\s+toan\s+bo|tom\s+tat\s+chuong|tom\s+tat\s+tai\s+lieu"
        r"|tong\s+hop\s+noi\s+dung|tom\s+tat\s+va\s+phan\s+tich"
        r"|tong\s+quan\s+tai\s+lieu|tom\s+luoc)\b"
    ),
}

# Retrieval config for each structured mode (overrides query-type classification)
_STRUCTURED_MODE_CONFIG: dict[str, dict] = {
    "outline":     {"top_k_mult": 3.0, "strategy": "vector_heavy", "vw": 1.6, "kw": 0.7},
    "script":      {"top_k_mult": 3.0, "strategy": "vector_heavy", "vw": 1.7, "kw": 0.6},
    "quiz":        {"top_k_mult": 2.5, "strategy": "balanced",     "vw": 1.2, "kw": 1.1},
    "summary_doc": {"top_k_mult": 4.0, "strategy": "vector_heavy", "vw": 1.8, "kw": 0.5},
}

# ─── Q&A query-type intent signals ────────────────────────────────────────────
_DEFINITION_RE = re.compile(
    r"\b(la\s+gi|dinh\s+nghia|khai\s+niem|nghia\s+la|co\s+nghia"
    r"|hieu\s+nhu\s+the\s+nao|tuc\s+la|duoc\s+hieu\s+la|duoc\s+goi\s+la"
    r"|bao\s+gom\s+nhung\s+gi|the\s+nao\s+la)\b",
)
_FACTUAL_RE = re.compile(
    r"\b(bao\s+nhieu|muc\s+toi\s+da|muc\s+toi\s+thieu|so\s+luong"
    r"|ty\s+le|phan\s+tram|thoi\s+han|thoi\s+gian|quy\s+dinh\s+muc"
    r"|dieu\s+kien\s+la\s+gi|tieu\s+chuan|tieu\s+chi|nam\s+nao"
    r"|ngay\s+nao|bao\s+lau|gia\s+tri|han\s+muc|cap\s+do|muc\s+phat"
    r"|muc\s+luong|quy\s+mo|tong\s+so|so\s+nam|bao\s+nhieu\s+nam)\b",
)
_PROCEDURAL_RE = re.compile(
    r"\b(lam\s+the\s+nao|quy\s+trinh|thu\s+tuc|cac\s+buoc|huong\s+dan"
    r"|can\s+lam\s+gi|cach\s+thuc|phuong\s+thuc|lam\s+sao|de\s+duoc"
    r"|dang\s+ky|xin\s+cap|trinh\s+tu|ky\s+ket|ky\s+hop\s+dong"
    r"|nop\s+ho\s+so|gui\s+don|yeu\s+cau|de\s+nghi|xin\s+phep"
    r"|thu\s+tuc\s+nhu\s+the\s+nao|can\s+nhung\s+gi)\b",
)
_COMPARATIVE_RE = re.compile(
    r"\b(so\s+sanh|khac\s+nhau|giong\s+nhau|su\s+khac\s+biet"
    r"|phan\s+biet|diem\s+giong|diem\s+khac|khac\s+gi|giong\s+gi"
    r"|uu\s+nhuoc\s+diem|uu\s+diem|nhuoc\s+diem|giua\s+\w+\s+va)\b",
)
_LISTING_RE = re.compile(
    r"\b(liet\s+ke|nhung\s+loai|cac\s+loai|cac\s+truong\s+hop|tat\s+ca"
    r"|danh\s+sach|co\s+nhung|co\s+may\s+loai|co\s+bao\s+nhieu\s+loai"
    r"|nhung\s+gi|bao\s+gom\s+nhung|cac\s+thanh\s+phan|cac\s+yeu\s+to"
    r"|cac\s+dieu\s+kien|nhung\s+truong\s+hop|cac\s+hinh\s+thuc)\b",
)
_SUMMARY_RE = re.compile(
    r"\b(tom\s+tat|tong\s+hop|toan\s+bo|noi\s+dung\s+chinh|y\s+chinh"
    r"|mo\s+ta|gioi\s+thieu|noi\s+dung\s+van\s+ban|quy\s+dinh\s+ve"
    r"|cac\s+noi\s+dung|noi\s+dung\s+co\s+ban|chu\s+yeu\s+la)\b",
)
_FOLLOWUP_RE = re.compile(
    r"\b(no\b|dieu\s+do|van\s+de\s+nay|dieu\s+nay|truong\s+hop\s+nay"
    r"|nhu\s+vay|dieu\s+tren|y\s+kien\s+do|quy\s+dinh\s+tren|nhu\s+da\s+neu)\b",
)


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class OrchestrationPlan:
    """
    Adaptive plan produced by classify_query().

    output_mode controls *what* to generate; the other fields control *how*
    to retrieve. Both concerns are resolved in one zero-latency pass.
    """

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

    def with_broader_search(self) -> "OrchestrationPlan":
        """Return a relaxed copy used on retry iterations."""
        return OrchestrationPlan(
            query_type=self.query_type,
            output_mode=self.output_mode,
            strategy="broad",
            top_k=self.top_k,
            vector_rrf_weight=max(self.vector_rrf_weight, 1.0),
            keyword_rrf_weight=max(self.keyword_rrf_weight, 1.0),
            use_reranker=self.use_reranker,
            candidate_pool=min(self.candidate_pool + 10, 40),
            max_iterations=1,
            expand_query=True,
            signals=[*self.signals, "retry_broader"],
        )


# ─── Public API ───────────────────────────────────────────────────────────────

def classify_query(
    query: str,
    base_top_k: int,
    output_mode_override: OutputMode | None = None,
) -> OrchestrationPlan:
    """
    Classify query and return an OrchestrationPlan.

    output_mode_override — when the caller (e.g. frontend) explicitly selects a
    mode, pass it here to skip auto-detection while still running the normal
    retrieval-parameter logic.
    """
    started_at = time.perf_counter()
    normalized = _normalize_lookup_text(query)
    terms = _lookup_terms(query)
    term_count = len(terms)
    signals: list[str] = []

    reranker_on = settings.reranker_enabled
    base_pool = settings.reranker_candidate_pool

    # ── Step 1: resolve output mode ───────────────────────────────────────────
    output_mode: OutputMode = output_mode_override or "qa"
    if output_mode == "qa":
        for mode, pattern in _OUTPUT_MODE_RE.items():
            if pattern.search(normalized):
                output_mode = mode  # type: ignore[assignment]
                signals.append(f"output_mode:{mode}")
                break

    # ── Step 2a: structured modes — fixed retrieval config ────────────────────
    if output_mode != "qa":
        cfg = _STRUCTURED_MODE_CONFIG[output_mode]
        top_k = max(base_top_k, int(base_top_k * cfg["top_k_mult"]))
        return _emit_plan(
            query, started_at,
            OrchestrationPlan(
                query_type=f"structured_{output_mode}",
                output_mode=output_mode,
                strategy=cfg["strategy"],  # type: ignore[arg-type]
                top_k=top_k,
                vector_rrf_weight=float(cfg["vw"]),
                keyword_rrf_weight=float(cfg["kw"]),
                use_reranker=reranker_on,
                candidate_pool=min(base_pool + 10, 30),
                max_iterations=1,
                expand_query=False,
                signals=signals,
            ),
        )

    # ── Step 2b: Q&A mode — nuanced query-type classification ─────────────────

    # 1. Document lookup
    if _DOC_CODE_RE.search(query) or _DOC_CODE_NORM_RE.search(normalized):
        signals.append("doc_code_detected")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="document_lookup", output_mode="qa",
            strategy="keyword_heavy", top_k=base_top_k,
            vector_rrf_weight=0.3, keyword_rrf_weight=2.5,
            use_reranker=False, candidate_pool=base_pool,
            max_iterations=2, expand_query=False, signals=signals,
        ))

    # 2. Short / follow-up
    if term_count <= 2 or _FOLLOWUP_RE.search(normalized):
        signals.append("short_or_followup")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="contextual_followup", output_mode="qa",
            strategy="vector_heavy", top_k=base_top_k,
            vector_rrf_weight=2.2, keyword_rrf_weight=0.2,
            use_reranker=False, candidate_pool=base_pool,
            max_iterations=1, expand_query=False, signals=signals,
        ))

    # 3. Definition
    if _DEFINITION_RE.search(normalized):
        signals.append("definition_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="definition_lookup", output_mode="qa",
            strategy="vector_heavy", top_k=base_top_k,
            vector_rrf_weight=1.9, keyword_rrf_weight=0.5,
            use_reranker=reranker_on, candidate_pool=min(base_pool, 15),
            max_iterations=1, expand_query=False, signals=signals,
        ))

    # 4. Comparative
    if _COMPARATIVE_RE.search(normalized):
        signals.append("comparative_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="comparative", output_mode="qa",
            strategy="broad", top_k=min(base_top_k * 2, 10),
            vector_rrf_weight=1.2, keyword_rrf_weight=1.0,
            use_reranker=reranker_on, candidate_pool=min(base_pool + 10, 30),
            max_iterations=2, expand_query=True, signals=signals,
        ))

    # 5. Procedural
    if _PROCEDURAL_RE.search(normalized):
        signals.append("procedural_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="procedural", output_mode="qa",
            strategy="vector_heavy", top_k=min(base_top_k + 2, 8),
            vector_rrf_weight=1.5, keyword_rrf_weight=0.8,
            use_reranker=reranker_on, candidate_pool=min(base_pool + 5, 25),
            max_iterations=1, expand_query=term_count < 5, signals=signals,
        ))

    # 6. Listing
    if _LISTING_RE.search(normalized):
        signals.append("listing_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="listing", output_mode="qa",
            strategy="vector_heavy", top_k=min(base_top_k + 2, 8),
            vector_rrf_weight=1.6, keyword_rrf_weight=0.6,
            use_reranker=False, candidate_pool=base_pool,
            max_iterations=1, expand_query=False, signals=signals,
        ))

    # 7. Factual / quantitative
    if _FACTUAL_RE.search(normalized):
        signals.append("factual_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="factual_specific", output_mode="qa",
            strategy="balanced", top_k=base_top_k,
            vector_rrf_weight=1.1, keyword_rrf_weight=1.4,
            use_reranker=reranker_on, candidate_pool=base_pool,
            max_iterations=2, expand_query=False, signals=signals,
        ))

    # 8. Short summary cue (single-query, not full-doc)
    if _SUMMARY_RE.search(normalized):
        signals.append("summary_signal")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="summary_overview", output_mode="qa",
            strategy="vector_heavy", top_k=min(base_top_k + 2, 8),
            vector_rrf_weight=1.7, keyword_rrf_weight=0.6,
            use_reranker=False, candidate_pool=base_pool,
            max_iterations=1, expand_query=False, signals=signals,
        ))

    # 9. Long / highly specific phrase
    if term_count >= 8:
        signals.append("long_specific_query")
        return _emit_plan(query, started_at, OrchestrationPlan(
            query_type="specific_query", output_mode="qa",
            strategy="balanced", top_k=base_top_k,
            vector_rrf_weight=1.0, keyword_rrf_weight=1.5,
            use_reranker=reranker_on, candidate_pool=base_pool,
            max_iterations=1, expand_query=False, signals=signals,
        ))

    # 10. Default hybrid
    signals.append("default_hybrid")
    return _emit_plan(query, started_at, OrchestrationPlan(
        query_type="general", output_mode="qa",
        strategy="balanced", top_k=base_top_k,
        vector_rrf_weight=settings.hybrid_vector_rrf_weight,
        keyword_rrf_weight=settings.hybrid_keyword_rrf_weight,
        use_reranker=reranker_on, candidate_pool=base_pool,
        max_iterations=1, expand_query=False, signals=signals,
    ))


# ─── Internal helper ──────────────────────────────────────────────────────────

def _emit_plan(query: str, started_at: float, plan: OrchestrationPlan) -> OrchestrationPlan:
    elapsed = round((time.perf_counter() - started_at) * 1000, 2)
    _emit_query_progress(
        "[orchestrator] type=%s mode=%s strategy=%s top_k=%d "
        "vw=%.1f kw=%.1f reranker=%s iters=%d signals=%s (%.2fms)",
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
            "elapsed_ms": elapsed,
        },
    )
    return plan
