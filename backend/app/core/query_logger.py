"""
Per-query structured log writer.

Mỗi query chat sẽ tạo ra một file log riêng tại:
  storage/logs/queries/{YYYYMMDD}/{HHMMSS_ms}_{session_id}_{query_slug}.log

File log ghi lại từng khâu trong pipeline với thời gian và các thông tin
quan trọng (chunk IDs, scores, modes) cho từng bước retrieval + generation.
"""
from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .settings import settings

_current_query_log: ContextVar[QueryLog | None] = ContextVar("current_query_log", default=None)


def get_query_log() -> QueryLog | None:
    """Lấy QueryLog của request hiện tại. Trả về None nếu không có context."""
    return _current_query_log.get()


@contextmanager
def query_logging_context(
    query: str,
    session_id: int | None,
    top_k: int | None,
) -> Iterator[QueryLog]:
    """
    Context manager tạo QueryLog, gắn vào context var,
    và flush ra file khi kết thúc request (kể cả khi có exception).
    """
    log = QueryLog(query=query, session_id=session_id, top_k=top_k)
    previous = _current_query_log.get()
    _current_query_log.set(log)
    try:
        yield log
    finally:
        log.flush_to_file()
        _current_query_log.set(previous)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview(text: str, limit: int = 80) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= limit else t[: limit - 3] + "..."


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^\w\s]", "", text.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:limit].rstrip("_") or "query"


# ---------------------------------------------------------------------------
# QueryLog
# ---------------------------------------------------------------------------

class QueryLog:
    """
    Thu thập dữ liệu từ các event trong pipeline và render ra file log
    dễ đọc theo từng phase: retrieval (semantic, keyword, RRF, rerank) + generation.
    """

    def __init__(self, query: str, session_id: int | None, top_k: int | None) -> None:
        self._query = query
        self._session_id = session_id
        self._top_k = top_k
        self._started_at = time.perf_counter()
        self._ts = datetime.now(timezone.utc)
        self._lock = threading.Lock()
        self._trace_id = ""

        # --- Orchestrator phase ---
        self._orchestrator: dict[str, Any] = {}           # orchestrator_plan (initial)
        self._orchestrator_ms: float = 0.0                # elapsed_ms từ classify_query
        self._orchestrator_retries: list[dict[str, Any]] = []  # orchestrator_retry events

        # --- Retrieval phase ---
        self._kw_start: dict[str, Any] = {}       # keyword_candidates_start
        self._qdrant_hits: dict[str, Any] = {}    # qdrant_child_search_done
        self._semantic: dict[str, Any] = {}       # semantic_candidates_done
        self._keyword: dict[str, Any] = {}        # keyword_candidates_*_done
        self._rrf: dict[str, Any] = {}            # rrf_merge_done
        self._rerank: dict[str, Any] = {}         # rerank_documents
        self._retrieval_service: dict[str, Any] = {}  # delegated retrieval_service summary
        self._final_chunks: list[dict[str, Any]] = []  # similarity_search_done
        self._timing: dict[str, float] = {}       # similarity_search_timing

        # --- Generation phase ---
        self._gen_started_at: float | None = None
        self._gen_elapsed_ms: float = 0.0
        self._gen_answer_len: int = 0

    # ------------------------------------------------------------------
    # Public: event collection
    # ------------------------------------------------------------------

    def record(
        self,
        event: str,
        details: dict[str, Any] | None,
        trace_id: str = "",
    ) -> None:
        """Nhận event từ _emit_query_progress và lưu data liên quan."""
        with self._lock:
            if trace_id and trace_id != "-" and not self._trace_id:
                self._trace_id = trace_id

            if not details:
                return

            if event == "orchestrator_plan":
                if not self._orchestrator:
                    self._orchestrator = details
                    self._orchestrator_ms = float(details.get("elapsed_ms", 0.0))

            elif event == "orchestrator_retry":
                self._orchestrator_retries.append(details)

            elif event == "keyword_candidates_start":
                self._kw_start = details

            elif event == "qdrant_child_search_done":
                self._qdrant_hits = details

            elif event == "semantic_candidates_done":
                self._semantic = details

            elif event == "keyword_candidates_qdrant_done":
                # Chỉ lưu nếu chưa có data từ DB fallback (richer)
                if not self._keyword:
                    self._keyword = details

            elif event == "keyword_candidates_done":
                # DB fallback — có scores, luôn ưu tiên
                self._keyword = details

            elif event == "rrf_merge_done":
                self._rrf = details

            elif event == "rerank_documents":
                self._rerank = details

            elif event == "similarity_search_done":
                self._final_chunks = details.get("final_chunks", [])

            elif event == "similarity_search_timing":
                self._timing = details.get("stage_timings_ms", {})

            elif event == "retrieval_service_done":
                self._retrieval_service = details

    def record_generation_start(self) -> None:
        self._gen_started_at = time.perf_counter()

    def record_generation_done(self, answer_len: int) -> None:
        if self._gen_started_at is not None:
            self._gen_elapsed_ms = (time.perf_counter() - self._gen_started_at) * 1000
        self._gen_answer_len = answer_len

    # ------------------------------------------------------------------
    # Public: flush
    # ------------------------------------------------------------------

    def flush_to_file(self) -> Path | None:
        date_str = self._ts.strftime("%Y%m%d")
        log_dir = settings.storage_dir / "logs" / "queries" / date_str
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None

        ts_str = self._ts.strftime("%H%M%S_%f")[:-3]
        sid = f"s{self._session_id}" if self._session_id is not None else "nosession"
        slug = _slugify(self._query)
        file_path = log_dir / f"{ts_str}_{sid}_{slug}.log"

        try:
            file_path.write_text(self._render(), encoding="utf-8")
            return file_path
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> str:
        W = 66
        total_ms = (time.perf_counter() - self._started_at) * 1000
        lines: list[str] = []

        def sep(c: str = "=") -> None:
            lines.append(c * W)

        def section(title: str) -> None:
            lines.append(f"\n[{title}]")
            lines.append("-" * W)

        def kv(label: str, value: Any, indent: int = 2) -> None:
            pad = " " * indent
            lines.append(f"{pad}{label:<24}: {value}")

        def table_header(*cols: str, indent: int = 4) -> None:
            pad = " " * indent
            lines.append(pad + "  ".join(cols))
            lines.append(pad + "-" * (sum(len(c) for c in cols) + 2 * (len(cols) - 1)))

        # ── HEADER ──────────────────────────────────────────────────────
        sep()
        lines.append("QUERY LOG")
        sep()
        kv("Thời gian", self._ts.strftime("%Y-%m-%d %H:%M:%S UTC"))
        kv("Session ID", self._session_id if self._session_id is not None else "-")
        kv("Trace ID", self._trace_id or "-")
        kv("Câu hỏi", _preview(self._query, 100))
        kv("Top-K", self._top_k if self._top_k is not None else "retrieval_service_default")
        sep()

        # ── PHASE 1 — RETRIEVAL ─────────────────────────────────────────
        section("PHASE 1 — RETRIEVAL")

        # 1a. Orchestrator
        orch_ms = self._orchestrator_ms
        orch_ms_str = f"  [{orch_ms:.2f}ms]" if orch_ms > 0 else ""
        lines.append(f"\n▶ ORCHESTRATOR{orch_ms_str}")
        od = self._orchestrator
        if od:
            kv("  Query type", od.get("query_type", "?"))
            kv("  Output mode", od.get("output_mode", "qa"))
            kv("  Strategy", od.get("strategy", "?"))
            kv("  Top-K (plan)", od.get("top_k", "?"))
            kv("  Retrieval defaults", od.get("retrieval_defaults_source", "retrieval_service"))
            kv("  Max iterations", od.get("max_iterations", 1))
            kv("  Signals", ", ".join(od.get("signals", [])) or "-")
            if self._orchestrator_retries:
                for i, retry in enumerate(self._orchestrator_retries, start=2):
                    prev_n = retry.get("prev_result_count", "?")
                    new_plan = retry.get("new_plan") or {}
                    lines.append(f"  ⚑ Retry #{i}: results thin "
                                 f"({prev_n} chunks) → broader search")
                    kv("    Strategy", new_plan.get("strategy", "?"))
                    kv("    Vector weight", new_plan.get("vector_rrf_weight", "?"))
                    kv("    Keyword weight", new_plan.get("keyword_rrf_weight", "?"))
        else:
            kv("  Trạng thái", "disabled (ORCHESTRATOR_ENABLED=false)")

        # 1c. Semantic Search
        sem_ms = self._timing.get("semantic_candidates", 0.0)
        retrieval_service_ms = self._timing.get("retrieval_service", 0.0)
        retrieval_skipped = self._orchestrator.get("requires_retrieval") is False
        lines.append(f"\n▶ SEMANTIC SEARCH (vector)  [{sem_ms:.1f}ms]")

        hits = self._qdrant_hits.get("points_preview", [])
        if hits:
            kv("  Qdrant child hits", self._qdrant_hits.get("hit_count", "?"))
            lines.append("  Top child hits (cosine similarity score):")
            lines.append(f"    {'score':<10}  {'parent_chunk_id':<18}  child_type")
            lines.append("    " + "-" * 44)
            for h in hits:
                score = h.get("score", 0.0)
                pid = h.get("parent_chunk_id", "?")
                ct = h.get("child_type", "")
                lines.append(f"    {score:<10.4f}  {str(pid):<18}  {ct}")

        d = self._semantic
        if d:
            parent_ids = d.get("semantic_parent_ids", [])
            child_types = d.get("semantic_child_type_preview", {})
            n = min(self._top_k or len(parent_ids), len(parent_ids))
            kv("  Parent chunks tổng", len(parent_ids))
            sem_chunks: dict[int, dict[str, Any]] = {
                int(c["chunk_id"]): c
                for c in d.get("semantic_selected_chunks", [])
                if c.get("chunk_id") is not None
            }
            lines.append(f"  Top {n} parent chunk IDs:")
            if sem_chunks:
                lines.append(f"    {'rank':<7}  {'chunk_id':<12}  {'score':<10}  page   excerpt")
                lines.append("    " + "-" * 66)
                for i, cid in enumerate(parent_ids[:n], start=1):
                    info = sem_chunks.get(int(cid), {})
                    score = info.get("score", "n/a")
                    sc_str = f"{score:.4f}" if isinstance(score, float) else str(score)
                    page = info.get("page") or "-"
                    excerpt = _preview(str(info.get("content", "")), 34)
                    lines.append(f"    #{i:<6}  {str(cid):<12}  {sc_str:<10}  {str(page):<5}  {excerpt}")
            else:
                lines.append(f"    {'rank':<7}  {'chunk_id':<12}  child_type")
                lines.append("    " + "-" * 34)
                for i, cid in enumerate(parent_ids[:n], start=1):
                    ct = child_types.get(cid, "")
                    lines.append(f"    #{i:<6}  {str(cid):<12}  {ct}")
        elif retrieval_skipped:
            kv("  Trạng thái", "skipped by orchestrator (requires_retrieval=false)")
        elif self._retrieval_service:
            kv("  Trạng thái", "delegated to retrieval_service")
            kv("  Service elapsed", f"{retrieval_service_ms:.1f}ms")
        else:
            kv("  Trạng thái", "no data")

        # 1c. Keyword Search
        kw_ms = self._timing.get("keyword_candidates", 0.0)
        lines.append(f"\n▶ KEYWORD SEARCH  [{kw_ms:.1f}ms]")
        ks = self._kw_start
        if ks:
            kv("  Terms", ks.get("query_terms", []))
            codes = ks.get("query_codes", [])
            if codes:
                kv("  Document codes", codes)

        d = self._keyword
        if d:
            selected = d.get("keyword_selected_parent_ids", [])
            n = min(self._top_k or len(selected), len(selected))
            kv("  Parent chunks tổng", len(selected))
            # DB fallback có keyword_selected_chunks với score đầy đủ
            kw_chunks: dict[int, dict[str, Any]] = {
                int(c["chunk_id"]): c
                for c in d.get("keyword_selected_chunks", [])
                if c.get("chunk_id") is not None
            }
            lines.append(f"  Top {n} chunk IDs:")
            if kw_chunks:
                lines.append(f"    {'rank':<7}  {'chunk_id':<12}  {'score':<10}  excerpt")
                lines.append("    " + "-" * 54)
                for i, cid in enumerate(selected[:n], start=1):
                    info = kw_chunks.get(int(cid), {})
                    score = info.get("score", "n/a")
                    sc_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                    excerpt = _preview(str(info.get("content", "")), 30)
                    lines.append(f"    #{i:<6}  {str(cid):<12}  {sc_str:<10}  {excerpt}")
            else:
                lines.append(f"    {'rank':<7}  {'chunk_id':<12}  score")
                lines.append("    " + "-" * 32)
                for i, cid in enumerate(selected[:n], start=1):
                    lines.append(f"    #{i:<6}  {str(cid):<12}  n/a (qdrant fast-path)")
        elif retrieval_skipped:
            kv("  Trạng thái", "skipped by orchestrator (requires_retrieval=false)")
        elif self._retrieval_service:
            kv("  Trạng thái", "delegated to retrieval_service")
            modes = self._retrieval_service.get("mode_counts", {})
            if modes:
                kv("  Returned modes", modes)
        else:
            kv("  Trạng thái", "no data")

        # 1d. RRF Merge
        rrf_ms = self._timing.get("rrf_merge", 0.0)
        lines.append(f"\n▶ RRF MERGE  [{rrf_ms:.2f}ms]")
        d = self._rrf
        if d:
            vec_ids: list[int] = d.get("semantic_parent_ids", [])
            kw_ids: list[int] = d.get("keyword_parent_ids", [])
            merged: list[int] = d.get("rrf_merged_parent_ids", [])
            score_preview: list[dict[str, Any]] = d.get("rrf_score_preview", [])
            kv("  Vector pool", f"{len(vec_ids)} chunks")
            kv("  Keyword pool", f"{len(kw_ids)} chunks")
            kv("  Merged pool", f"{len(merged)} chunks (input cho reranker/output)")
            n = min(self._top_k or len(score_preview), len(score_preview))
            lines.append(f"  Top {n} merged (RRF score):")
            lines.append(f"    {'rank':<7}  {'chunk_id':<12}  {'rrf_score':<14}  mode")
            lines.append("    " + "-" * 50)
            vec_set = set(vec_ids)
            kw_set = set(kw_ids)
            for i, item in enumerate(score_preview[:n], start=1):
                cid = item.get("chunk_id")
                score = item.get("score", 0.0)
                in_v = cid in vec_set
                in_k = cid in kw_set
                mode = "hybrid" if in_v and in_k else ("vector" if in_v else "keyword")
                lines.append(f"    #{i:<6}  {str(cid):<12}  {score:<14.6f}  {mode}")
        elif retrieval_skipped:
            kv("  Trạng thái", "skipped by orchestrator (requires_retrieval=false)")
        elif self._retrieval_service:
            kv("  Trạng thái", "delegated to retrieval_service")
            kv("  Returned contexts", self._retrieval_service.get("raw_context_count", 0))
        else:
            kv("  Trạng thái", "no data")

        # 1e. Reranking
        rerank_ms = self._timing.get("reranker", 0.0)
        reranker_model = (
            self._rerank.get("model")
            or self._retrieval_service.get("reranker_model")
            or "retrieval_service"
        )
        lines.append(f"\n▶ RERANKING ({reranker_model})  [{rerank_ms:.1f}ms]")
        d = self._rerank
        if d:
            in_c = d.get("input_count", 0)
            out_c = d.get("output_count", 0)
            status = d.get("status")
            kv("  Input → Output", f"{in_c} → {out_c} chunks")
            if status:
                kv("  Status", status)
            original_score = d.get("original_top_score")
            reranked_score = d.get("reranked_top_score")
            if isinstance(original_score, (int, float)):
                kv("  Top score trước rerank", f"{original_score:.4f}")
            if isinstance(reranked_score, (int, float)):
                kv("  Top score sau rerank", f"{reranked_score:.4f}")
            if d.get("error"):
                kv("  Error", d.get("error"))
        elif retrieval_skipped:
            kv("  Trạng thái", "skipped by orchestrator (requires_retrieval=false)")
        elif self._retrieval_service:
            kv("  Trạng thái", self._retrieval_service.get("reranker_status", "delegated to retrieval_service"))
        else:
            kv("  Trạng thái", "skipped (pool size <= top_k)")

        # 1f. Kết quả cuối retrieval
        sem_ms_v = self._timing.get("semantic_candidates", 0.0)
        kw_ms_v = self._timing.get("keyword_candidates", 0.0)
        parallel_wall = max(sem_ms_v, kw_ms_v)
        rewrite_ms = self._timing.get("query_rewrite", 0.0)
        embed_ms = self._timing.get("query_embedding", 0.0)
        retrieval_total = (
            retrieval_service_ms
            if retrieval_service_ms > 0
            else rewrite_ms + parallel_wall + rrf_ms + rerank_ms
        )
        lines.append(f"\n▶ KẾT QUẢ CUỐI RETRIEVAL  [tổng wall-clock: {retrieval_total:.1f}ms]")
        if self._final_chunks:
            lines.append(
                f"  {'rank':<7}  {'chunk_id':<10}  {'mode':<10}  "
                f"{'rrf_score':<12}  {'rerank_score':<14}  {'page':<5}  file"
            )
            lines.append("  " + "-" * 72)
            for item in self._final_chunks:
                rank = item.get("rank", "?")
                cid = item.get("chunk_id", "?")
                mode = str(item.get("retrieval_mode") or "")
                rscore = item.get("retrieval_score") or 0.0
                reranker_s = item.get("reranker_score")
                page = item.get("source_page") or "-"
                sm = item.get("source_metadata") or {}
                si = sm.get("source_info") or {} if isinstance(sm, dict) else {}
                fname = si.get("file_name") or "-" if isinstance(si, dict) else "-"
                rerank_str = f"{reranker_s:.4f}" if reranker_s is not None else "n/a"
                lines.append(
                    f"  #{str(rank):<6}  {str(cid):<10}  {mode:<10}  "
                    f"{rscore:<12.4f}  {rerank_str:<14}  {str(page):<5}  {_preview(str(fname), 26)}"
                )
        elif retrieval_skipped:
            kv("  Trạng thái", "skipped by orchestrator (requires_retrieval=false)")
        elif self._retrieval_service:
            kv("  Trạng thái", "retrieval_service returned 0 contexts")
        else:
            kv("  Trạng thái", "no data")

        # ── PHASE 2 — GENERATION ─────────────────────────────────────────
        section("PHASE 2 — GENERATION")
        lines.append(f"\n▶ LLM STREAM ({settings.ollama_chat_model})  [{self._gen_elapsed_ms:.1f}ms]")
        kv("  Context docs", len(self._final_chunks))
        kv("  Answer length", f"{self._gen_answer_len} chars")

        # ── TIMING SUMMARY ───────────────────────────────────────────────
        section("TIMING SUMMARY")
        lines.append("")
        if self._orchestrator_ms > 0:
            retry_count = len(self._orchestrator_retries)
            retry_note = f"  (+{retry_count} retry)" if retry_count else ""
            kv("Orchestrator classify", f"{self._orchestrator_ms:.2f}ms{retry_note}", indent=2)
        if rewrite_ms > 0:
            kv("Query rewrite", f"{rewrite_ms:.1f}ms", indent=2)
        if embed_ms > 0:
            kv("Query embedding", f"{embed_ms:.1f}ms", indent=2)
        if retrieval_service_ms > 0:
            kv("Retrieval service", f"{retrieval_service_ms:.1f}ms", indent=2)
        kv("Semantic search", f"{sem_ms_v:.1f}ms", indent=2)
        kv("Keyword search", f"{kw_ms_v:.1f}ms  (chạy song song với semantic)", indent=2)
        kv("Parallel wall-clock", f"{parallel_wall:.1f}ms  (= max của 2 bước trên)", indent=2)
        kv("RRF merge", f"{rrf_ms:.2f}ms", indent=2)
        kv("Reranking", f"{rerank_ms:.1f}ms", indent=2)
        lines.append("  " + "-" * 44)
        retrieval_total_with_orch = retrieval_total + self._orchestrator_ms
        kv("Retrieval tổng", f"{retrieval_total_with_orch:.1f}ms", indent=2)
        kv("Generation", f"{self._gen_elapsed_ms:.1f}ms", indent=2)
        lines.append("  " + "=" * 44)
        kv("TỔNG (end-to-end)", f"{total_ms:.1f}ms", indent=2)
        lines.append("")
        sep()

        return "\n".join(lines) + "\n"
