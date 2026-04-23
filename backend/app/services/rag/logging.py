from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterator

from ...core.request_logger import get_request_logger
from ...core.settings import settings
from .utils import _json_safe_value

logger = logging.getLogger(__name__)

_QUERY_LOG_FILE_NAME = "query_trace.json"
_LEGACY_QUERY_LOG_FILE_NAME = "query_trace.jsonl"
_query_trace_id_ctx: ContextVar[str] = ContextVar("query_trace_id", default="-")


def _query_log_file_path() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _QUERY_LOG_FILE_NAME


def _legacy_query_log_file_path() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / _LEGACY_QUERY_LOG_FILE_NAME


def _load_query_log_entries(log_file: Path) -> list[dict[str, Any]]:
    if log_file.exists():
        try:
            raw = log_file.read_text(encoding="utf-8").strip()
            if not raw:
                return []

            payload = json.loads(raw)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict):
                entries = payload.get("entries")
                if isinstance(entries, list):
                    return [item for item in entries if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            return []

    legacy_log_file = _legacy_query_log_file_path()
    if not legacy_log_file.exists():
        return []

    migrated_entries: list[dict[str, Any]] = []
    try:
        for raw_line in legacy_log_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                migrated_entries.append(row)
    except OSError:
        return []

    return migrated_entries


def _append_query_log_entry(event: str, details: dict[str, Any] | None = None) -> None:
    trace_id = _query_trace_id_ctx.get()
    payload: dict[str, Any] = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
        "event": event,
    }

    if details:
        payload["details"] = _json_safe_value(details)

    try:
        log_file = _query_log_file_path()
        entries = _load_query_log_entries(log_file)
        entries.append(payload)
        log_file.write_text(
            json.dumps({"entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("[query][file-log] Failed to write query log file: %s", exc)


def _emit_query_progress(
    message: str,
    *args: object,
    event: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raw_text = message % args if args else message
    trace_id = _query_trace_id_ctx.get()

    # Tự động nối các trường thời gian vào message nếu có trong details
    time_fields = []
    if details:
        for k in ["elapsed_ms", "predict_elapsed_ms", "duration_ms", "step_time_ms"]:
            if k in details:
                time_fields.append(f"{k}={details[k]}")
    time_info = f" | {'; '.join(time_fields)}" if time_fields else ""

    text = raw_text + time_info
    if trace_id and trace_id != "-":
        text = f"[trace={trace_id}] {raw_text}{time_info}"

    # Ghi vào per-request file logger (không ra terminal)
    get_request_logger().info(text)

    # Vẫn ghi vào file trace JSON để backward compat
    _append_query_log_entry(
        event=event or "progress",
        details={
            "message": raw_text,
            **(details or {}),
        },
    )


@contextmanager
def _timed_query_step(
    step: str,
    *,
    event_prefix: str,
    details: dict[str, Any] | None = None,
) -> Iterator[None]:
    started_at = time.perf_counter()
    base_details = dict(details or {})
    _emit_query_progress(
        "[timing][query] START step=%s",
        step,
        event=f"{event_prefix}_start",
        details={"step": step, **base_details},
    )
    try:
        yield
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        _emit_query_progress(
            "[timing][query] FAIL step=%s elapsed_ms=%.2f error=%s",
            step,
            elapsed_ms,
            str(exc),
            event=f"{event_prefix}_error",
            details={"step": step, "elapsed_ms": elapsed_ms, "error": str(exc), **base_details},
        )
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    _emit_query_progress(
        "[timing][query] DONE step=%s elapsed_ms=%.2f",
        step,
        elapsed_ms,
        event=f"{event_prefix}_done",
        details={"step": step, "elapsed_ms": elapsed_ms, **base_details},
    )


def _emit_reindex_progress(message: str, *args: object) -> None:
    text = message % args if args else message
    # Ghi vào per-request file logger (không ra terminal)
    get_request_logger().info(text)
