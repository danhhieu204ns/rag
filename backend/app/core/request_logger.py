"""
Per-request file logger.

Mỗi request (chat query hoặc document embed) sẽ được ghi vào một file log
riêng trong thư mục storage/logs/requests/<kind>/ với định dạng dễ đọc.

Cách dùng:
    from app.core.request_logger import request_logging_context, get_request_logger

    # Bọc một request:
    with request_logging_context("chat", session_id=12, query_preview="...") as req_log:
        req_log.step_start("resolve_session")
        ...
        req_log.step_done("resolve_session")

    # Trong các hàm sâu hơn (rag_runtime, v.v.):
    log = get_request_logger()
    log.info("some message")
"""
from __future__ import annotations

import io
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .settings import settings

# ---------------------------------------------------------------------------
# Module-level logger (chỉ dùng cho warning/error của chính module này)
# ---------------------------------------------------------------------------
_module_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context var: lưu RequestLogger hiện hành cho request đang chạy
# ---------------------------------------------------------------------------
_current_request_logger: ContextVar["RequestLogger | None"] = ContextVar(
    "current_request_logger", default=None
)

_FALLBACK_LOGGER = logging.getLogger("request_logger.fallback")

# ---------------------------------------------------------------------------
# RequestLogger
# ---------------------------------------------------------------------------


class RequestLogger:
    """
    Logger ghi log ra file riêng theo từng request.

    - Mỗi instance tương ứng 1 file log.
    - Thread-safe (dùng Lock).
    - Ghi dạng text thuần dễ đọc, có timestamp tuyệt đối + elapsed từng bước.
    """

    def __init__(
        self,
        *,
        kind: str,
        log_dir: Path,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        self._kind = kind
        self._log_dir = log_dir
        self._extra = extra_fields or {}
        self._lock = threading.Lock()
        self._buf: io.StringIO = io.StringIO()
        self._request_started_at: float = time.perf_counter()
        self._step_started_at: dict[str, float] = {}
        self._file_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def info(self, message: str, *args: object) -> None:
        """Ghi một dòng log thông thường."""
        text = message % args if args else message
        self._write_line("INFO", text)

    def warning(self, message: str, *args: object) -> None:
        text = message % args if args else message
        self._write_line("WARN", text)

    def error(self, message: str, *args: object) -> None:
        text = message % args if args else message
        self._write_line("ERROR", text)

    def step_start(self, step: str, **kwargs: Any) -> None:
        """Đánh dấu bắt đầu một bước xử lý."""
        with self._lock:
            self._step_started_at[step] = time.perf_counter()
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._write_line("STEP↓", f"{step}{extra}")

    def step_done(self, step: str, **kwargs: Any) -> None:
        """Đánh dấu hoàn thành một bước, in thời gian thực thi."""
        elapsed_ms = self._pop_elapsed(step)
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        timing = f"  ⏱ {elapsed_ms:.1f}ms" if elapsed_ms is not None else ""
        self._write_line("STEP✓", f"{step}{timing}{extra}")

    def step_fail(self, step: str, error: str) -> None:
        """Đánh dấu một bước thất bại."""
        elapsed_ms = self._pop_elapsed(step)
        timing = f"  ⏱ {elapsed_ms:.1f}ms" if elapsed_ms is not None else ""
        self._write_line("STEP✗", f"{step}{timing}  error={error}")

    def total_done(self, **kwargs: Any) -> None:
        """Ghi tổng thời gian toàn bộ request."""
        elapsed_ms = (time.perf_counter() - self._request_started_at) * 1000
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._write_line("TOTAL", f"⏱ {elapsed_ms:.1f}ms{extra}")

    def flush_to_file(self) -> Path | None:
        """Ghi toàn bộ buffer vào file log. Trả về đường dẫn file."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._file_path or self._build_file_path()
            self._file_path = file_path
            with self._lock:
                content = self._buf.getvalue()
            file_path.write_text(content, encoding="utf-8")
            return file_path
        except OSError as exc:
            _module_logger.warning("[request_logger] Failed to write log file: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now_str(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _elapsed_request_ms(self) -> float:
        return (time.perf_counter() - self._request_started_at) * 1000

    def _pop_elapsed(self, step: str) -> float | None:
        with self._lock:
            started = self._step_started_at.pop(step, None)
        if started is None:
            return None
        return (time.perf_counter() - started) * 1000

    def _write_line(self, level: str, text: str) -> None:
        elapsed_ms = self._elapsed_request_ms()
        line = f"[{self._now_str()}] [{elapsed_ms:>9.1f}ms] [{level:<6}] {text}\n"
        with self._lock:
            self._buf.write(line)

    def _build_file_path(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        extra_parts = "_".join(
            f"{k}{v}" for k, v in list(self._extra.items())[:2]
            if v is not None
        )
        slug = f"{ts}_{extra_parts}" if extra_parts else ts
        # Giới hạn độ dài tên file
        slug = slug[:120]
        return self._log_dir / f"{slug}.log"


# ---------------------------------------------------------------------------
# _NullLogger: dự phòng khi không có logger trong context
# ---------------------------------------------------------------------------


class _NullLogger:
    """Logger no-op dùng khi không có request context."""

    def info(self, message: str, *args: object) -> None:
        _FALLBACK_LOGGER.debug(message, *args)

    def warning(self, message: str, *args: object) -> None:
        _FALLBACK_LOGGER.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        _FALLBACK_LOGGER.error(message, *args)

    def step_start(self, step: str, **kwargs: Any) -> None:
        pass

    def step_done(self, step: str, **kwargs: Any) -> None:
        pass

    def step_fail(self, step: str, error: str) -> None:
        _FALLBACK_LOGGER.warning("[step_fail] %s: %s", step, error)

    def total_done(self, **kwargs: Any) -> None:
        pass

    def flush_to_file(self) -> None:
        return None


_NULL_LOGGER = _NullLogger()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_request_logger() -> "RequestLogger | _NullLogger":
    """
    Lấy logger cho request hiện tại.
    Nếu không có context (chạy ngoài request), trả về NullLogger.
    """
    return _current_request_logger.get() or _NULL_LOGGER


def _log_dir_for_kind(kind: str) -> Path:
    return settings.storage_dir / "logs" / "requests" / kind


@contextmanager
def request_logging_context(
    kind: str,
    **extra_fields: Any,
) -> Iterator[RequestLogger]:
    """
    Context manager tạo RequestLogger cho một request, gắn vào context var,
    và tự động flush ra file khi kết thúc.

    Ví dụ:
        with request_logging_context("chat", session_id=12) as log:
            log.step_start("resolve_session")
            ...
    """
    log_dir = _log_dir_for_kind(kind)
    req_logger = RequestLogger(kind=kind, log_dir=log_dir, extra_fields=extra_fields)

    # Ghi header
    header_fields = "  ".join(f"{k}={v}" for k, v in extra_fields.items())
    req_logger._write_line("START", f"kind={kind}  {header_fields}")

    token = _current_request_logger.set(req_logger)
    try:
        yield req_logger
    except Exception as exc:
        req_logger._write_line("ERROR", f"Unhandled exception: {exc!r}")
        raise
    finally:
        req_logger.total_done()
        req_logger.flush_to_file()
        _current_request_logger.reset(token)
