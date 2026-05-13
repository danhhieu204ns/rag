from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import logging
import re
import os
import time
from contextlib import contextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
import httpx
from pydantic import BaseModel, Field

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent
load_dotenv(_REPO_ROOT / ".env", override=False)
load_dotenv(_SERVICE_ROOT / ".env", override=True)

from .core.settings import settings
from .services.document_processing import (
    load_documents_from_parsed_markdown,
    parse_source_to_markdown,
    split_source_documents,
)


logger = logging.getLogger(__name__)


def _configure_ingestion_file_logging() -> Path:
    log_dir = settings.storage_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ingestion_service_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler_exists = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_filename = getattr(handler, "baseFilename", "")
            if Path(base_filename) == log_path:
                file_handler_exists = True
                handler.setFormatter(formatter)
                break

    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return log_path


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


@contextmanager
def _timed_step(step: str, **context: Any):
    started = time.perf_counter()
    details = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info("[ingestion][timing] step=%s status=start %s", step, details)
    try:
        yield
        logger.info("[ingestion][timing] step=%s status=ok elapsed_ms=%.2f %s", step, _ms(started), details)
    except Exception:
        logger.exception("[ingestion][timing] step=%s status=error elapsed_ms=%.2f %s", step, _ms(started), details)
        raise


class SplitRequest(BaseModel):
    markdown: str = Field(default="")
    source_file_path: str
    source_parser: str = "legacy"
    source_type: str = "text"
    chunk_size: int = 500
    chunk_overlap: int = 50


class ChunkPayload(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SplitResponse(BaseModel):
    chunks: list[ChunkPayload]


class ParseResponse(BaseModel):
    markdown: str
    source_parser: str
    source_type: str


class IndexBuildChunkPayload(BaseModel):
    chunk_index: int
    content: str
    source_page: int | None = None
    source_kind: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexBuildChildPayload(BaseModel):
    chunk_index: int
    child_type: str
    child_index: int
    child_text: str
    source_page: int | None = None
    source_kind: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexBuildResponse(BaseModel):
    parent_chunks: list[IndexBuildChunkPayload]
    child_rows: list[IndexBuildChildPayload]


class IndexUpsertChildPayload(BaseModel):
    document_id: int
    parent_chunk_id: int
    source_page: int | None = None
    child_type: str
    child_index: int
    child_text: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexUpsertRequest(BaseModel):
    document_id: int
    child_rows: list[IndexUpsertChildPayload] = Field(default_factory=list)


class IndexUpsertResponse(BaseModel):
    indexed_chunks: int


_DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b")
_DOC_CODE_PATTERN = re.compile(r"\b\d{1,6}[/-][A-Za-z]{1,12}(?:[/-][A-Za-z0-9]{1,16})+\b")
_INDEXING_INSTRUCTION = """
Bạn là hệ thống xử lý tài liệu cho RAG indexing.

Hãy phân tích văn bản và trả về JSON hợp lệ, không giải thích thêm.
Ràng buộc output:
- Chỉ trả về đúng 1 JSON object.
- `summary` tối đa 5 câu.
- `hyq` đúng 3 câu hỏi, mỗi câu tối đa 160 ký tự.
- Không lặp lại bảng, HTML, số trang, danh mục dẫn chiếu hoặc nội dung dạng `<br>`.

Schema bắt buộc:
{
  "summary": "Tóm tắt ngắn 3-5 câu",
  "hyq": [
    "Câu hỏi giả định 1",
    "Câu hỏi giả định 2",
    "Câu hỏi giả định 3"
  ],
  "metadata": {
    "title": null,
    "topic": null,
    "keywords": [],
    "document_type": null,
    "department_or_unit": null,
    "date": null,
    "people": [],
    "risk_level": "low"
  },
  "language": "vi"
}
""".strip()


def _normalize_spaces(text: str) -> str:
    return " ".join(str(text or "").split())


def _extract_context(raw_metadata: dict[str, Any]) -> dict[str, str | None]:
    headers = raw_metadata.get("markdown_headers")
    if not isinstance(headers, dict):
        return {"h2": None, "h3": None}
    h2 = headers.get("h2") or headers.get("h1")
    h3 = headers.get("h3") or headers.get("h4")
    return {
        "h2": _normalize_spaces(str(h2)) if h2 else None,
        "h3": _normalize_spaces(str(h3)) if h3 else None,
    }


def _source_kind(metadata: dict[str, Any], suffix: str) -> str:
    source_parser = str(metadata.get("source_parser") or "legacy").lower()
    source_type = str(metadata.get("source_type") or "").lower() or ("pdf" if suffix == ".pdf" else "text")
    if source_type == "pdf" and source_parser == "marker":
        return "pdf_marker_page"
    if source_type == "pdf":
        return "pdf_page"
    if source_type == "text":
        return "text_chunk"
    return source_type


def _fallback_search(chunk_text: str) -> dict[str, list[str]]:
    dates = list(dict.fromkeys(_DATE_PATTERN.findall(chunk_text)))[:15]
    doc_codes = list(dict.fromkeys(code.upper() for code in _DOC_CODE_PATTERN.findall(chunk_text)))[:15]
    keywords = [*doc_codes, *dates][:20]
    return {
        "keywords": keywords,
        "entities": [],
        "organizations": [],
        "dates": dates,
        "document_codes": doc_codes,
    }


def _build_hyq(summary: str, hyq_questions: list[str], context: dict[str, str | None]) -> dict[str, Any]:
    normalized_questions = [q if q.endswith("?") else f"{q}?" for q in hyq_questions if str(q).strip()]
    if not normalized_questions:
        if context.get("h3"):
            normalized_questions = [f"{context['h3']} được trình bày như thế nào?"]
        elif context.get("h2"):
            normalized_questions = [f"Nội dung trong mục {context['h2']} là gì?"]
        else:
            normalized_questions = ["Thông tin chính của đoạn này là gì?"]
    return {
        "summary": _normalize_spaces(summary) or "Không có tóm tắt.",
        "questions": normalized_questions[:3],
    }


def _indexing_headers() -> dict[str, str]:
    return {"x-api-key": settings.api_key} if settings.api_key else {}


def _indexing_batch(chunk_texts: list[str]) -> list[dict[str, Any]]:
    if not chunk_texts:
        return []
    ollama_base = str(os.getenv("OLLAMA_BASE_URL", "")).strip().rstrip("/")
    if not ollama_base:
        raise RuntimeError("OLLAMA_BASE_URL is required for indexing enrichment in ingestion_service.")
    endpoint = f"{ollama_base}/v1/indexing/batch"

    results: list[dict[str, Any]] = []
    batch_size = settings.indexing_batch_size
    with httpx.Client(timeout=settings.indexing_timeout_seconds, headers=_indexing_headers()) as client:
        for start in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[start : start + batch_size]
            payload = {
                "texts": batch,
                "instruction": _INDEXING_INSTRUCTION,
                "options": {"num_predict": 768},
            }
            batch_started = time.perf_counter()
            try:
                response = client.post(endpoint, json=payload)
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    f"Indexing batch request timed out for chunks {start}-{start + len(batch) - 1} "
                    f"after {settings.indexing_timeout_seconds:.0f}s. "
                    f"Increase INDEXING_TIMEOUT_SECONDS or lower INDEXING_BATCH_SIZE."
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(
                    f"Indexing batch request failed for chunks {start}-{start + len(batch) - 1}: {exc}"
                ) from exc
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Indexing batch request failed for chunks {start}-{start + len(batch) - 1}: "
                    f"{response.status_code} {response.text}"
                )
            try:
                upstream = response.json()
            except ValueError as exc:
                raise RuntimeError(f"Indexing batch returned non-JSON response for chunks {start}-{start + len(batch) - 1}.") from exc
            items = upstream.get("items") if isinstance(upstream, dict) else None
            if not isinstance(items, list):
                raise RuntimeError(f"Indexing batch response missing items list for chunks {start}-{start + len(batch) - 1}.")
            if len(items) != len(batch):
                raise RuntimeError(
                    f"Indexing batch response item count mismatch for chunks {start}-{start + len(batch) - 1}: "
                    f"expected {len(batch)}, got {len(items)}."
                )

            for offset, item in enumerate(items):
                index = start + offset
                if not isinstance(item, dict):
                    raise RuntimeError(f"Indexing batch returned invalid item at index {index}.")
                if item.get("error"):
                    raise RuntimeError(f"Index enrichment failed for chunk {index}: {item['error']}")
                results.append(item)
            logger.info(
                "[ingestion][timing] step=enrich_batch.call_indexing_batch_api.sub_batch status=ok elapsed_ms=%.2f batch_start=%d batch_size=%d indexed_total=%d",
                _ms(batch_started),
                start,
                len(batch),
                len(results),
            )
    return results

app = FastAPI(title=settings.app_name)
_INGESTION_LOG_PATH = _configure_ingestion_file_logging()
logger.info("[ingestion] file logging enabled path=%s", _INGESTION_LOG_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ingestion-service",
        "version": "1.0.0",
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ingestion-service",
        "parser_mode": settings.pdf_parser_mode,
    }


@app.post("/v1/parse", response_model=ParseResponse)
async def parse(file: UploadFile = File(...)) -> ParseResponse:
    request_started = time.perf_counter()
    suffix = Path(file.filename or "").suffix.lower()
    logger.info("[ingestion] /v1/parse filename=%s suffix=%s", file.filename, suffix)
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")

    with TemporaryDirectory(prefix="ingestion_upload_") as tmp_dir:
        temp_path = Path(tmp_dir) / (file.filename or "uploaded.bin")
        with _timed_step("parse.read_upload", filename=file.filename):
            payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with _timed_step("parse.write_temp_file", temp_path=temp_path):
            temp_path.write_bytes(payload)

        try:
            with _timed_step("parse.to_markdown", parser_mode=settings.pdf_parser_mode, suffix=suffix):
                markdown, source_parser, source_type = parse_source_to_markdown(temp_path)
        except Exception as exc:  # pragma: no cover
            logger.exception("[ingestion] /v1/parse failed filename=%s", file.filename)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "[ingestion] /v1/parse done filename=%s parser=%s type=%s markdown_chars=%d",
        file.filename,
        source_parser,
        source_type,
        len(markdown),
    )
    logger.info(
        "[ingestion][timing] route=/v1/parse status=ok total_elapsed_ms=%.2f filename=%s",
        _ms(request_started),
        file.filename,
    )
    return ParseResponse(
        markdown=markdown,
        source_parser=source_parser,
        source_type=source_type,
    )


@app.post("/v1/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
    request_started = time.perf_counter()
    logger.info(
        "[ingestion] /v1/split source=%s parser=%s type=%s chars=%d chunk_size=%d overlap=%d",
        request.source_file_path,
        request.source_parser,
        request.source_type,
        len(request.markdown),
        request.chunk_size,
        request.chunk_overlap,
    )
    with TemporaryDirectory(prefix="ingestion_markdown_") as tmp_dir:
        markdown_path = Path(tmp_dir) / "parsed.md"
        with _timed_step("split.write_markdown", source=request.source_file_path):
            markdown_path.write_text(request.markdown.strip(), encoding="utf-8")

        try:
            with _timed_step("split.load_documents", source=request.source_file_path):
                loaded = load_documents_from_parsed_markdown(
                    markdown_path,
                    source_file_path=Path(request.source_file_path),
                    source_parser=request.source_parser,
                    source_type=request.source_type,
                )
            with _timed_step(
                "split.chunk_documents",
                source=request.source_file_path,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                loaded_docs=len(loaded),
            ):
                chunks = split_source_documents(
                    loaded,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                )
        except Exception as exc:  # pragma: no cover
            logger.exception("[ingestion] /v1/split failed source=%s", request.source_file_path)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[ingestion] /v1/split done source=%s chunks=%d", request.source_file_path, len(chunks))
    logger.info(
        "[ingestion][timing] route=/v1/split status=ok total_elapsed_ms=%.2f source=%s",
        _ms(request_started),
        request.source_file_path,
    )
    return SplitResponse(
        chunks=[
            ChunkPayload(page_content=item.page_content, metadata=dict(item.metadata or {}))
            for item in chunks
            if str(item.page_content or "").strip()
        ]
    )


@app.post("/v1/index/build", response_model=IndexBuildResponse)
async def index_build(file: UploadFile = File(...)) -> IndexBuildResponse:
    request_started = time.perf_counter()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")

    with TemporaryDirectory(prefix="ingestion_index_build_") as tmp_dir:
        temp_path = Path(tmp_dir) / (file.filename or "uploaded.bin")
        with _timed_step("index_build.read_upload", filename=file.filename):
            payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with _timed_step("index_build.write_temp_file", temp_path=temp_path):
            temp_path.write_bytes(payload)

        with _timed_step("index_build.parse_to_markdown", parser_mode=settings.pdf_parser_mode, suffix=suffix):
            markdown, source_parser, source_type = parse_source_to_markdown(temp_path)
        markdown_path = Path(tmp_dir) / "parsed.md"
        with _timed_step("index_build.write_markdown", markdown_chars=len(markdown)):
            markdown_path.write_text(markdown.strip(), encoding="utf-8")
        with _timed_step("index_build.load_documents"):
            loaded = load_documents_from_parsed_markdown(
                markdown_path,
                source_file_path=temp_path,
                source_parser=source_parser,
                source_type=source_type,
            )
        with _timed_step("index_build.chunk_documents", loaded_docs=len(loaded), chunk_size=1000, chunk_overlap=150):
            chunks = split_source_documents(
                loaded,
                chunk_size=1000,
                chunk_overlap=150,
            )

    parent_chunks: list[IndexBuildChunkPayload] = []
    child_rows: list[IndexBuildChildPayload] = []
    texts: list[str] = []
    raw_rows: list[tuple[int, str, int | None, str, dict[str, Any], dict[str, str | None]]] = []

    prepare_started = time.perf_counter()
    for item in chunks:
        text = str(item.page_content or "").strip()
        if not text:
            continue
        metadata = dict(item.metadata or {})
        chunk_index = len(parent_chunks)
        source_page = metadata.get("source_page") if isinstance(metadata.get("source_page"), int) else None
        source_kind = _source_kind(metadata, suffix)
        context = _extract_context(metadata)
        parent_chunks.append(
            IndexBuildChunkPayload(
                chunk_index=chunk_index,
                content=text,
                source_page=source_page,
                source_kind=source_kind,
                source_metadata={},
            )
        )
        texts.append(text)
        raw_rows.append((chunk_index, text, source_page, source_kind, metadata, context))
    logger.info(
        "[ingestion][timing] step=enrich_batch.prepare_payload status=ok elapsed_ms=%.2f batch_size=%d input_chars=%d",
        _ms(prepare_started),
        len(texts),
        sum(len(t) for t in texts),
    )

    try:
        with _timed_step("enrich_batch.call_indexing_batch_api", model_name="ollama_indexing_batch", batch_size=len(texts)):
            llm_items = _indexing_batch(texts)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    generate_started = time.perf_counter()
    total_children_generated = 0
    total_parent_elapsed_ms = 0.0
    for idx, raw in enumerate(raw_rows):
        parent_started = time.perf_counter()
        chunk_index, text, source_page, source_kind, metadata, context = raw
        llm_item = llm_items[idx] if idx < len(llm_items) else {}
        summary = str(llm_item.get("summary") or "")
        hyq = llm_item.get("hyq")
        questions = [str(q) for q in hyq] if isinstance(hyq, list) else []
        meta = llm_item.get("metadata") if isinstance(llm_item.get("metadata"), dict) else {}
        keywords = [str(k) for k in (meta.get("keywords") or []) if str(k).strip()]
        search_opt = _fallback_search(text)
        if keywords:
            search_opt["keywords"] = list(dict.fromkeys([*search_opt["keywords"], *keywords]))[:20]
            search_opt["entities"] = keywords[:15]

        metadata_started = time.perf_counter()
        structured = {
            "source_info": {
                "file_name": str(file.filename or ""),
                "page_number": source_page,
                "doc_type": "Tài_liệu_nội_bộ",
            },
            "context": context,
            "search_optimization": search_opt,
            "admin_tags": {
                "security_level": "Nội_bộ",
                "department": "Tổng_hợp",
            },
            "hyq": _build_hyq(summary, questions, context),
        }
        logger.info(
            "[ingestion][timing] step=enrich_batch.build_metadata status=ok elapsed_ms=%.2f chunk_index=%d",
            _ms(metadata_started),
            chunk_index,
        )
        parent_chunks[idx].source_metadata = structured
        children_before = len(child_rows)
        child_rows.append(
            IndexBuildChildPayload(
                chunk_index=chunk_index,
                child_type="summary",
                child_index=0,
                child_text=f"Tóm tắt: {structured['hyq']['summary']}",
                source_page=source_page,
                source_kind=source_kind,
                source_metadata=structured,
            )
        )
        for q_idx, q in enumerate(structured["hyq"]["questions"], start=1):
            child_rows.append(
                IndexBuildChildPayload(
                    chunk_index=chunk_index,
                    child_type="question",
                    child_index=q_idx,
                    child_text=str(q),
                    source_page=source_page,
                    source_kind=source_kind,
                    source_metadata=structured,
                )
            )
        generated_for_parent = len(child_rows) - children_before
        total_children_generated += generated_for_parent
        parent_elapsed_ms = _ms(parent_started)
        total_parent_elapsed_ms += parent_elapsed_ms
        elapsed_per_child_ms = parent_elapsed_ms / generated_for_parent if generated_for_parent > 0 else 0.0
        logger.info(
            "[ingestion][timing] step=enrich_batch.generate_child_chunks status=ok chunk_index=%d output_children=%d elapsed_per_parent_ms=%.2f elapsed_per_child_ms=%.2f",
            chunk_index,
            generated_for_parent,
            parent_elapsed_ms,
            elapsed_per_child_ms,
        )

    logger.info(
        "[ingestion][timing] step=enrich_batch.postprocess status=ok elapsed_ms=%.2f parent_chunks=%d output_children=%d avg_elapsed_per_parent_ms=%.2f",
        _ms(generate_started),
        len(parent_chunks),
        total_children_generated,
        (total_parent_elapsed_ms / len(raw_rows)) if raw_rows else 0.0,
    )
    logger.info(
        "[ingestion][timing] route=/v1/index/build status=ok total_elapsed_ms=%.2f filename=%s parent_chunks=%d child_rows=%d",
        _ms(request_started),
        file.filename,
        len(parent_chunks),
        len(child_rows),
    )
    return IndexBuildResponse(parent_chunks=parent_chunks, child_rows=child_rows)


@app.post("/v1/index/upsert", response_model=IndexUpsertResponse)
def index_upsert(request: IndexUpsertRequest) -> IndexUpsertResponse:
    request_started = time.perf_counter()
    if not settings.retrieval_service_url:
        raise HTTPException(status_code=500, detail="RETRIEVAL_SERVICE_URL is not configured in ingestion_service.")
    if not request.child_rows:
        return IndexUpsertResponse(indexed_chunks=0)

    chunks = [
        {
            "chunk_id": f"{row.parent_chunk_id}:{row.child_type}:{row.child_index}",
            "document_id": row.document_id,
            "content": row.child_text,
            "page": row.source_page,
            "metadata": {
                "document_id": row.document_id,
                "parent_chunk_id": row.parent_chunk_id,
                "chunk_id": row.parent_chunk_id,
                "source_page": row.source_page,
                "source_metadata": row.source_metadata,
                "child_type": row.child_type,
                "child_index": row.child_index,
            },
        }
        for row in request.child_rows
    ]

    batch_size = 100
    indexed_total = 0
    with httpx.Client(timeout=settings.retrieval_timeout_seconds, headers=_indexing_headers()) as client:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            payload = {
                "purge_document_ids": [request.document_id] if start == 0 else [],
                "chunks": batch,
            }
            batch_started = time.perf_counter()
            resp = client.post(f"{settings.retrieval_service_url}/v1/index/chunks", json=payload)
            if resp.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Retrieval indexing failed: {resp.status_code} {resp.text}")
            result = resp.json()
            indexed_total += int(result.get("indexed_chunks") or 0) if isinstance(result, dict) else 0
            logger.info(
                "[ingestion][timing] step=index_upsert.batch_post status=ok elapsed_ms=%.2f document_id=%s batch_start=%d batch_size=%d indexed_total=%d",
                _ms(batch_started),
                request.document_id,
                start,
                len(batch),
                indexed_total,
            )
    logger.info(
        "[ingestion][timing] route=/v1/index/upsert status=ok total_elapsed_ms=%.2f document_id=%s indexed_chunks=%d",
        _ms(request_started),
        request.document_id,
        indexed_total,
    )
    return IndexUpsertResponse(indexed_chunks=indexed_total)
