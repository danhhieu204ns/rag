from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import logging
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
from .services.indexing_pipeline import build_section_parent_child_index
from .services.parent_child_chunker import INDEX_TYPE


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


class IndexBuildFromMarkdownRequest(BaseModel):
    markdown: str = Field(default="")
    source_file_path: str
    source_parser: str = "legacy"
    source_type: str = "text"


class IndexBuildChunkPayload(BaseModel):
    parent_id: str
    section_id: str
    chunk_index: int
    content: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    source_page: int | None = None
    source_kind: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexBuildChildPayload(BaseModel):
    chunk_index: int
    chunk_id: str
    parent_id: str
    section_id: str
    child_type: str
    child_index: int
    child_text: str
    embedding_text: str
    token_count: int
    title: str
    section_title: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_page: int | None = None
    source_kind: str
    index_type: str = INDEX_TYPE
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexBuildResponse(BaseModel):
    parent_chunks: list[IndexBuildChunkPayload]
    child_rows: list[IndexBuildChildPayload]


class IndexUpsertChildPayload(BaseModel):
    document_id: int
    parent_chunk_id: int
    chunk_id: str
    parent_id: str | None = None
    section_id: str | None = None
    source_page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    child_type: str = "section_child"
    child_index: int
    child_text: str
    embedding_text: str
    token_count: int | None = None
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    index_type: str = INDEX_TYPE
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexUpsertRequest(BaseModel):
    document_id: int
    child_rows: list[IndexUpsertChildPayload] = Field(default_factory=list)


class IndexUpsertResponse(BaseModel):
    indexed_chunks: int


def _indexing_headers() -> dict[str, str]:
    return {"x-api-key": settings.api_key} if settings.api_key else {}

app = FastAPI(title=settings.app_name)
_INGESTION_LOG_PATH = _configure_ingestion_file_logging()
logger.info("[ingestion] file logging enabled path=%s", _INGESTION_LOG_PATH)


def _build_index_response_from_markdown(
    *,
    markdown: str,
    source_file_path: Path,
    source_parser: str,
    source_type: str,
    filename: str,
    route_name: str,
    request_started: float,
) -> IndexBuildResponse:
    if not markdown.strip():
        raise HTTPException(status_code=400, detail="Parsed markdown is empty.")

    if settings.chunking_strategy != INDEX_TYPE or settings.indexing_index_type != INDEX_TYPE:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported indexing strategy. Expected {INDEX_TYPE}.",
        )

    try:
        with _timed_step(
            f"{route_name}.section_parent_child_chunking",
            markdown_chars=len(markdown),
            parent_max_tokens=settings.parent_max_tokens,
            child_chunk_size=settings.child_chunk_size,
            child_chunk_overlap=settings.child_chunk_overlap,
        ):
            index_bundle = build_section_parent_child_index(
                markdown=markdown,
                source_file_path=source_file_path,
                source_parser=source_parser,
                source_type=source_type,
                parent_max_tokens=settings.parent_max_tokens,
                child_chunk_size=settings.child_chunk_size,
                child_chunk_overlap=settings.child_chunk_overlap,
                prepend_heading_path=settings.prepend_heading_path,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    parent_chunks = [IndexBuildChunkPayload(**item) for item in index_bundle.parent_payloads]
    child_rows = [IndexBuildChildPayload(**item) for item in index_bundle.child_payloads]
    logger.info(
        "[ingestion][timing] route=/%s status=ok total_elapsed_ms=%.2f filename=%s parent_chunks=%d child_rows=%d",
        route_name,
        _ms(request_started),
        filename,
        len(parent_chunks),
        len(child_rows),
    )
    return IndexBuildResponse(parent_chunks=parent_chunks, child_rows=child_rows)


@app.get("/health")
def health() -> dict[str, str]:
    request_start = time.perf_counter()
    logger.info("[ingestion][health] request received")
    result = {
        "status": "ok",
        "service": "ingestion-service",
        "version": "1.0.0",
    }
    logger.info("[ingestion][health] response sent elapsed_ms=%.2f", _ms(request_start))
    return result


@app.get("/ready")
def ready() -> dict[str, str]:
    request_start = time.perf_counter()
    logger.info("[ingestion][ready] request received parser_mode=%s", settings.pdf_parser_mode)
    result = {
        "status": "ok",
        "service": "ingestion-service",
        "parser_mode": settings.pdf_parser_mode,
    }
    logger.info("[ingestion][ready] response sent elapsed_ms=%.2f", _ms(request_start))
    return result


@app.post("/v1/parse", response_model=ParseResponse)
async def parse(file: UploadFile = File(...)) -> ParseResponse:
    request_started = time.perf_counter()
    suffix = Path(file.filename or "").suffix.lower()
    logger.info("[ingestion][parse] request received filename=%s suffix=%s", file.filename, suffix)
    
    logger.debug("[ingestion][parse] step=validate_file_type suffix=%s", suffix)
    if suffix not in {".pdf", ".txt", ".md"}:
        logger.warning("[ingestion][parse] unsupported file type suffix=%s", suffix)
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")
    logger.debug("[ingestion][parse] step=validate_ok suffix=%s", suffix)

    with TemporaryDirectory(prefix="ingestion_upload_") as tmp_dir:
        temp_path = Path(tmp_dir) / (file.filename or "uploaded.bin")
        with _timed_step("parse.read_upload", filename=file.filename):
            logger.debug("[ingestion][parse] step=read_upload filename=%s", file.filename)
            payload = await file.read()
        
        if not payload:
            logger.error("[ingestion][parse] uploaded file is empty filename=%s", file.filename)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        logger.debug("[ingestion][parse] step=write_temp_file temp_path=%s file_size=%d", temp_path, len(payload))
        with _timed_step("parse.write_temp_file", temp_path=temp_path):
            temp_path.write_bytes(payload)

        try:
            logger.debug("[ingestion][parse] step=parse_to_markdown parser_mode=%s suffix=%s", settings.pdf_parser_mode, suffix)
            with _timed_step("parse.to_markdown", parser_mode=settings.pdf_parser_mode, suffix=suffix):
                markdown, source_parser, source_type = parse_source_to_markdown(temp_path)
        except Exception as exc:
            logger.exception("[ingestion][parse] parsing failed filename=%s error=%s", file.filename, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "[ingestion][parse] parsing complete filename=%s parser=%s type=%s markdown_chars=%d",
        file.filename,
        source_parser,
        source_type,
        len(markdown),
    )
    logger.info(
        "[ingestion][timing] route=/v1/parse status=ok total_elapsed_ms=%.2f filename=%s markdown_chars=%d",
        _ms(request_started),
        file.filename,
        len(markdown),
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
        "[ingestion][split] request received source=%s chars=%d chunk_size=%d overlap=%d",
        request.source_file_path,
        len(request.markdown),
        request.chunk_size,
        request.chunk_overlap,
    )
    
    logger.debug("[ingestion][split] step=validate_input parser=%s type=%s", request.source_parser, request.source_type)
    if not request.markdown.strip():
        logger.warning("[ingestion][split] empty markdown source=%s", request.source_file_path)
        raise HTTPException(status_code=400, detail="Markdown content is empty.")
    logger.debug("[ingestion][split] step=validate_ok chars=%d", len(request.markdown))
    
    with TemporaryDirectory(prefix="ingestion_markdown_") as tmp_dir:
        markdown_path = Path(tmp_dir) / "parsed.md"
        logger.debug("[ingestion][split] step=write_markdown temp_path=%s chars=%d", markdown_path, len(request.markdown))
        with _timed_step("split.write_markdown", source=request.source_file_path):
            markdown_path.write_text(request.markdown.strip(), encoding="utf-8")

        try:
            logger.debug("[ingestion][split] step=load_documents source=%s", request.source_file_path)
            with _timed_step("split.load_documents", source=request.source_file_path):
                loaded = load_documents_from_parsed_markdown(
                    markdown_path,
                    source_file_path=Path(request.source_file_path),
                    source_parser=request.source_parser,
                    source_type=request.source_type,
                )
            logger.debug("[ingestion][split] step=load_ok loaded_docs=%d", len(loaded))
            
            logger.debug("[ingestion][split] step=chunk_documents chunk_size=%d overlap=%d loaded=%d", request.chunk_size, request.chunk_overlap, len(loaded))
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
            logger.debug("[ingestion][split] step=chunk_ok chunks=%d", len(chunks))
        except Exception as exc:
            logger.exception("[ingestion][split] chunking failed source=%s error=%s", request.source_file_path, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[ingestion][split] splitting complete source=%s chunks=%d", request.source_file_path, len(chunks))
    logger.info(
        "[ingestion][timing] route=/v1/split status=ok total_elapsed_ms=%.2f source=%s chunks=%d",
        _ms(request_started),
        request.source_file_path,
        len(chunks),
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
    logger.info("[ingestion][index_build] request received filename=%s suffix=%s", file.filename, suffix)
    
    logger.debug("[ingestion][index_build] step=validate_file_type suffix=%s", suffix)
    if suffix not in {".pdf", ".txt", ".md"}:
        logger.warning("[ingestion][index_build] unsupported file type suffix=%s", suffix)
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")
    logger.debug("[ingestion][index_build] step=validate_ok suffix=%s", suffix)

    with TemporaryDirectory(prefix="ingestion_index_build_") as tmp_dir:
        temp_path = Path(tmp_dir) / (file.filename or "uploaded.bin")
        logger.debug("[ingestion][index_build] step=read_upload filename=%s", file.filename)
        with _timed_step("index_build.read_upload", filename=file.filename):
            payload = await file.read()
        
        if not payload:
            logger.error("[ingestion][index_build] uploaded file is empty filename=%s", file.filename)
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        logger.debug("[ingestion][index_build] step=write_temp_file temp_path=%s file_size=%d", temp_path, len(payload))
        with _timed_step("index_build.write_temp_file", temp_path=temp_path):
            temp_path.write_bytes(payload)

        logger.debug("[ingestion][index_build] step=parse_to_markdown parser_mode=%s suffix=%s", settings.pdf_parser_mode, suffix)
        with _timed_step("index_build.parse_to_markdown", parser_mode=settings.pdf_parser_mode, suffix=suffix):
            markdown, source_parser, source_type = parse_source_to_markdown(temp_path)
        
        logger.debug("[ingestion][index_build] step=build_index markdown_chars=%d", len(markdown))
        return _build_index_response_from_markdown(
            markdown=markdown,
            source_file_path=temp_path,
            source_parser=source_parser,
            source_type=source_type,
            filename=str(file.filename or ""),
            route_name="v1/index/build",
            request_started=request_started,
        )


@app.post("/v1/index/build-from-markdown", response_model=IndexBuildResponse)
def index_build_from_markdown(request: IndexBuildFromMarkdownRequest) -> IndexBuildResponse:
    request_started = time.perf_counter()
    logger.info("[ingestion][index_build_from_markdown] request received source=%s markdown_chars=%d", request.source_file_path, len(request.markdown))
    logger.debug("[ingestion][index_build_from_markdown] step=validate_input parser=%s type=%s", request.source_parser, request.source_type)
    
    if not request.markdown.strip():
        logger.error("[ingestion][index_build_from_markdown] empty markdown source=%s", request.source_file_path)
        raise HTTPException(status_code=400, detail="Markdown content is empty.")
    logger.debug("[ingestion][index_build_from_markdown] step=validate_ok markdown_chars=%d", len(request.markdown))
    
    return _build_index_response_from_markdown(
        markdown=request.markdown,
        source_file_path=Path(request.source_file_path),
        source_parser=request.source_parser,
        source_type=request.source_type,
        filename=Path(request.source_file_path).name,
        route_name="v1/index/build-from-markdown",
        request_started=request_started,
    )


@app.post("/v1/index/upsert", response_model=IndexUpsertResponse)
def index_upsert(request: IndexUpsertRequest) -> IndexUpsertResponse:
    request_started = time.perf_counter()
    logger.info("[ingestion][index_upsert] request received document_id=%s child_rows=%d", request.document_id, len(request.child_rows))
    
    logger.debug("[ingestion][index_upsert] step=validate_config retrieval_url=%s", bool(settings.retrieval_service_url))
    if not settings.retrieval_service_url:
        logger.error("[ingestion][index_upsert] retrieval service url not configured")
        raise HTTPException(status_code=500, detail="RETRIEVAL_SERVICE_URL is not configured in ingestion_service.")
    logger.debug("[ingestion][index_upsert] step=validate_ok retrieval_url=%s", settings.retrieval_service_url)
    
    if not request.child_rows:
        logger.info("[ingestion][index_upsert] no child rows to upsert document_id=%s", request.document_id)
        return IndexUpsertResponse(indexed_chunks=0)

    logger.debug("[ingestion][index_upsert] step=prepare_chunks child_rows=%d", len(request.child_rows))
    chunks: list[dict[str, Any]] = []
    for row in request.child_rows:
        page_start = row.page_start if row.page_start is not None else row.source_page
        source_metadata = dict(row.source_metadata or {})
        source_metadata.update(
            {
                "document_id": row.document_id,
                "parent_id": row.parent_chunk_id,
                "parent_chunk_id": row.parent_chunk_id,
                "logical_parent_id": row.parent_id,
                "chunk_id": row.chunk_id,
                "child_chunk_id": row.chunk_id,
                "child_index": row.child_index,
                "child_type": row.child_type,
                "index_type": row.index_type,
                "page_start": page_start,
                "page_end": row.page_end,
                "heading_path": list(row.heading_path),
                "section_title": row.section_title,
            }
        )
        chunks.append(
            {
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "content": row.embedding_text,
                "page": page_start,
                "metadata": {
                    "document_id": row.document_id,
                    "parent_id": row.parent_chunk_id,
                    "parent_chunk_id": row.parent_chunk_id,
                    "chunk_id": row.chunk_id,
                    "child_chunk_id": row.chunk_id,
                    "logical_parent_id": row.parent_id,
                    "section_id": row.section_id,
                    "index_type": row.index_type,
                    "heading_path": list(row.heading_path),
                    "section_title": row.section_title,
                    "page_start": page_start,
                    "page_end": row.page_end,
                    "source_page": page_start,
                    "source_metadata": source_metadata,
                    "child_type": row.child_type,
                    "child_index": row.child_index,
                    "child_text": row.child_text,
                    "embedding_text": row.embedding_text,
                    "token_count": row.token_count,
                },
            }
        )
    logger.debug("[ingestion][index_upsert] step=chunks_prepared total_chunks=%d", len(chunks))

    logger.info(
        "[ingestion][indexing] upsert_child_chunks document_id=%s embedding_count=%d index_type=%s",
        request.document_id,
        len(chunks),
        INDEX_TYPE,
    )

    logger.debug("[ingestion][index_upsert] step=connect_retrieval_service url=%s timeout=%s", settings.retrieval_service_url, settings.retrieval_timeout_seconds)
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
            logger.debug("[ingestion][index_upsert] step=post_batch document_id=%s batch_start=%d batch_size=%d", request.document_id, start, len(batch))
            
            try:
                resp = client.post(f"{settings.retrieval_service_url}/v1/index/chunks", json=payload)
                if resp.status_code >= 400:
                    logger.error("[ingestion][index_upsert] retrieval service error status=%d document_id=%s", resp.status_code, request.document_id)
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
            except httpx.RequestError as exc:
                logger.exception("[ingestion][index_upsert] retrieval service request failed document_id=%s batch_start=%d", request.document_id, start)
                raise HTTPException(status_code=502, detail=f"Failed to connect to retrieval service: {exc}") from exc
    
    logger.info(
        "[ingestion][timing] route=/v1/index/upsert status=ok total_elapsed_ms=%.2f document_id=%s indexed_chunks=%d",
        _ms(request_started),
        request.document_id,
        indexed_total,
    )
    return IndexUpsertResponse(indexed_chunks=indexed_total)
