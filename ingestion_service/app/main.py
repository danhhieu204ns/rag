from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
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


app = FastAPI(title=settings.app_name)


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
    suffix = Path(file.filename or "").suffix.lower()
    logger.info("[ingestion] /v1/parse filename=%s suffix=%s", file.filename, suffix)
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")

    with TemporaryDirectory(prefix="ingestion_upload_") as tmp_dir:
        temp_path = Path(tmp_dir) / (file.filename or "uploaded.bin")
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        temp_path.write_bytes(payload)

        try:
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
    return ParseResponse(
        markdown=markdown,
        source_parser=source_parser,
        source_type=source_type,
    )


@app.post("/v1/split", response_model=SplitResponse)
def split(request: SplitRequest) -> SplitResponse:
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
        markdown_path.write_text(request.markdown.strip(), encoding="utf-8")

        try:
            loaded = load_documents_from_parsed_markdown(
                markdown_path,
                source_file_path=Path(request.source_file_path),
                source_parser=request.source_parser,
                source_type=request.source_type,
            )
            chunks = split_source_documents(
                loaded,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("[ingestion] /v1/split failed source=%s", request.source_file_path)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("[ingestion] /v1/split done source=%s chunks=%d", request.source_file_path, len(chunks))
    return SplitResponse(
        chunks=[
            ChunkPayload(page_content=item.page_content, metadata=dict(item.metadata or {}))
            for item in chunks
            if str(item.page_content or "").strip()
        ]
    )
