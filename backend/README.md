# RAG App Backend (FastAPI)

## Features

- Document CRUD and upload API
- Async incremental indexing pipeline per document (BackgroundTasks)
- Dual PDF parser mode via env (`legacy` or `marker`)
- Structured chunk metadata schema for hybrid search:
	- `source_info` (file, page, doc_type)
	- `context` (h2/h3)
	- `search_optimization` (entities, organizations, dates, document_codes)
	- `admin_tags` (security_level, department)
- HyQ enrichment at indexing time (`summary` + hypothetical `questions`) is provided by `ollama_service`
- HyQ LLM batching for metadata generation
- Overlapped ingest pipeline: metadata/HyQ batch `N+1` can run while embedding batch `N` is in-flight
- Metadata cache in SQLite (`chunk_metadata_cache`) keyed by `document_id + file_hash + chunk_fingerprint`
- Parent-child retrieval: child vectors are indexed, parent chunk text is returned to LLM
- Hybrid retrieval (vector + keyword) with reciprocal-rank-fusion
- Qdrant as vector store backend (local mode by default, remote mode optional)
- Chat query endpoint with persistent chat memory
- Remote Indexing: Metadata, summary, and HyQ generation are handled by `ollama_service`.
- Remote Ingestion: Document parse/split is handled by `ingestion_service`.
- Ollama Shield: Backend calls `ollama_service` instead of exposing raw Ollama directly.

## Auth Model

- Document APIs (`/api/documents/*`) require admin JWT Bearer token.
- Chat APIs (`/api/chat/*`) are public in current implementation.

## Run

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```

Run server:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run Ollama shield service:

```bash
cd ollama_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

Run ingestion service (separate process):

```bash
cd ingestion_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Set backend client config in `backend/.env` before starting backend:

```env
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default
INGESTION_SERVICE_URL=http://localhost:8100
```

## Environment

### Required
- `OLLAMA_BASE_URL`: Required URL to the `ollama_service`, example `http://localhost:8200`.
- `OLLAMA_API_KEY`: Required API key for X-API-KEY authentication against `ollama_service`.
- `OLLAMA_CHAT_MODEL`: Can be `default`; `ollama_service` overrides the real model via `CHAT_MODEL`.
- `OLLAMA_EMBEDDING_MODEL`: Can be `default`; `ollama_service` overrides the real model via `EMBEDDING_MODEL`.

### Ollama Shield Service (`ollama_service/.env`)
- `SHIELD_API_KEY`: Must match backend `OLLAMA_API_KEY`.
- `UPSTREAM_OLLAMA_BASE_URL`: Real Ollama server behind the shield, example `http://127.0.0.1:11434`.
- `CHAT_MODEL`: Real chat model served by Ollama.
- `INDEXING_MODEL`: Real model for metadata/HyQ indexing endpoints.
- `EMBEDDING_MODEL`: Real embedding model served by Ollama.
- `RATE_LIMIT_PER_MINUTE`: Per-key, per-route rate limit.

If `OLLAMA_BASE_URL` or `OLLAMA_API_KEY` is missing, or `ollama_service` is unavailable, backend LLM/chat/embedding/indexing requests fail and return an error.

### Core Settings
- `APP_NAME`: Backend application name.
- `SECRET_KEY`: JWT signing key.
- `ADMIN_DEFAULT_USERNAME` / `ADMIN_DEFAULT_PASSWORD`: Initial admin credentials.

### Vector Storage (Qdrant)
- `QDRANT_URL`: Empty for local embedded mode, or remote Qdrant URL.
- `QDRANT_API_KEY`: Required for remote Qdrant.
- `QDRANT_COLLECTION_NAME`: Default collection name.

### Ingestion & Retrieval
- `PDF_PARSER_MODE`: `legacy` (PyMuPDF) or `marker`.
- `INGESTION_SERVICE_URL`: Required URL for the parse/split service (example: `http://localhost:8100`).
- `INGESTION_TIMEOUT_SECONDS`: HTTP timeout when backend calls ingestion service.
- `CHUNK_SIZE`: Target chunk length (default: 1000).
- `CHUNK_OVERLAP`: Overlap between chunks (default: 150).
- `RERANKER_ENABLED`: Enable BGE Reranker (default: true).
- `QUERY_REWRITE_ENABLED`: Enable HyDE-like query rewriting.

If `QDRANT_URL` is empty, backend uses local embedded Qdrant persisted at:

- `backend/storage/indexes/global_qdrant/`

If `PDF_PARSER_MODE=marker` and `INGESTION_SERVICE_URL` is set, install Marker in the ingestion service venv:

```bash
cd ingestion_service
pip install marker-pdf
```

If `INGESTION_SERVICE_URL` is empty or the service is unavailable, document parse/split requests fail and backend returns an error.

When using remote ingestion, parsed markdown is cached by backend at:

- `backend/storage/parsed_markdown/<document_id>.md`

Marker/debug markdown logs are written by the process that performs parsing. With remote ingestion this is usually:

- `ingestion_service/storage/markdown_logs/marker/<uploaded_file_stem>.md`

This file is regenerated on each embed so you can quickly inspect parsing output.

## Async Indexing Behavior

- `POST /api/documents/{document_id}/embed` now queues background indexing and returns `202 Accepted` immediately.
- `POST /api/documents/{document_id}/process` runs parse-if-needed + indexing in one background job and is the preferred UI path.
- `POST /api/documents/reindex` now queues pending documents in background instead of blocking request time.
- Document `status` transitions: `uploaded` -> `indexing` -> `embedded` (or `index_failed` when background task fails).
- Qdrant upsert is executed with async write mode (`wait=false`) for faster ingestion throughput.
- Incremental vector updates are scoped per `document_id` (no collection recreate in upload/embed flow).
- Immediate response semantics:
	- queued embed: `chunks_created=0`, `indexed_chunks=0`
	- unchanged file hash and already embedded: returns cached counts without re-indexing

## Metadata Optimization

- HyQ LLM calls are now batched: multiple chunks are grouped into one inference call to reduce Ollama I/O overhead.
- Metadata is cached per `document_id + file_hash + chunk_fingerprint` in SQLite table `chunk_metadata_cache`.
- Re-indexing the same content reuses cached metadata and skips repeated LLM generation.

## Operational Notes

- BackgroundTasks are in-process jobs. If backend restarts during indexing, in-flight jobs may stop.
- `POST /api/documents/reindex` is used to queue pending documents again after restart/failure.
- Increasing `METADATA_LLM_BATCH_SIZE` and `VECTOR_BATCH_SIZE` can improve throughput but may increase RAM/VRAM usage.

## Optimizations Applied for Indexing

To resolve the bottleneck during metadata extraction and chunk indexing (which was spending time on LLM I/O and error-prone JSON parsing):
1. **Delegation of Workload (Regex Fallback)**: The indexing process now uses optimized Regex for entity, organization, dates, and document code metadata extraction (search_optimization), bypassing the LLM for these fields entirely.
2. **Structured Output Enforcement (Pydantic)**: The LLM is exclusively used for generating summaries and hypothetical questions (HyQ). It now utilizes ChatOllama.with_structured_output alongside Pydantic models (HyQResultModel), eliminating JSON parsing loops.
3. **Optimized Prompts:** Prompt logic has been minimized to reduce the Time-To-First-Token (TTFT).
