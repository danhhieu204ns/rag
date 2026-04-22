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
- HyQ enrichment at indexing time (`summary` + hypothetical `questions`)
- HyQ LLM batching for metadata generation
- Overlapped ingest pipeline: metadata/HyQ batch `N+1` can run while embedding batch `N` is in-flight
- Metadata cache in SQLite (`chunk_metadata_cache`) keyed by `document_id + file_hash + chunk_fingerprint`
- Parent-child retrieval: child vectors are indexed, parent chunk text is returned to LLM
- Hybrid retrieval (vector + keyword) with reciprocal-rank-fusion
- Qdrant as vector store backend (local mode by default, remote mode optional)
- Chat query endpoint with persistent chat memory

## Auth Model

- Document APIs (`/api/documents/*`) require admin JWT Bearer token.
- Chat APIs (`/api/chat/*`) are public in current implementation.

## Run

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run server:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Environment

Optional tuning:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `EMBEDDING_BACKEND` (default: `ollama`, supports: `ollama`, `sentence-transformers`)
- `EMBEDDING_MODEL_NAME` (default: `BAAI/bge-m3`)
- `EMBEDDING_MAX_LENGTH` (default: `512`, only for `sentence-transformers` backend)
- `EMBEDDING_USE_FP16` (default: `true`, only for `sentence-transformers` backend)
- `EMBEDDING_BATCH_SIZE` (default: `64`, only for `sentence-transformers` backend)
- `EMBEDDING_DEVICE` (default: `auto`, values: `auto`, `cuda`, `cpu`)
- `EMBEDDING_LOCAL_FILES_ONLY` (default: `false`, set `true` after first successful download to skip network fetch)
- `QDRANT_URL` (default: empty -> local embedded Qdrant)
- `QDRANT_API_KEY` (default: empty)
- `QDRANT_COLLECTION_NAME` (default: `global_child_chunks`)
- `PDF_PARSER_MODE` (default: `legacy`, supports: `legacy`, `marker`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVER_K`
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE`
- `LLM_NUM_CTX` (default: `2048`, giảm KV cache cho LLM chính)
- `LLM_KEEP_ALIVE` (default: `10m`, thời gian giữ model chat trên VRAM)
- `OLLAMA_NUM_THREAD` (default: `8`)
- `HYQ_ENABLED` (default: `true`)
- `HYQ_USE_LLM` (default: `false`)
- `HYQ_MODEL` (default: fallback to `METADATA_MODEL`, then `LLM_MODEL`)
- `METADATA_USE_LLM` (default: fallback to `HYQ_USE_LLM`)
- `METADATA_MODEL` (default: fallback to `HYQ_MODEL`, then `LLM_MODEL`)
- `METADATA_SUMMARY_USE_HIGH_ACCURACY` (default: `false`, bật thì dùng model riêng cho phần summary)
- `METADATA_SUMMARY_MODEL` (default: empty, ví dụ `llama3.1:8b` nếu cần summary chất lượng cao)
- `METADATA_SUMMARY_NUM_CTX` (default: `2048`)
- `METADATA_OLLAMA_NUM_THREAD` (default: fallback to `OLLAMA_NUM_THREAD`)
- `METADATA_OLLAMA_NUM_PREDICT` (default: `256`)
- `METADATA_NUM_CTX` (default: `1536`, giảm KV cache cho metadata/HyQ)
- `METADATA_KEEP_ALIVE` (default: `-1`, giữ model metadata resident)
- `EMBEDDING_KEEP_ALIVE` (default: `-1`, giữ model embedding resident)
- `METADATA_LLM_BATCH_SIZE` (default: `8`, số chunk gọi LLM mỗi lượt)
- `METADATA_LLM_BATCH_MAX_CHARS` (default: `12000`, ngưỡng ký tự prompt cho mỗi batch)
- `VECTOR_BATCH_SIZE` (default: `64`, can increase to `128` if RAM allows)
- `HYQ_SUMMARY_WORDS` (default: `50`)
- `HYQ_QUESTIONS_PER_CHUNK` (default: `3`)
- `HYBRID_VECTOR_RRF_WEIGHT` (default: `1.0`)
- `HYBRID_KEYWORD_RRF_WEIGHT` (default: `1.2`)
- `HYBRID_RRF_K` (default: `60`)
- `HYBRID_PROBE_MULTIPLIER` (default: `4`)
- `MODEL_WARMUP_ON_STARTUP` (default: `false`, warmup model sau khi backend khởi động)
- `MODEL_WARMUP_METADATA` (default: `true`)
- `MODEL_WARMUP_EMBEDDING` (default: `true`)
- `MODEL_WARMUP_CHAT` (default: `false`)

If `HYQ_USE_LLM=true` or `METADATA_USE_LLM=true`, ensure the selected model is available in Ollama.

If `QDRANT_URL` is empty, backend uses local embedded Qdrant persisted at:

- `backend/storage/indexes/global_qdrant/`

If `PDF_PARSER_MODE=marker`, install Marker in backend venv:

```bash
pip install marker-pdf
```

When using `PDF_PARSER_MODE=marker`, parsed markdown is also logged to:

- `backend/storage/markdown_logs/marker/<uploaded_file_stem>.md`

This file is regenerated on each embed so you can quickly inspect parsing output.

Before running queries, ensure Ollama is available for chat generation:

```bash
ollama pull llama3.1:8b
ollama pull bge-m3
```

Optional if HyQ LLM generation is enabled:

```bash
ollama pull llama3.2:3b
```

For 12GB VRAM, prioritize quantized tags (`q4` or `q5`) for all chat/metadata LLM models when your registry exposes them.

Set Ollama daemon memory knobs before starting `ollama serve`:

```bash
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
ollama serve
```

On Windows PowerShell:

```powershell
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_NUM_PARALLEL="2"
$env:OLLAMA_MAX_LOADED_MODELS="3"
$env:OLLAMA_KEEP_ALIVE="-1"
ollama serve
```

## Deployment Orchestration (12GB VRAM)

Recommended tiered pipeline to reduce model-switch overhead:

- `llama3.2:3b` as persistent model for metadata + HyQ.
- `bge-m3` as persistent embedding model.
- `llama3.1:8b` only for high-accuracy summary when needed.

Profile A (default, fastest switching / lowest VRAM pressure):

```env
HYQ_MODEL=llama3.2:3b
METADATA_MODEL=llama3.2:3b
METADATA_SUMMARY_USE_HIGH_ACCURACY=false
METADATA_KEEP_ALIVE=-1
EMBEDDING_KEEP_ALIVE=-1
LLM_KEEP_ALIVE=10m
```

Profile B (precision summary, accepts extra model switch):

```env
HYQ_MODEL=llama3.2:3b
METADATA_MODEL=llama3.2:3b
METADATA_SUMMARY_USE_HIGH_ACCURACY=true
METADATA_SUMMARY_MODEL=llama3.1:8b
```

When `METADATA_SUMMARY_USE_HIGH_ACCURACY=true`, only summary is refined by `METADATA_SUMMARY_MODEL`; keyword extraction and HyQ question generation stay on the smaller metadata model.

Recommended setup for lower VRAM/CPU load on metadata tasks (keyword extraction, summary, hypothetical questions):

```env
HYQ_USE_LLM=true
HYQ_MODEL=llama3.2:3b
METADATA_USE_LLM=true
METADATA_MODEL=llama3.2:3b
OLLAMA_NUM_THREAD=8
METADATA_OLLAMA_NUM_THREAD=8
METADATA_OLLAMA_NUM_PREDICT=192
LLM_NUM_CTX=2048
METADATA_NUM_CTX=1536
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_FLASH_ATTENTION=1
```

Embedding model is served via Ollama using `langchain-ollama`.

For BGE-M3 speed optimization with fixed length + fp16, use:

```env
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_MAX_LENGTH=512
EMBEDDING_USE_FP16=true
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DEVICE=auto
EMBEDDING_LOCAL_FILES_ONLY=false
```

Notes:

- `EMBEDDING_MAX_LENGTH` should align with chunk size to avoid extra padding compute.
- `EMBEDDING_USE_FP16=true` gives best speed on modern RTX GPUs and is ignored on CPU.

To reduce model load time in practice:

- First run (download once): keep `EMBEDDING_LOCAL_FILES_ONLY=false`.
- Next runs: switch `EMBEDDING_LOCAL_FILES_ONLY=true` to avoid network checks/download.
- For large indexing jobs, prefer running backend without `--reload` to avoid extra process restart overhead.

## Async Indexing Behavior

- `POST /api/documents/{document_id}/embed` now queues background indexing and returns `202 Accepted` immediately.
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
