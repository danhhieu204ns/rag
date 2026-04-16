# RAG App Backend (FastAPI)

## Features

- Document CRUD and upload API
- Embedding/chunking pipeline for uploaded documents
- Dual PDF parser mode via env (`legacy` or `marker`)
- Structured chunk metadata schema for hybrid search:
	- `source_info` (file, page, doc_type)
	- `context` (h2/h3)
	- `search_optimization` (entities, organizations, dates, document_codes)
	- `admin_tags` (security_level, department)
- HyQ enrichment at indexing time (`summary` + hypothetical `questions`)
- Parent-child retrieval: child vectors are indexed, parent chunk text is returned to LLM
- Hybrid retrieval (vector + keyword) with reciprocal-rank-fusion
- Qdrant as vector store backend (local mode by default, remote mode optional)
- Chat query endpoint with persistent chat memory

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
- `EMBEDDING_MODEL_NAME` (default: `BAAI/bge-m3`)
- `QDRANT_URL` (default: empty -> local embedded Qdrant)
- `QDRANT_API_KEY` (default: empty)
- `QDRANT_COLLECTION_NAME` (default: `global_child_chunks`)
- `PDF_PARSER_MODE` (default: `legacy`, supports: `legacy`, `marker`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVER_K`
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE`
- `OLLAMA_NUM_THREAD` (default: `8`)
- `HYQ_ENABLED` (default: `true`)
- `HYQ_USE_LLM` (default: `false`)
- `HYQ_MODEL` (default: fallback to `METADATA_MODEL`, then `LLM_MODEL`)
- `METADATA_USE_LLM` (default: fallback to `HYQ_USE_LLM`)
- `METADATA_MODEL` (default: fallback to `HYQ_MODEL`, then `LLM_MODEL`)
- `METADATA_OLLAMA_NUM_THREAD` (default: fallback to `OLLAMA_NUM_THREAD`)
- `METADATA_OLLAMA_NUM_PREDICT` (default: `256`)
- `HYQ_SUMMARY_WORDS` (default: `50`)
- `HYQ_QUESTIONS_PER_CHUNK` (default: `3`)
- `HYBRID_VECTOR_RRF_WEIGHT` (default: `1.0`)
- `HYBRID_KEYWORD_RRF_WEIGHT` (default: `1.2`)
- `HYBRID_RRF_K` (default: `60`)
- `HYBRID_PROBE_MULTIPLIER` (default: `4`)

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

Recommended setup for lower VRAM/CPU load on metadata tasks (keyword extraction, summary, hypothetical questions):

```env
HYQ_USE_LLM=true
HYQ_MODEL=llama3.2:3b
METADATA_USE_LLM=true
METADATA_MODEL=llama3.2:3b
OLLAMA_NUM_THREAD=8
METADATA_OLLAMA_NUM_THREAD=8
METADATA_OLLAMA_NUM_PREDICT=192
```

Embedding model is served via Ollama using `langchain-ollama`.
