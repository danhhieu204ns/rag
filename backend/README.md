# RAG App Backend (FastAPI)

## Features

- Document CRUD and upload API
- Embedding/chunking pipeline for uploaded documents
- Dual PDF parser mode via env (`legacy` or `marker`)
- Page-aware source metadata on saved chunks for better tracing
- Global FAISS index rebuild and retrieval
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
- `PDF_PARSER_MODE` (default: `legacy`, supports: `legacy`, `marker`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVER_K`
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE`

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

Embedding model is served via Ollama using `langchain-ollama`.
