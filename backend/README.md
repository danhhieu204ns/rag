# RAG App Backend (FastAPI)

## Features

- Document CRUD and upload API
- Embedding/chunking pipeline for uploaded documents
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
- `EMBEDDING_MODEL_NAME` (default: `BAAI/bge-m3` or `bge-m3`)
- `EMBEDDING_DEVICE` (default: `cpu`, set `cuda` if GPU is available)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVER_K`
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE`

Before running queries, ensure Ollama is available for chat generation:

```bash
ollama pull llama3.1:8b
```

Embedding model `bge-m3` is loaded directly in Python via `sentence-transformers`.
The first run will download model weights from Hugging Face.
