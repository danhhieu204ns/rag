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
uvicorn app.main:app --reload
```

## Environment

Optional tuning:

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `EMBEDDING_MODEL_NAME` (default: `bge-m3`)
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVER_K`
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE`

Before running queries, ensure Ollama is available and models are pulled:

```bash
ollama pull bge-m3
ollama pull llama3.1:8b
```
