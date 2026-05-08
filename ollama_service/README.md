# Ollama Service

Service shield/proxy đặt trước Ollama. Backend chỉ gọi service này, còn service sẽ:

- Bảo vệ API bằng header `x-api-key`.
- Ép model theo cấu hình server cho chat và embedding.
- Giới hạn số request/phút, độ dài prompt và `num_predict`.
- Cung cấp endpoint tương thích Ollama cho `ChatOllama` và `OllamaEmbeddings`.
- Chỉ cung cấp inference endpoints (`generate` / `embed` / `chat`), không chứa business logic RAG.

## Run

```bash
cd ollama_service
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

URL mặc định: `http://localhost:8200`

- Health: `GET /health`
- Swagger UI: `http://localhost:8200/docs`

## Environment

```env
UPSTREAM_OLLAMA_BASE_URL=http://127.0.0.1:11434
SHIELD_API_KEY=change-this-key

CHAT_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
INDEXING_MODEL=qwen3:4b-instruct-2507-q4_K_M
EMBEDDING_MODEL=qwen3-embedding:0.6b

MAX_CHAT_CHARS=24000
MAX_INDEXING_CHARS=12000
MAX_MESSAGES=20
MAX_CHAT_NUM_PREDICT=2048
MAX_INDEXING_NUM_PREDICT=768
INDEXING_CONCURRENCY=8
RATE_LIMIT_PER_MINUTE=30
```

Backend cần trỏ vào service này:

```env
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default
```

Nếu dùng chung `.env` ở repo root cho cả backend và shield, dùng `UPSTREAM_OLLAMA_BASE_URL` cho Ollama thật. `OLLAMA_BASE_URL` lúc đó nên là URL của shield để backend gọi.

`OLLAMA_CHAT_MODEL` và `OLLAMA_EMBEDDING_MODEL` ở backend có thể để `default` vì service sẽ override model bằng `CHAT_MODEL` và `EMBEDDING_MODEL`.

## API

Public:

- `GET /health`

Protected bằng `x-api-key`:

- `GET /v1/models`
- `GET /api/tags`
- `POST /v1/chat`
- `POST /v1/generate`
- `POST /v1/indexing/batch`
- `POST /v1/embed`
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`

## Backend Contract

Backend hiện dùng:

- Chat/RAG: `POST /api/chat` qua `ChatOllama`.
- Embedding: `POST /api/embed` qua `OllamaEmbeddings`.
- Metadata indexing: ingestion service gọi `POST /v1/indexing/batch`; shield tự build prompt/schema và fan-out sang Ollama.

Service này giữ cả `/api/embed` và `/api/embeddings` để tương thích với các phiên bản Ollama client mới/cũ.
