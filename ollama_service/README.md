# Ollama Service

Service shield/proxy đặt trước Ollama. Backend chỉ gọi service này, còn service sẽ:

- Bảo vệ API bằng header `x-api-key`.
- Ép model theo cấu hình server cho chat và embedding.
- Giới hạn số request/phút, độ dài prompt/input, số message và `num_predict`.
- Ép `stream=false` trên các route proxy tương thích và giữ `num_predict` không vượt quá giới hạn cấu hình.
- Cung cấp cả route proxy ổn định (`/v1/*`) lẫn native Ollama (`/api/*`) để dùng với `ChatOllama` và `OllamaEmbeddings`.
- Chỉ cung cấp inference endpoints; `POST /v1/indexing/batch` bị vô hiệu hóa và trả `410 Gone`.

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
OLLAMA_UPSTREAM_BASE_URL=http://127.0.0.1:11434
SHIELD_API_KEY=change-this-key

CHAT_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
EMBEDDING_MODEL=qwen3-embedding:0.6b

MAX_CHAT_CHARS=24000
MAX_EMBEDDING_CHARS=12000
MAX_MESSAGES=20
MAX_CHAT_NUM_PREDICT=2048
RATE_LIMIT_PER_MINUTE=30

# Timeout settings (seconds)
OLLAMA_CONNECT_TIMEOUT_SECONDS=10
OLLAMA_CHAT_TIMEOUT_SECONDS=240
OLLAMA_EMBEDDING_TIMEOUT_SECONDS=180

# Optional
OLLAMA_STORAGE_DIR=./storage
OLLAMA_SHIELD_APP_NAME=Ollama FastAPI Shield
```

Backend cần trỏ vào service này:

```env
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default
```

Shield đọc upstream theo thứ tự `UPSTREAM_OLLAMA_BASE_URL` -> `OLLAMA_UPSTREAM_BASE_URL` -> `OLLAMA_BASE_URL`. Nếu dùng chung `.env` ở repo root cho cả backend và shield, hãy đặt một trong hai biến upstream đầu tiên trỏ tới Ollama thật; `OLLAMA_BASE_URL` lúc đó nên là URL của shield để backend gọi.

`OLLAMA_CHAT_MODEL` và `OLLAMA_EMBEDDING_MODEL` ở backend có thể để `default` vì service sẽ override model bằng `CHAT_MODEL` và `EMBEDDING_MODEL`.

## API

Public:

- `GET /health` - kiểm tra trạng thái service
- `GET /ready` - kiểm tra kết nối upstream Ollama, trả về status code 503 nếu upstream degraded

Protected bằng `x-api-key`:

- `GET /v1/models`
- `GET /api/tags`
- `POST /v1/chat`
- `POST /v1/generate`
- `POST /v1/indexing/batch` trả `410 Gone`; LLM metadata indexing đã bị tắt.
- `POST /v1/embed`
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`

Ghi chú theo code hiện tại:

- `/v1/chat`, `/v1/generate`, `/api/chat` và `/api/generate` đều ép `stream=false` và cap `num_predict` theo `MAX_CHAT_NUM_PREDICT`.
- `/v1/embed` nhận `input` là chuỗi hoặc danh sách chuỗi; `/api/embed` nhận `input` hoặc `prompt` và forward thêm `truncate`, `options`, `keep_alive`, `dimensions` khi có.
- `/api/embeddings` là biến thể Ollama cũ hơn, chỉ nhận `prompt`.
- `/v1/models` trả thêm `configured_models` ngoài dữ liệu `/api/tags` upstream.

## Backend Contract

Backend hiện dùng:

- Chat/RAG: `POST /api/chat` qua `ChatOllama`.
- Embedding: `POST /api/embed` qua `OllamaEmbeddings`.
- Indexing only uses embeddings. Ingestion no longer calls LLM metadata indexing.

Service này giữ cả `/api/embed` và `/api/embeddings` để tương thích với các phiên bản Ollama client mới/cũ.
