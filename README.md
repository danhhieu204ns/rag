# RAG App

Ứng dụng RAG dùng FastAPI, React, Ollama và Qdrant để upload tài liệu, tạo index vector theo mô hình parent-child, rồi chat hỏi đáp có nguồn tham chiếu.

## Thành phần

```text
rag/
├─ backend/             # API gateway, auth, document CRUD, chat, SQLite
├─ ingestion_service/   # Parse PDF/TXT/MD và split markdown thành chunks
├─ ollama_service/      # Shield/proxy trước Ollama, bảo vệ bằng x-api-key
├─ retrieval_service/   # Qdrant retrieval/indexing service
├─ frontend/            # React + Vite UI
└─ infra/               # Docker Compose cho môi trường local
```

## Kiến trúc tổng thể

```text
Frontend
   |
   v
Backend / Orchestrator
   |
   |---------------------------|----------------------|
   v                           v                      v
Ingestion Service        Retrieval Service       Ollama Service
   |                           |                      |
   v                           v                      v
Parse / split docs       Qdrant vector DB        Ollama runtime
```

Ranh giới trách nhiệm hiện tại:

| Thành phần | Trách nhiệm |
| --- | --- |
| `backend` | API chính cho frontend, auth, document CRUD, chat history, điều phối RAG, prompt, generation và citation. |
| `ingestion_service` | Parse file `.pdf`, `.txt`, `.md` thành markdown và split thành chunks. Đây là phần xử lý tài liệu nặng, tách khỏi backend để dễ scale. |
| `retrieval_service` | Sở hữu truy cập Qdrant, nhận chunks để upsert, embedding query, vector search và trả context cho backend. |
| `ollama_service` | Gateway bảo vệ Ollama thật, ép model theo cấu hình, rate limit, phục vụ chat, embedding và metadata/HyQ. |
| `qdrant` | Vector database nội bộ, không gọi trực tiếp từ frontend. |

Frontend chỉ nên gọi `backend` qua `http://localhost:8000` hoặc `/api`. Các service còn lại là nội bộ.

Luồng chính:

1. Frontend gọi backend qua `/api`.
2. Backend lưu metadata, file upload và lịch sử chat trong SQLite.
3. Backend gọi `ingestion_service` để parse/split tài liệu.
4. Backend gọi `ollama_service` để sinh metadata/HyQ và câu trả lời.
5. Backend gọi `retrieval_service` để ghi/truy vấn vector trong Qdrant. Khi bật `RETRIEVAL_SERVICE_URL`, embedding chunk/query do `retrieval_service` gọi `ollama_service`; nếu để rỗng, backend dùng Qdrant và embedding trực tiếp trong process.

Luồng upload và index tài liệu:

```text
Frontend
   ↓
Backend nhận upload và lưu metadata/file
   ↓
Backend gọi Ingestion Service để parse + split
   ↓
Backend gọi Ollama Service để enrich metadata/HyQ
   ↓
Backend gọi Retrieval Service để embed chunks và upsert vào Qdrant
   ↓
Frontend xem trạng thái: uploaded / indexing / embedded / index_failed
```

Luồng hỏi đáp:

```text
Frontend
   ↓
Backend nhận câu hỏi
   ↓
Backend gọi Retrieval Service lấy top_k context qua hybrid search
   ↓
Retrieval Service gọi Ollama Service để embedding query và search Qdrant
   ↓
Backend build prompt, gọi Ollama Service để generate answer
   ↓
Backend lưu chat history và trả answer + sources về frontend
```

Ghi chú thiết kế: repo đang đi theo hướng `retrieval_service` sở hữu Vector DB. Backend có fallback truy cập Qdrant trực tiếp khi `RETRIEVAL_SERVICE_URL` rỗng để giữ tương thích local/dev, nhưng cấu hình khuyến nghị là đi qua `retrieval_service`.

## Yêu cầu

- Docker Desktop + Docker Compose nếu chạy bằng compose.
- Hoặc Python 3.11+, Node.js 20+, npm, Ollama nếu chạy thủ công.
- Model Ollama mặc định trong `.env.example`:
  - `qwen3:30b-a3b-instruct-2507-q4_K_M`
  - `qwen3:4b-instruct-2507-q4_K_M`
  - `qwen3-embedding:0.6b`

Pull model khi dùng Ollama local:

```powershell
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3-embedding:0.6b
```

## Cài đặt nhanh bằng Docker Compose

Từ thư mục gốc:

```powershell
Copy-Item .env.example .env
```

Sửa các giá trị tối thiểu trong `.env`:

```env
SECRET_KEY=change-this-secret-key-in-production
SHIELD_API_KEY=change-this-key
OLLAMA_API_KEY=change-this-key
ADMIN_DEFAULT_USERNAME=admin
ADMIN_DEFAULT_PASSWORD=Admin@123
```

Khởi động toàn bộ hệ thống:

```powershell
docker compose -f infra/docker-compose.yml up -d --build
```

Địa chỉ sau khi chạy:

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`

Kiểm tra trạng thái:

```powershell
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml logs api-gateway
```

Tắt hệ thống:

```powershell
docker compose -f infra/docker-compose.yml down
```

## Chạy local compose (không cần Docker)

Script local-compose cung cấp các lệnh tương đương `up`, `down`, `logs` cho từng service.

Chạy toàn bộ:

```powershell
./scripts/local-compose.sh up
```

Dừng toàn bộ:

```powershell
./scripts/local-compose.sh down
```

Xem log từng service:

```powershell
./scripts/local-compose.sh logs api-gateway
./scripts/local-compose.sh logs retrieval-service
./scripts/local-compose.sh logs ollama-service
```

## Cài đặt thủ công cho development

Tạo env cho từng service:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item ingestion_service/.env.example ingestion_service/.env
Copy-Item ollama_service/.env.example ollama_service/.env
Copy-Item retrieval_service/.env.example retrieval_service/.env
Copy-Item frontend/.env.example frontend/.env
```

Các cấu hình local quan trọng:

```env
# backend/.env
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default
INGESTION_SERVICE_URL=http://localhost:8100
RETRIEVAL_SERVICE_URL=http://localhost:8300
QDRANT_URL=http://localhost:6333
```

```env
# ollama_service/.env
SHIELD_API_KEY=change-this-key
UPSTREAM_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

```env
# retrieval_service/.env
OLLAMA_SERVICE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
QDRANT_URL=http://localhost:6333
RETRIEVAL_DATABASE_PATH=backend/storage/app.db
```

### Chạy riêng từng service

Chạy theo thứ tự dưới đây để các service phụ thuộc không lỗi kết nối lúc khởi động. Mỗi block nên chạy ở một terminal riêng.

1. Ollama runtime gốc

```powershell
ollama serve
```

Nếu Ollama Desktop đã chạy sẵn trên `http://127.0.0.1:11434`, có thể bỏ qua lệnh này. Kiểm tra model:

```powershell
ollama list
```

2. Qdrant vector database

```powershell
docker run --rm -p 6333:6333 -v ${PWD}/.qdrant:/qdrant/storage qdrant/qdrant:v1.14.0
```

Health check:

```powershell
Invoke-RestMethod http://localhost:6333/healthz
```

3. Ollama Service

Service này là gateway trước Ollama, backend và retrieval service đều gọi qua đây.

```powershell
cd ollama_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8200/ready
```

4. Ingestion Service

Service này parse/split tài liệu. Backend gọi qua `INGESTION_SERVICE_URL`.

```powershell
cd ingestion_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8100/ready
```

5. Retrieval Service

Service này sở hữu Qdrant, tự gọi `ollama_service` để embedding query/chunks.

```powershell
cd retrieval_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8300/ready
```

6. Backend / Orchestrator

Backend là API duy nhất frontend gọi. Trước khi chạy, kiểm tra `backend/.env` có các URL nội bộ sau:

```env
OLLAMA_BASE_URL=http://localhost:8200
INGESTION_SERVICE_URL=http://localhost:8100
RETRIEVAL_SERVICE_URL=http://localhost:8300
QDRANT_URL=http://localhost:6333
```

Chạy backend:

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/ready
```

7. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend dev mặc định chạy tại `http://localhost:5173`. Nếu dùng Vite dev server, `frontend/.env` nên trỏ về:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

Tóm tắt port local:

| Service | Port | URL |
| --- | ---: | --- |
| Frontend dev | 5173 | `http://localhost:5173` |
| Backend / Orchestrator | 8000 | `http://localhost:8000` |
| Ingestion Service | 8100 | `http://localhost:8100` |
| Ollama Service | 8200 | `http://localhost:8200` |
| Retrieval Service | 8300 | `http://localhost:8300` |
| Qdrant | 6333 | `http://localhost:6333` |
| Ollama runtime | 11434 | `http://localhost:11434` |


## API chính

- Auth: `POST /api/auth/login`, `GET /api/auth/me`
- Users: `GET/POST/PUT/DELETE /api/users`
- Documents:
  - `GET /api/documents`
  - `POST /api/documents/upload`
  - `POST /api/documents/{id}/parse`
  - `POST /api/documents/{id}/embed`
  - `POST /api/documents/{id}/process`
  - `DELETE /api/documents/{id}`
- Chat:
  - `GET /api/chat/sessions`
  - `POST /api/chat/sessions`
  - `GET /api/chat/sessions/{id}/messages`
  - `POST /api/chat/query`

Document và user APIs yêu cầu JWT admin. Chat API hiện không bắt buộc đăng nhập.

## Biến môi trường đáng chú ý

- `SECRET_KEY`: khóa ký JWT.
- `ADMIN_DEFAULT_USERNAME`, `ADMIN_DEFAULT_PASSWORD`: tài khoản admin được tạo khi DB trống.
- `OLLAMA_BASE_URL`: URL của `ollama_service` đối với backend.
- `OLLAMA_API_KEY` và `SHIELD_API_KEY`: phải trùng nhau.
- `UPSTREAM_OLLAMA_BASE_URL`: Ollama thật phía sau shield.
- `INGESTION_SERVICE_URL`: URL parse/split service; để rỗng nếu muốn backend parse local.
- `RETRIEVAL_SERVICE_URL`: URL retrieval service; để rỗng nếu muốn backend truy cập Qdrant trực tiếp.
- `QDRANT_URL`: URL Qdrant server; để rỗng thì dùng Qdrant local embedded path.
- `PDF_PARSER_MODE`: `legacy` hoặc `marker`.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: cấu hình split tài liệu.
- `QUERY_REWRITE_ENABLED`, `QUERY_REWRITE_MIN_TERMS`, `QUERY_REWRITE_MAX_TERMS`: cấu hình rewrite query.
- `HYBRID_PROBE_MULTIPLIER`, `HYBRID_RRF_K`, `HYBRID_VECTOR_RRF_WEIGHT`, `HYBRID_KEYWORD_RRF_WEIGHT`: cấu hình hybrid retrieval.
- `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_CANDIDATE_POOL`: cấu hình reranker.

## Dữ liệu runtime

Các thư mục sau được sinh tự động và đã được ignore:

- `backend/storage/`: SQLite DB, uploads, parsed markdown, local Qdrant, logs.
- `ingestion_service/storage/`: cache/log parser.
- `retrieval_service/storage/`: storage runtime của retrieval service.
- `frontend/node_modules/`, `frontend/dist/`.
- `.venv/`, `__pycache__/`.

Không xóa `backend/storage` nếu cần giữ tài liệu, index và lịch sử chat.

## Kiểm tra nhanh

```powershell
python -m compileall backend ingestion_service ollama_service retrieval_service
cd frontend
npm run build
```

Nếu dùng sandbox hoặc môi trường hạn chế quyền, `npm run build` có thể cần chạy ngoài sandbox vì Vite/esbuild phải spawn binary native.
