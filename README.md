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

Luồng chính:

1. Frontend gọi backend qua `/api`.
2. Backend lưu metadata, file upload và lịch sử chat trong SQLite.
3. Backend gọi `ingestion_service` để parse/split tài liệu.
4. Backend gọi `ollama_service` để sinh metadata/HyQ, embedding và câu trả lời.
5. Backend gọi `retrieval_service` để ghi/truy vấn vector trong Qdrant. Nếu không cấu hình `RETRIEVAL_SERVICE_URL`, backend dùng Qdrant trực tiếp trong process.

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
RETRIEVAL_SERVICE_URL=http://localhost:8030
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

Chạy Qdrant nếu dùng retrieval service:

```powershell
docker run --rm -p 6333:6333 -v ${PWD}/.qdrant:/qdrant/storage qdrant/qdrant:v1.14.0
```

Mở các terminal riêng:

```powershell
cd ollama_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

```powershell
cd ingestion_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

```powershell
cd retrieval_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8030
```

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

```powershell
cd frontend
npm install
npm run dev
```

Frontend dev mặc định chạy tại `http://localhost:5173`.

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
