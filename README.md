# RAG App

Ứng dụng RAG đầy đủ dùng **FastAPI** (backend), **React + Vite** (frontend), **Ollama** (LLM), và **Qdrant** (vector DB) để:
- Upload và index tài liệu (PDF/TXT/MD)
- Tạo chunks theo mô hình section-aware parent-child
- Truy vấn retrieval (vector + keyword + rerank) qua `retrieval_service`
- Chat hỏi đáp với trích dẫn nguồn và lịch sử persistent

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, Vite, TypeScript |
| **Backend** | FastAPI, SQLAlchemy, SQLite |
| **Document Processing** | PyMuPDF/Marker (PDF), LangChain |
| **Vector DB** | Qdrant, qdrant-client |
| **LLM** | Ollama, LangChain |
| **Search** | Retrieval Service (Vector + Keyword), RRF ranking, BGE Reranker |
| **Auth** | JWT, bcrypt |
| **Deployment** | Docker, Cloudflare Tunnel (optional) |

## Documentation

Nhảy nhanh tới service-specific docs:

| Service | Docs |
|---------|------|
| **Backend** (Orchestrator) | [backend/README.md](backend/README.md) |
| **Ingestion** (Document Processing) | [ingestion_service/README.md](ingestion_service/README.md) |
| **Ollama Shield** (LLM Proxy) | [ollama_service/README.md](ollama_service/README.md) |
| **Retrieval** (Vector Search) | [retrieval_service/README.md](retrieval_service/README.md) |
| **Frontend** (React UI) | [frontend/README.md](frontend/README.md) |

## Thành phần

```text
rag/
├─ backend/             # API gateway, auth, document CRUD, chat, SQLite
├─ ingestion_service/   # Parse PDF/TXT/MD và split markdown thành chunks
├─ ollama_service/      # Shield/proxy trước Ollama, bảo vệ bằng x-api-key
├─ retrieval_service/   # Qdrant retrieval/indexing service
└─ frontend/            # React + Vite UI
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
| `ingestion_service` | Indexing service: nhận tài liệu đầu vào, render Markdown, parse heading sections, tạo section-aware parent-child chunks và index child chunks qua retrieval API. |
| `retrieval_service` | Retrieval service: query embedding, vector search, filter, rerank và trả context cho backend. |
| `ollama_service` | Gateway bảo vệ Ollama thật, ép model theo cấu hình, rate limit, phục vụ chat và embedding. Endpoint metadata indexing cũ đã bị vô hiệu hóa. |
| `qdrant` | Vector database nội bộ, không gọi trực tiếp từ frontend. |

Frontend chỉ nên gọi `backend` qua `http://localhost:8000` hoặc `/api`. Các service còn lại là nội bộ.

Luồng chính:

1. Frontend gọi backend qua `/api`.
2. Backend lưu metadata, file upload và lịch sử chat trong SQLite.
3. Backend gọi `ingestion_service` để xử lý indexing phase (parse -> section parent-child chunk -> embed child -> index).
4. Backend gọi `retrieval_service` để truy vấn context.
5. Backend gọi `ollama_service` để generation câu trả lời.

Luồng upload và index tài liệu:

```text
Frontend
   ↓
Backend nhận upload và lưu metadata/file
   ↓
Backend gọi Ingestion Service để parse + section-aware chunk + embed child + index
   ↓
Frontend xem trạng thái: uploaded / indexing / embedded / index_failed
```

Luồng hỏi đáp:

```text
Frontend
   ↓
Backend nhận câu hỏi
   ↓
Backend gọi Retrieval Service lấy top_k context
   ↓
Retrieval Service gọi Ollama Service để embedding query và search Qdrant
   ↓
Backend build prompt, gọi Ollama Service để generate answer
   ↓
Backend lưu chat history và trả answer + sources về frontend
```

Ghi chú thiết kế: `retrieval_service` sở hữu Vector DB và backend luôn gọi retrieval qua service này.

## Yêu cầu

- Python 3.11+, Node.js 20+, npm, Ollama.
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

## Cài đặt nhanh local

Tạo env cho từng service:

```bash
cp backend/.env.example backend/.env
cp ingestion_service/.env.example ingestion_service/.env
cp ollama_service/.env.example ollama_service/.env
cp retrieval_service/.env.example retrieval_service/.env
cp frontend/.env.example frontend/.env
```

Trên Windows PowerShell:

```powershell
Copy-Item backend/.env.example backend/.env
Copy-Item ingestion_service/.env.example ingestion_service/.env
Copy-Item ollama_service/.env.example ollama_service/.env
Copy-Item retrieval_service/.env.example retrieval_service/.env
Copy-Item frontend/.env.example frontend/.env
```

Khởi động toàn bộ hệ thống trên Ubuntu:

```bash
./scripts/dev.sh
```

Khởi động toàn bộ hệ thống trên Windows PowerShell:

```powershell
.\scripts\dev.ps1
```

Dừng Ubuntu bằng `Ctrl+C` trong terminal đang chạy script. Trên Windows, script mở từng service trong cửa sổ PowerShell riêng; đóng các cửa sổ đó để dừng service.

Địa chỉ sau khi chạy:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- Backend docs: `http://localhost:8000/docs`

## Script chạy riêng từng service

Chạy theo thứ tự dưới đây để các service phụ thuộc không lỗi kết nối lúc khởi động.

Ubuntu:

```bash
./scripts/start-service.sh ollama
./scripts/start-service.sh ollama-service
./scripts/start-service.sh ingestion-service
./scripts/start-service.sh retrieval-service
./scripts/start-service.sh backend
./scripts/start-service.sh frontend
```

Windows PowerShell:

```powershell
.\scripts\start-service.ps1 ollama
.\scripts\start-service.ps1 ollama-service
.\scripts\start-service.ps1 ingestion-service
.\scripts\start-service.ps1 retrieval-service
.\scripts\start-service.ps1 backend
.\scripts\start-service.ps1 frontend
```

Mỗi service nên chạy ở một terminal riêng khi dùng `start-service`.

## Lệnh chạy trực tiếp từng service

Chạy theo thứ tự dưới đây, mỗi service ở một terminal riêng.

1. Ollama runtime

```bash
ollama serve
```

2. Qdrant embedded

Mặc định không cần chạy lệnh `qdrant` riêng. Để `QDRANT_URL` rỗng trong `backend/.env` và `retrieval_service/.env`, app sẽ dùng Qdrant embedded qua `qdrant-client` và lưu dữ liệu theo `QDRANT_PATH`.

```env
QDRANT_URL=
QDRANT_PATH=backend/storage/indexes/global_qdrant
```

Nếu muốn dùng Qdrant server ngoài, cần cài Qdrant binary trước rồi mới chạy các lệnh dưới đây và set `QDRANT_URL=http://localhost:6333`.

Ubuntu khi đã cài Qdrant binary:

```bash
mkdir -p .qdrant
QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__STORAGE__STORAGE_PATH="$(pwd)/.qdrant" qdrant
```

Windows PowerShell khi đã cài Qdrant binary:

```powershell
New-Item -ItemType Directory -Force .qdrant
$env:QDRANT__SERVICE__HTTP_PORT="6333"
$env:QDRANT__STORAGE__STORAGE_PATH="$PWD\.qdrant"
qdrant
```

3. Ollama Service

```bash
cd ollama_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ollama_service
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

Windows PowerShell:

```powershell
cd ollama_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8200

```

4. Ingestion Service

```bash
cd ingestion_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd ingestion_service
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Windows PowerShell:

```powershell
cd ingestion_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

5. Retrieval Service

```bash
cd retrieval_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd retrieval_service
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

Windows PowerShell:

```powershell
cd retrieval_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

6. Backend / Orchestrator

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. Frontend

```bash
cd frontend
npm install

cd frontend
npm run dev
```

## Script kiểm tra service

Ubuntu:

```bash
./scripts/check-services.sh
systemctl status ollama --no-pager
journalctl -u ollama -n 100 --no-pager
```

Windows PowerShell:

```powershell
.\scripts\check-services.ps1
Get-Service Ollama
```

Các script kiểm tra sẽ in process, port đang lắng nghe và health endpoint của từng service.

## Cấu hình local quan trọng

```env
# backend/.env
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default
INGESTION_SERVICE_URL=http://localhost:8100
RETRIEVAL_SERVICE_URL=http://localhost:8300
QDRANT_URL=
```

```env
# ollama_service/.env
SHIELD_API_KEY=change-this-key
UPSTREAM_OLLAMA_BASE_URL=http://127.0.0.1:11434
CHAT_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
ORCHESTRATOR_MODEL=qwen3:4b-instruct-2507-q4_K_M
EMBEDDING_MODEL=qwen3-embedding:0.6b
```

```env
# retrieval_service/.env
OLLAMA_SERVICE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
QDRANT_URL=
QDRANT_PATH=backend/storage/indexes/global_qdrant
RETRIEVAL_DATABASE_PATH=backend/storage/app.db
```

Frontend dev mặc định chạy tại `http://localhost:5173`. `frontend/.env` nên trỏ về:

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
| Qdrant embedded | N/A | `backend/storage/indexes/global_qdrant` |
| Ollama runtime | 11434 | `http://localhost:11434` |


## API chính

Frontend gọi backend qua `/api`. Xem [backend/README.md](backend/README.md#api-endpoints) để danh sách đầy đủ.

**Auth (Public):**
- `POST /api/auth/login` - Login với username/password
- `GET /api/auth/me` - Lấy thông tin user hiện tại

**Users (Admin required):**
- `GET /api/users` - Danh sách users
- `POST /api/users` - Tạo user mới
- `PUT /api/users/{user_id}` - Cập nhật user
- `DELETE /api/users/{user_id}` - Xóa user

**Documents (Admin required):**
- `GET /api/documents` - Danh sách documents
- `POST /api/documents/upload` - Upload file
- `GET /api/documents/{id}` - Chi tiết document
- `POST /api/documents/{id}/process` - Queue parse + embed (async, 202)
- `POST /api/documents/{id}/embed` - Queue embed only (async, 202)
- `GET /api/documents/{id}/chunks` - Danh sách chunks
- `PUT /api/documents/{id}` - Cập nhật metadata
- `DELETE /api/documents/{id}` - Xóa document
- `POST /api/documents/reindex` - Reindex pending documents

**Chat (Public):**
- `GET /api/chat/sessions` - Danh sách chat sessions
- `POST /api/chat/sessions` - Tạo session mới
- `GET /api/chat/sessions/{id}/messages` - Lịch sử messages
- `DELETE /api/chat/sessions/{id}` - Xóa session
- `POST /api/chat/query` - Gửi question, nhận answer + sources (streaming)

**Health:**
- `GET /api/health` hoặc `GET /health` - Health check
- `GET /api/ready` hoặc `GET /ready` - Readiness check

Xem [backend/README.md](backend/README.md#api-endpoints) để chi tiết đầy đủ.

## Biến môi trường chính

**Khuyến nghị**: Xem chi tiết môi trường từng service:
- [backend/README.md#environment-variables](backend/README.md#environment-variables)
- [ingestion_service/README.md#environment](ingestion_service/README.md#environment)
- [ollama_service/README.md#environment](ollama_service/README.md#environment)
- [retrieval_service/README.md](retrieval_service/README.md)

**Critical (Backend):**
- `OLLAMA_BASE_URL`: URL tới ollama_service (e.g., `http://localhost:8200`, **bắt buộc**)
- `OLLAMA_API_KEY`: API key xác thực (phải trùng với `SHIELD_API_KEY` của ollama_service)
- `SECRET_KEY`: JWT signing key (thay đổi trong production)
- `ADMIN_DEFAULT_USERNAME` / `ADMIN_DEFAULT_PASSWORD`: Admin account tạo lần đầu

**Optional pero Recommended:**
- `INGESTION_SERVICE_URL`: URL ingestion service (e.g., `http://localhost:8100`)
  - Để trống → backend parse local (chậm)
- `RETRIEVAL_SERVICE_URL`: URL retrieval service (e.g., `http://localhost:8300`)
  - Để trống → backend dùng embedded Qdrant

**Vector DB (Qdrant):**
- `QDRANT_URL`: Để trống → embedded mode, hoặc set remote URL
- `QDRANT_PATH`: Path cho embedded Qdrant (default: `backend/storage/indexes/global_qdrant`)

**Document Processing:**
- `PDF_PARSER_MODE`: `legacy` (PyMuPDF) hoặc `marker` (layout-aware)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Chunking parameters

**Legacy/Advanced** (không khuyến nghị thay đổi):
- `RETRIEVAL_SEARCH_CHILD_CHUNKS`: Search strategy
- `HYBRID_PROBE_MULTIPLIER`, `HYBRID_RRF_K`: Hybrid search tuning
- `RERANKER_ENABLED`, `RERANKER_MODEL`: Reranking parameters

## Dữ liệu runtime

Các thư mục sau được sinh tự động và đã được ignore:

- `backend/storage/`: SQLite DB, uploads, local Qdrant, logs.
- `ingestion_service/storage/`: cache/log parser.
- `retrieval_service/storage/`: storage runtime của retrieval service.
- `frontend/node_modules/`, `frontend/dist/`.
- `.venv/`, `__pycache__/`.

Không xóa `backend/storage` nếu cần giữ tài liệu, index và lịch sử chat.

## Troubleshooting

### Services không khởi động

**Backend yêu cầu `OLLAMA_BASE_URL` và `OLLAMA_API_KEY`:**
```bash
export OLLAMA_BASE_URL=http://localhost:8200
export OLLAMA_API_KEY=change-this-key
```

**SQLite database locked:**
```bash
rm backend/storage/app.db
# Restart backend
```

**Port đã bị sử dụng:**
```bash
# Kiểm tra process
lsof -i :8000  # Backend
lsof -i :8100  # Ingestion
lsof -i :8200  # Ollama Service
lsof -i :8300  # Retrieval Service
lsof -i :5173  # Frontend dev
```

### Upload/Indexing failed

- Kiểm tra `INGESTION_SERVICE_URL` có khả dụng: `curl http://localhost:8100/health`
- Xem logs: `tail -f backend/storage/logs/backend_service_*.log`
- Xem ingestion logs: `tail -f ingestion_service/storage/logs/*`

### Chat không trả kết quả

- Kiểm tra Retrieval Service: `curl http://localhost:8300/ready`
- Xem backend logs cho RAG timing traces
- Kiểm tra documents indexing status: `GET /api/documents`
- Xem Ollama service logs: `tail -f ollama_service/storage/logs/*`

### Vectors không index

- Kiểm tra Retrieval Service sẵn sàng: `curl http://localhost:8300/health`
- Kiểm tra Qdrant accessible: `curl http://localhost:6333/health` (nếu remote)
- Xem ingestion service logs

Xem chi tiết tại [backend/README.md#troubleshooting](backend/README.md#troubleshooting).

## Development

### File Structure

```
rag/
├── backend/               # Orchestrator (FastAPI)
│   ├── app/
│   │   ├── main.py       # FastAPI app, routers
│   │   ├── models.py     # SQLAlchemy models
│   │   ├── schemas.py    # Pydantic schemas
│   │   ├── db.py         # Database config
│   │   ├── api/          # Route handlers
│   │   ├── core/         # Settings, auth, logging
│   │   └── services/     # Business logic, clients
│   └── storage/          # Runtime data
├── ingestion_service/     # Document processing
│   └── app/              # FastAPI app
├── ollama_service/        # Ollama proxy/shield
│   └── app/              # FastAPI app
├── retrieval_service/     # Vector search
│   └── app/              # FastAPI app
├── frontend/             # React + Vite UI
│   ├── src/
│   ├── public/
│   └── index.html
├── scripts/              # Automation scripts
│   ├── dev.sh            # Start all services
│   ├── start-service.sh
│   └── check-services.sh
└── runctl/               # Cloudflare Tunnel setup
```

### Development Workflow

**Modify backend:**
```bash
cd backend
# Make changes in app/
# Backend will auto-reload with --reload flag
```

**Modify frontend:**
```bash
cd frontend
npm run dev  # Vite HMR enabled
```

**Modify service logic:**
```bash
cd ingestion_service  # or ollama_service, retrieval_service
# Make changes in app/
# Restart service to pick up changes
```

### Build & Deploy

**Python services:**
```bash
python -m compileall backend ingestion_service ollama_service retrieval_service
```

**Frontend:**
```bash
cd frontend
npm run build  # Outputs to dist/
```

**Docker** (if available):
```bash
docker build -f Dockerfile.backend -t rag-backend .
docker build -f Dockerfile.ingestion -t rag-ingestion .
docker build -f Dockerfile.frontend -t rag-frontend .
```

## Service Documentation

Mỗi service có README riêng với chi tiết:

- [backend/README.md](backend/README.md) - Orchestrator, API, database, RAG logic
- [ingestion_service/README.md](ingestion_service/README.md) - Document parsing, chunking
- [ollama_service/README.md](ollama_service/README.md) - Ollama proxy, rate limiting
- [retrieval_service/README.md](retrieval_service/README.md) - Vector search, Qdrant ownership
- [frontend/README.md](frontend/README.md) - React UI setup

## Performance Tips

### Indexing Tuning

```env
# In backend/.env / ingestion_service/.env
CHUNK_SIZE=1000              # Smaller = more chunks, more vectors
CHUNK_OVERLAP=150            # Avoid context loss at boundaries
PARENT_MAX_TOKENS=2500       # Parent chunk size limit
CHILD_CHUNK_SIZE=500         # Child chunk size
```

### Search Tuning

```env
# In backend/.env / retrieval_service/.env
RERANKER_ENABLED=true        # Better results, slower
QUERY_REWRITE_ENABLED=false  # Query expansion, slower
RETRIEVAL_TOP_K=10           # More results = slower
HYBRID_RRF_K=60              # Reciprocal rank fusion parameter
```

### Scaling

- **Nhiều documents**: Dùng `ingestion_service` riêng (không local parsing)
- **Hàng triệu vectors**: Dùng remote Qdrant server
- **Nhiều users**: Dùng load balancer phía trước backend
- **GPU acceleration**: Dùng `PDF_PARSER_MODE=marker` + GPU, reranker với GPU

## Kiểm tra nhanh

```bash
# Syntax check
python -m compileall backend ingestion_service ollama_service retrieval_service

# Frontend build
cd frontend
npm run build

# All services health
curl http://localhost:8000/health
curl http://localhost:8100/health
curl http://localhost:8200/health
curl http://localhost:8300/health
```

Nếu dùng sandbox hoặc môi trường hạn chế quyền, `npm run build` có thể cần chạy ngoài sandbox vì Vite/esbuild phải spawn binary native.
