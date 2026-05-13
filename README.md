# RAG App

Ứng dụng RAG dùng FastAPI, React, Ollama và Qdrant để upload tài liệu, tạo index vector theo mô hình parent-child, rồi chat hỏi đáp có nguồn tham chiếu.

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
| `ingestion_service` | Indexing service: nhận tài liệu đầu vào, extract/clean/chunk, enrich metadata, embed và index (qua retrieval API). |
| `retrieval_service` | Retrieval service: query embedding, vector search, filter, rerank và trả context cho backend. |
| `ollama_service` | Gateway bảo vệ Ollama thật, ép model theo cấu hình, rate limit, phục vụ chat, embedding và metadata/HyQ. |
| `qdrant` | Vector database nội bộ, không gọi trực tiếp từ frontend. |

Frontend chỉ nên gọi `backend` qua `http://localhost:8000` hoặc `/api`. Các service còn lại là nội bộ.

Luồng chính:

1. Frontend gọi backend qua `/api`.
2. Backend lưu metadata, file upload và lịch sử chat trong SQLite.
3. Backend gọi `ingestion_service` để xử lý indexing phase (parse -> enrich -> embed -> index).
4. Backend gọi `retrieval_service` để truy vấn context.
5. Backend gọi `ollama_service` để generation câu trả lời.

Luồng upload và index tài liệu:

```text
Frontend
   ↓
Backend nhận upload và lưu metadata/file
   ↓
Backend gọi Ingestion Service để parse + enrich + embed + index
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
- `INGESTION_SERVICE_URL`: URL parse/split service; backend yêu cầu service này cho luồng parse/split tài liệu.
- `RETRIEVAL_SERVICE_URL`: URL retrieval service; để rỗng nếu muốn backend truy cập Qdrant trực tiếp.
- `QDRANT_URL`: URL Qdrant server; để rỗng thì dùng Qdrant local embedded path.
- `PDF_PARSER_MODE`: `legacy` hoặc `marker`.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: cấu hình split tài liệu.
- `QUERY_REWRITE_ENABLED`, `QUERY_REWRITE_MIN_TERMS`, `QUERY_REWRITE_MAX_TERMS`: cấu hình rewrite query.
- `HYBRID_PROBE_MULTIPLIER`, `HYBRID_RRF_K`, `HYBRID_VECTOR_RRF_WEIGHT`, `HYBRID_KEYWORD_RRF_WEIGHT`: cấu hình hybrid retrieval.
- `RERANKER_ENABLED`, `RERANKER_MODEL`, `RERANKER_CANDIDATE_POOL`: cấu hình reranker.

## Dữ liệu runtime

Các thư mục sau được sinh tự động và đã được ignore:

- `backend/storage/`: SQLite DB, uploads, local Qdrant, logs.
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
