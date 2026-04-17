# RAG App (FastAPI + React)

Ứng dụng RAG (Retrieval-Augmented Generation) cho phép:

- Upload và quản lý tài liệu (`.pdf`, `.txt`, `.md`)
- Chunk + embedding theo parent-child (HyQ) và lưu index trên Qdrant
- Indexing bất đồng bộ theo từng tài liệu (incremental), không rebuild toàn bộ mỗi lần embed
- Batching LLM cho HyQ metadata và cache metadata theo `file_hash`
- Chat hỏi đáp dựa trên ngữ cảnh đã truy xuất
- Truy vết nguồn theo document/chunk/trang + metadata schema từ kết quả retrieval
- Lưu lịch sử phiên chat bằng SQLite

## 1) Kiến trúc dự án

```text
rag/
├─ backend/                    # FastAPI + SQLAlchemy + LangChain
│  ├─ app/
│  │  ├─ api/                  # REST endpoints (documents, chat)
│  │  ├─ core/                 # settings
│  │  ├─ services/             # document processing + RAG runtime
│  │  ├─ db.py                 # SQLite engine/session
│  │  ├─ models.py             # ORM models
│  │  └─ main.py               # FastAPI app entrypoint
│  └─ requirements.txt
├─ frontend/                   # React + Vite
│  ├─ src/pages/               # ChatPage, DocumentsPage
│  └─ package.json
├─ .env.example
└─ README.md
```

## 2) Công nghệ sử dụng

- Backend: FastAPI, SQLAlchemy, LangChain, Qdrant
- Embedding model: `bge-m3` phục vụ qua Ollama (`langchain-ollama`)
- Chat model: Ollama (`langchain-ollama`)
- Database: SQLite (`backend/storage/app.db` được tạo tự động)
- Frontend: React, Vite, Axios, React Router

## 3) Yêu cầu môi trường

- Python 3.10+
- Node.js 18+
- NPM 9+
- Ollama chạy local hoặc qua remote endpoint

## 4) Setup Ollama

### Cài Ollama

- Windows / macOS: tải và cài từ trang chính thức: https://ollama.com/download
- Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Khởi động Ollama service

- Windows / macOS: mở ứng dụng Ollama (service sẽ chạy nền)
- Linux:

```bash
ollama serve
```

### Kiểm tra Ollama đang chạy

```bash
curl http://localhost:11434/api/tags
```

Nếu service hoạt động, lệnh sẽ trả về danh sách models (JSON).

### Pull model chat cần cho dự án

```bash
ollama pull llama3.1:8b
```

Embedding `bge-m3` được tải trực tiếp trong Python ở lần chạy đầu tiên.

## 5) Cài đặt và chạy

### Bước 1: Tạo biến môi trường

Tại thư mục gốc dự án:

```bash
cp .env.example .env
```

Mở `.env` và thêm tối thiểu:

```env
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL_NAME=BAAI/bge-m3
QDRANT_COLLECTION_NAME=global_child_chunks
PDF_PARSER_MODE=legacy
LLM_MODEL=llama3.1:8b
```

`PDF_PARSER_MODE`:

- `legacy`: dùng parser cũ (`PyPDFLoader`)
- `marker`: dùng Marker để parse PDF theo layout + metadata trang

Nếu dùng `marker`, cài thêm trong backend venv:

```bash
pip install marker-pdf
```

Nếu chưa setup Ollama và pull models, làm theo mục "Setup Ollama" phía trên.

### Bước 2: Chạy backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Khi chạy embed PDF lớn (OCR/Marker), nên chạy không `--reload` để ổn định hơn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`

- Health check: `GET /api/health`
- Swagger UI: `http://localhost:8000/docs`

### Bước 3: Chạy frontend

Mở terminal khác:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL mặc định: `http://localhost:5173`

## 6) Cấu hình frontend API

Frontend gọi backend qua:

- Mặc định: `http://localhost:8000/api`
- Override bằng biến `VITE_API_BASE_URL`

Tạo file `frontend/.env` nếu cần:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## 7) Luồng sử dụng chuẩn

1. Mở trang `Documents` để upload file.
2. Bấm `Embed` trên từng tài liệu (hoặc `Rebuild index`).
3. Backend trả về ngay, chuyển trạng thái tài liệu sang `indexing` và xử lý nền.
4. Khi hoàn tất, tài liệu chuyển sang `embedded` (hoặc `index_failed` nếu lỗi).
5. Chuyển sang trang `Chat` để đặt câu hỏi.
6. Hệ thống sẽ:
	 - truy xuất hybrid (vector + keyword) từ child index,
	 - kéo ngược parent chunk text để đưa vào prompt,
	 - gửi ngữ cảnh + lịch sử chat vào LLM,
	 - lưu cả tin nhắn user và assistant vào DB.

## 8) API chính

### Authentication

- Document APIs yêu cầu JWT Bearer token (đăng nhập qua `POST /api/auth/login`).
- Chat APIs hiện public (không yêu cầu token).
- Kiểm tra phiên admin hiện tại qua `GET /api/auth/me`.

### Health

- `GET /api/health`

### Documents

- `GET /api/documents`: danh sách tài liệu
- `POST /api/documents/upload` (multipart form): upload tài liệu
	- fields:
		- `file` (required)
		- `title` (optional)
- `GET /api/documents/{document_id}`: chi tiết tài liệu
- `GET /api/documents/{document_id}/chunks`: danh sách chunks + metadata (hỗ trợ query `offset`, `limit`)
- `PUT /api/documents/{document_id}`: cập nhật tiêu đề
- `DELETE /api/documents/{document_id}`: xóa tài liệu + xóa vectors của tài liệu trong Qdrant
- `POST /api/documents/{document_id}/embed`: queue indexing nền cho tài liệu (`202 Accepted`)
- `POST /api/documents/reindex`: queue indexing nền cho các tài liệu pending/thay đổi
	- `embed` trả về ngay với `chunks_created=0` và `indexed_chunks=0` khi vừa queue.
	- nếu tài liệu không đổi `file_hash` và đã `embedded`, `embed` trả về số chunk/vector đã có (không re-index).

### Chat

- `GET /api/chat/sessions`: danh sách phiên chat
- `POST /api/chat/sessions`: tạo phiên chat rỗng
- `DELETE /api/chat/sessions/{session_id}`: xóa phiên chat
- `GET /api/chat/sessions/{session_id}/messages`: lấy lịch sử tin nhắn
- `POST /api/chat/query`: gửi câu hỏi RAG
	- body:
		- `session_id` (optional)
		- `message` (required)
		- `top_k` (optional, 1..20)
		- `document_ids` (optional, lọc theo danh sách tài liệu)

## 9) Biến môi trường backend

### Tùy chọn

- `APP_NAME` (default: `RAG App Backend`)
- `APP_ENV` (default: `development`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `EMBEDDING_MODEL_NAME` (default: `BAAI/bge-m3`)
- `QDRANT_URL` (default: rỗng, dùng local embedded Qdrant)
- `QDRANT_API_KEY` (default: rỗng)
- `QDRANT_COLLECTION_NAME` (default: `global_child_chunks`)
- `PDF_PARSER_MODE` (default: `legacy`, hỗ trợ `legacy` hoặc `marker`)
- `CHUNK_SIZE` (default: `500`)
- `CHUNK_OVERLAP` (default: `50`)
- `RETRIEVER_K` (default: `4`)
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE` (default: `0.0`)
- `HYQ_ENABLED` (default: `true`)
- `HYQ_USE_LLM` (default: `false`)
- `HYQ_MODEL` (default: dùng lại `LLM_MODEL` nếu để trống)
- `METADATA_USE_LLM` (default: kế thừa từ `HYQ_USE_LLM`)
- `METADATA_MODEL` (default: kế thừa từ `HYQ_MODEL`, sau đó `LLM_MODEL`)
- `METADATA_OLLAMA_NUM_THREAD` (default: kế thừa `OLLAMA_NUM_THREAD`)
- `METADATA_OLLAMA_NUM_PREDICT` (default: `256`)
- `METADATA_LLM_BATCH_SIZE` (default: `8`)
- `METADATA_LLM_BATCH_MAX_CHARS` (default: `12000`)
- `VECTOR_BATCH_SIZE` (default: `64`)
- `HYQ_SUMMARY_WORDS` (default: `50`)
- `HYQ_QUESTIONS_PER_CHUNK` (default: `3`)
- `HYBRID_VECTOR_RRF_WEIGHT` (default: `1.0`)
- `HYBRID_KEYWORD_RRF_WEIGHT` (default: `1.2`)
- `HYBRID_RRF_K` (default: `60`)
- `HYBRID_PROBE_MULTIPLIER` (default: `4`)

## 10) Dữ liệu runtime được sinh tự động

Thư mục `backend/storage/` được tạo khi chạy backend, bao gồm:

- `uploads/`: file đã upload
- `indexes/global_qdrant/`: dữ liệu Qdrant local (khi không cấu hình `QDRANT_URL`)
- `app.db`: SQLite database

Các file/thư mục này đã được ignore trong git.

## 11) Một số lưu ý vận hành

- Nếu chưa embed tài liệu hoặc index rỗng, chat vẫn chạy nhưng sẽ không có ngữ cảnh truy xuất.
- Sau khi xóa tài liệu, hệ thống chỉ xóa vectors thuộc tài liệu đó (incremental).
- CORS đã mở cho frontend local (`5173`, `3000`) trong backend.
- Cần đảm bảo Ollama đang chạy cho chat model.
- Lần chạy đầu cho embedding cần internet để tải weights `bge-m3` từ Hugging Face.
- BackgroundTasks chạy trong process FastAPI; nếu server restart giữa chừng, job indexing đang chạy có thể bị gián đoạn.
- Metadata cache được lưu trong SQLite theo `document_id + file_hash + chunk_fingerprint` để tái sử dụng khi re-index cùng nội dung.

## 12) Lệnh nhanh (Windows PowerShell)

Backend:

```powershell
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```
