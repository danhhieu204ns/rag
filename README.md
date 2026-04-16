# RAG App (FastAPI + React)

Ứng dụng RAG (Retrieval-Augmented Generation) cho phép:

- Upload và quản lý tài liệu (`.pdf`, `.txt`, `.md`)
- Chunk + embedding tài liệu và lưu FAISS index
- Chat hỏi đáp dựa trên ngữ cảnh đã truy xuất
- Truy vết nguồn theo document/chunk/trang từ kết quả retrieval
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

- Backend: FastAPI, SQLAlchemy, LangChain, FAISS
- Embedding model: `bge-m3` chạy trực tiếp bằng Python (`sentence-transformers`)
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
3. Chuyển sang trang `Chat` để đặt câu hỏi.
4. Hệ thống sẽ:
	 - truy xuất chunks liên quan từ FAISS,
	 - gửi ngữ cảnh + lịch sử chat vào LLM,
	 - lưu cả tin nhắn user và assistant vào DB.

## 8) API chính

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
- `DELETE /api/documents/{document_id}`: xóa tài liệu + rebuild index
- `POST /api/documents/{document_id}/embed`: tạo chunks + rebuild index
- `POST /api/documents/reindex`: rebuild toàn bộ index từ chunks trong DB

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
- `PDF_PARSER_MODE` (default: `legacy`, hỗ trợ `legacy` hoặc `marker`)
- `CHUNK_SIZE` (default: `500`)
- `CHUNK_OVERLAP` (default: `50`)
- `RETRIEVER_K` (default: `4`)
- `LLM_MODEL` (default: `llama3.1:8b`)
- `LLM_TEMPERATURE` (default: `0.0`)

## 10) Dữ liệu runtime được sinh tự động

Thư mục `backend/storage/` được tạo khi chạy backend, bao gồm:

- `uploads/`: file đã upload
- `indexes/global_faiss/`: FAISS index (`index.faiss`, `index.pkl`)
- `app.db`: SQLite database

Các file/thư mục này đã được ignore trong git.

## 11) Một số lưu ý vận hành

- Nếu chưa embed tài liệu hoặc index rỗng, chat vẫn chạy nhưng sẽ không có ngữ cảnh truy xuất.
- Sau khi xóa tài liệu, hệ thống tự rebuild index từ các chunks còn lại.
- CORS đã mở cho frontend local (`5173`, `3000`) trong backend.
- Cần đảm bảo Ollama đang chạy cho chat model.
- Lần chạy đầu cho embedding cần internet để tải weights `bge-m3` từ Hugging Face.

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
