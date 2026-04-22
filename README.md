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
ollama pull bge-m3
ollama pull llama3.2:3b
```

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

## 13) Future Work (tối ưu RAG)

Mục này tập trung vào các hạng mục chưa có trong code hiện tại nhưng có tác động lớn đến chất lượng trả lời, tốc độ truy xuất, và độ ổn định vận hành.

### P0 - Ưu tiên cao (nên làm trước)

1. Bổ sung reranker sau bước hybrid retrieval
	- Hiện tại đang dùng RRF để trộn vector + keyword, chưa có bước rerank cuối theo mức độ liên quan ngữ nghĩa sâu.
	- Đề xuất: lấy top N (ví dụ 20) từ hybrid, sau đó rerank về top K bằng cross-encoder hoặc LLM reranker nhẹ.
	- Kỳ vọng: tăng độ chính xác câu trả lời cho truy vấn dài, đa ý, hoặc nhiều chunk na ná nhau.

2. Tối ưu keyword retrieval để tránh full scan SQLite
	- Hiện tại keyword scoring đang duyệt toàn bộ `document_chunks` theo Python, dễ thành nút thắt khi dữ liệu tăng.
	- Đề xuất: đưa lexical retrieval sang BM25/FTS (SQLite FTS5 hoặc OpenSearch/Meilisearch), rồi mới fusion với vector.
	- Kỳ vọng: giảm mạnh độ trễ query trên tập dữ liệu lớn.

3. Thêm filter trực tiếp ở tầng vector search
	- Với `document_ids`, hiện tại có bước lọc sau khi lấy kết quả vector; có thể giảm hiệu quả khi dữ liệu lớn.
	- Đề xuất: dùng payload filter ngay trong truy vấn Qdrant (`document_id in [...]`) để giảm candidate không cần thiết.
	- Kỳ vọng: latency thấp hơn, kết quả ổn định hơn khi có document scope.

4. Tách indexing job khỏi FastAPI process
	- Hiện indexing đang chạy bằng `BackgroundTasks`; nếu API restart có thể gián đoạn job.
	- Đề xuất: chuyển sang queue worker (Celery/RQ/Dramatiq + Redis) và có trạng thái retry/dead-letter.
	- Kỳ vọng: indexing bền vững hơn trong production.

### P1 - Ưu tiên trung bình (nâng chất lượng và vận hành)

1. Query rewrite và multi-query retrieval
	- Tự động sinh 2-4 biến thể câu hỏi để truy xuất rộng hơn (đồng nghĩa, viết tắt, mã văn bản).
	- Hợp nhất kết quả bằng RRF hoặc weighted merge.

2. Context compression trước khi vào LLM
	- Hiện prompt có thể chứa nguyên văn nhiều chunk; dễ tốn token và nhiễu.
	- Đề xuất: thêm bước nén context theo câu liên quan hoặc sentence-level extraction trước khi generate.

3. Guardrails groundedness
	- Thêm bước kiểm tra câu trả lời có bám nguồn hay không (citation coverage / unsupported-claim check).
	- Nếu thiếu bằng chứng, phản hồi theo chế độ "insufficient context" thay vì suy diễn.

4. Cơ chế fallback model/runtime
	- Khi Ollama chậm hoặc lỗi, bổ sung timeout + fallback model cho chat/rerank để tránh fail cứng.

### P2 - Nâng cao (scale và observability)

1. Đánh giá RAG tự động (offline + regression)
	- Xây bộ eval gồm câu hỏi chuẩn, expected facts, và metrics: Recall@K, MRR, groundedness, answer relevance.
	- Chạy định kỳ sau mỗi thay đổi prompt/retrieval/indexing.

2. Telemetry chi tiết theo từng stage
	- Ghi latency tách bạch: query rewrite, vector search, keyword search, rerank, generation.
	- Export metrics qua Prometheus/OpenTelemetry để theo dõi P95/P99.

3. Chính sách lifecycle cho logs và index
	- Query logs hiện ghi file JSON nội bộ; cần thêm rotation/retention để tránh phình storage.
	- Bổ sung quy trình backup/restore cho SQLite + Qdrant.

4. Multi-tenant và quyền truy cập theo tài liệu
	- Chuẩn hóa schema metadata cho ACL để lọc retrieval theo user/role/team.
	- Chuẩn bị sẵn kiến trúc nếu mở rộng từ nội bộ sang nhiều nhóm dữ liệu độc lập.

### KPI gợi ý để đo hiệu quả tối ưu

- Chất lượng: Recall@5 >= 0.75, grounded answer rate >= 0.85.
- Tốc độ: P95 `/api/chat/query` < 2.5s (không tính thời gian model lớn trong tải cao).
- Indexing: thời gian embed theo tài liệu giảm >= 30% khi bật cache + batching.
- Độ ổn định: tỷ lệ job indexing lỗi < 1% và có retry tự động.
