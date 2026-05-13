# RAG App Backend (FastAPI)

Backend là **Orchestrator** chính của hệ thống RAG. Nó là service duy nhất mà frontend gọi trực tiếp.

## Features

- **Document Management**: CRUD API, upload, persistence
- **Async Indexing**: Incremental indexing pipeline per document (BackgroundTasks)
- **Dual PDF Parser**: Support `legacy` (PyMuPDF) hoặc `marker` mode
- **Section-Aware Chunking**: Parent-child chunk model theo markdown heading sections
- **Hybrid Search**: Vector + keyword search with reciprocal-rank-fusion
- **Delegation Architecture**: Parse → ingestion_service, Search → retrieval_service, LLM → ollama_service
- **Parent-Child Retrieval**: Child vectors indexed, parent text returned to LLM
- **Chat History**: Persistent SQLite chat sessions and messages
- **Vector Storage**: Qdrant backend (embedded local hoặc remote mode)
- **Authentication**: JWT-based admin auth, public chat endpoints

## Architecture Role

Backend là trung tâm điều phối (**Orchestrator**):

```text
Frontend
   |
   v
Backend / Orchestrator
   |
   |--------------------------|--------------------------|
   v                          v                          v
ingestion_service         retrieval_service          ollama_service
(parse + chunk + embed)   (vector search)            (LLM + embedding)
   |                          |                          |
   └──────────────────────────┴──────────────────────────┘
                        |
                        v
                    Qdrant DB
```

### Responsibilities

| Thành phần | Sở hữu | Gọi |
|-----------|--------|-----|
| **backend** | API chính (auth, documents, chat, users), SQLite metadata, chat history, RAG orchestration | → ingestion_service, → retrieval_service, → ollama_service |
| **ingestion_service** | Parse (PDF/TXT/MD), section-aware chunking, document processing pipeline | → retrieval_service (để index) |
| **retrieval_service** | Qdrant vector DB, vector search, keyword search, reranking, embedding queries | → ollama_service (để embed), → Qdrant |
| **ollama_service** | Ollama proxy/shield, model enforcement, rate limiting, inference endpoints | → Ollama upstream |
| **Qdrant** | Vector store, không gọi trực tiếp từ frontend, frontend không truy cập trực tiếp | - |

### Backend Ownership

**Owns:**
- Public API (`/api/auth`, `/api/documents`, `/api/chat`, `/api/users`)
- SQLite database: Users, Documents, DocumentChunks, DocumentIndexState, ChatSessions, ChatMessages
- Auth & admin user management
- Document upload & metadata persistence
- RAG orchestration (question → retrieval → prompt → LLM → answer + sources)
- Indexing orchestration (queue jobs → call ingestion → persist state)
- Fallback local Qdrant mode (when `RETRIEVAL_SERVICE_URL` empty)

**Should NOT own:**
- Heavy document parsing (→ ingestion_service)
- Vector DB implementation (→ retrieval_service)
- Direct Ollama access (→ ollama_service)

### Recommended Service Boundary

```text
backend
  ├─ POST → ingestion_service         # /v1/index/build, /v1/index/upsert
  ├─ POST → retrieval_service         # /v1/search/hybrid, /v1/index/chunks, DELETE /v1/index/document
  ├─ POST → ollama_service            # /api/chat, /api/embed (via ChatOllama, OllamaEmbeddings)
  └─ SQLite
  
retrieval_service
  ├─ POST → ollama_service            # /api/embed (query embedding)
  └─ → Qdrant
```

## API Endpoints

### Public Endpoints

- `GET /health`, `GET /api/health` - Health check
- `GET /ready`, `GET /api/ready` - Readiness check (kiểm tra DB, ollama_service, ingestion_service, retrieval_service)

### Authentication & Users

Require **JWT Bearer token** (admin role):

- `POST /api/auth/login` - Đăng nhập (username/password → JWT)
- `POST /api/auth/register` - Đăng ký admin mới
- `GET /api/users` - Danh sách users
- `GET /api/users/{user_id}` - Lấy user
- `PUT /api/users/{user_id}` - Cập nhật user
- `DELETE /api/users/{user_id}` - Xóa user

### Documents

Require **JWT Bearer token** (admin role):

- `POST /api/documents/upload` - Upload file (PDF/TXT/MD)
- `GET /api/documents` - Danh sách documents
- `GET /api/documents/{document_id}` - Chi tiết document
- `DELETE /api/documents/{document_id}` - Xóa document
- `POST /api/documents/{document_id}/process` - Queue parse + chunk + embed (async, returns 202)
- `POST /api/documents/{document_id}/embed` - Queue embedding only (async, returns 202)
- `POST /api/documents/reindex` - Reindex pending documents (async)
- `GET /api/documents/{document_id}/chunks` - Danh sách chunks
- `GET /api/documents/{document_id}/status` - Indexing status

### Chat

Public endpoints (no auth required):

- `POST /api/chat/sessions` - Tạo chat session
- `GET /api/chat/sessions` - Danh sách sessions
- `GET /api/chat/sessions/{session_id}` - Chi tiết session
- `DELETE /api/chat/sessions/{session_id}` - Xóa session
- `POST /api/chat/sessions/{session_id}/messages` - Gửi message (question), nhận answer + sources
- `GET /api/chat/sessions/{session_id}/messages` - Lịch sử messages

## Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, routers, logging
│   ├── db.py                   # SQLAlchemy session, engine, init_db()
│   ├── models.py               # SQLAlchemy ORM models (User, Document, Chat*)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── api/                    # Route handlers
│   │   ├── auth.py             # Login, register
│   │   ├── users.py            # User CRUD
│   │   ├── documents.py        # Document CRUD, upload, indexing
│   │   └── chat.py             # Chat sessions, messages, RAG orchestration
│   ├── core/                   # Settings, security, logging
│   │   ├── settings.py         # Pydantic settings from .env
│   │   ├── security.py         # JWT verification, password hashing
│   │   ├── request_logger.py   # Request logging middleware
│   │   └── query_logger.py     # RAG query logging
│   └── services/               # Business logic
│       ├── ingestion_client.py    # Client to ingestion_service
│       ├── retrieval_client.py    # Client to retrieval_service
│       ├── ollama_service_client.py # Client to ollama_service
│       ├── chunk_metadata.py      # Chunk metadata builders
│       └── rag/                    # RAG orchestration (fallback local mode)
│           ├── orchestrator.py    # RAG query orchestration
│           ├── generation.py      # Answer generation
│           ├── retrieval.py       # Hybrid search (local fallback)
│           ├── query.py           # Query processing
│           ├── qdrant.py          # Direct Qdrant access (local fallback)
│           ├── models.py          # RAG data models
│           ├── logging.py         # RAG logging
│           └── utils.py           # Helper functions
├── storage/
│   ├── indexes/
│   │   └── global_qdrant/      # Local embedded Qdrant DB (if QDRANT_URL empty)
│   ├── uploads/                # Uploaded files
│   └── logs/                   # Service logs
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md                   # This file
```

## Database Schema

SQLite models (in `app/models.py`):

### User

```python
id: int (PK)
username: str (unique, indexed)
hashed_password: str
role: str (e.g., "admin", "user")
is_active: bool
created_at: datetime
```

### Document

```python
id: int (PK)
title: str
original_filename: str
stored_filename: str (unique)
content_type: str (MIME type)
status: str (uploaded, indexing, embedded, index_failed)
created_at: datetime
updated_at: datetime
```

### DocumentChunk

```python
id: int (PK)
document_id: int (FK → Document, cascade delete)
chunk_index: int
content: str (text)
source_page: int (nullable)
source_kind: str (e.g., "heading_section")
source_metadata_json: str (nullable, JSON)
created_at: datetime
```

### DocumentIndexState

```python
document_id: int (PK, FK → Document, cascade delete)
file_hash: str (indexed, for change detection)
indexed_parent_chunks: int
indexed_child_chunks: int
indexed_at: datetime
updated_at: datetime
```

### ChatSession

```python
id: int (PK)
title: str
created_at: datetime
updated_at: datetime
```

### ChatMessage

```python
id: int (PK)
session_id: int (FK → ChatSession, cascade delete)
role: str (user, assistant)
content: str (text)
sources_json: str (nullable, JSON array of citations)
created_at: datetime
```

## Installation & Run

### 1. Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment

Edit `backend/.env` với các URL của các services khác:

```env
# Core settings
APP_NAME=RAG Backend
SECRET_KEY=change-this-secret-key
ADMIN_DEFAULT_USERNAME=admin
ADMIN_DEFAULT_PASSWORD=admin

# External Service URLs (REQUIRED)
OLLAMA_BASE_URL=http://localhost:8200
OLLAMA_API_KEY=change-this-key
OLLAMA_CHAT_MODEL=default
OLLAMA_EMBEDDING_MODEL=default

# Optional: Delegated services
INGESTION_SERVICE_URL=http://localhost:8100
RETRIEVAL_SERVICE_URL=http://localhost:8300

# Optional: Vector Storage (Qdrant)
QDRANT_URL=                              # Empty = embedded local
QDRANT_PATH=backend/storage/indexes/global_qdrant
QDRANT_COLLECTION_NAME=documents

# Optional: Document processing
PDF_PARSER_MODE=legacy                   # legacy or marker
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

### 3. Run Backend

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Health check**: `curl http://localhost:8000/health`  
**API docs**: `http://localhost:8000/docs`

### 4. Start Other Services

Backend phụ thuộc vào các services sau. Chạy chúng trước (hoặc cùng lúc):

#### Ollama Service Shield

Yêu cầu, bảo vệ Ollama thực:

```bash
cd ollama_service
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8200
```

**Cấu hình** (`ollama_service/.env`):

```env
UPSTREAM_OLLAMA_BASE_URL=http://127.0.0.1:11434
SHIELD_API_KEY=change-this-key
CHAT_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
EMBEDDING_MODEL=qwen3-embedding:0.6b
```

Xem [ollama_service/README.md](../ollama_service/README.md) để chi tiết.

#### Ingestion Service

Tùy chọn, xử lý parse + chunking:

```bash
cd ingestion_service
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

Để `INGESTION_SERVICE_URL` trống → backend sẽ parse local (chậm hơn, không khuyên).

Xem [ingestion_service/README.md](../ingestion_service/README.md) để chi tiết.

#### Retrieval Service

Tùy chọn, sở hữu Qdrant:

```bash
cd retrieval_service
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

Để `RETRIEVAL_SERVICE_URL` trống → backend dùng Qdrant embedded local.

Xem [retrieval_service/README.md](../retrieval_service/README.md) để chi tiết.

### 5. Verify Setup

```bash
curl http://localhost:8000/api/ready
```

Response `status=ok` nếu tất cả services sẵn sàng.

## Environment Variables

### Required

**Ollama Service Integration:**
- `OLLAMA_BASE_URL`: URL tới `ollama_service`, ví dụ `http://localhost:8200` (yêu cầu)
- `OLLAMA_API_KEY`: API key xác thực qua header `x-api-key` (yêu cầu, phải trùng với `SHIELD_API_KEY` của ollama_service)
- `OLLAMA_CHAT_MODEL`: Model name cho chat (có thể để `default`, ollama_service sẽ override)
- `OLLAMA_EMBEDDING_MODEL`: Model name cho embedding (có thể để `default`)

### Core Settings

```env
APP_NAME=RAG Backend
SECRET_KEY=change-this-secret-key          # JWT signing key
ADMIN_DEFAULT_USERNAME=admin
ADMIN_DEFAULT_PASSWORD=admin

CORS_ALLOW_ORIGINS=*
CORS_ALLOW_ORIGIN_REGEX=^http(s)?://(localhost|.*\.example\.com).*
```

### Service Integration

**Ingestion Service** (tùy chọn, khuyến nghị):
- `INGESTION_SERVICE_URL`: URL tới ingestion_service, ví dụ `http://localhost:8100`
  - Nếu trống → backend sẽ parse local (chậm, không khuyên)
  - Nếu không khả dụng → upload/embed sẽ fail
- `INGESTION_TIMEOUT_SECONDS`: HTTP timeout (default: 300s)

**Retrieval Service** (tùy chọn, khuyến nghị):
- `RETRIEVAL_SERVICE_URL`: URL tới retrieval_service, ví dụ `http://localhost:8300`
  - Nếu trống → backend dùng local Qdrant (fallback mode)
  - Khuyến nghị để cho retrieval_service sở hữu Qdrant
- `RETRIEVAL_TIMEOUT_SECONDS`: HTTP timeout (default: 60s)

### Vector Storage (Qdrant)

```env
QDRANT_URL=                                 # Empty = local embedded mode
QDRANT_PATH=backend/storage/indexes/global_qdrant  # Local storage path
QDRANT_COLLECTION_NAME=documents
QDRANT_API_KEY=                             # Required if QDRANT_URL is set
```

- Nếu `QDRANT_URL` trống → backend dùng embedded Qdrant (local mode)
- Nếu `QDRANT_URL` set → backend dùng remote Qdrant server (khuyến nghị cho production)
- Data persisted tại: `backend/storage/indexes/global_qdrant/`

### Document Processing

```env
PDF_PARSER_MODE=legacy              # legacy (PyMuPDF) or marker
CHUNK_SIZE=1000                     # Target chunk length (characters)
CHUNK_OVERLAP=150                   # Overlap between chunks
RERANKER_ENABLED=true               # Enable BGE reranking
QUERY_REWRITE_ENABLED=false         # Enable query rewriting (HyDE-like)
```

**PDF_PARSER_MODE:**
- `legacy`: PyMuPDF + TextLoader cho PDF/TXT/MD
- `marker`: Marker layout-aware PDF parser (cần GPU, chậm)

### Optional Features

```env
MAX_UPLOAD_FILE_SIZE_MB=100
DATABASE_PATH=backend/storage/app.db
STORAGE_DIR=backend/storage
```

## Data Flows

### Document Indexing Flow

```
Frontend upload
    ↓
Backend nhận file, lưu metadata/file
    ↓
Backend queue background job: POST /v1/index/build → ingestion_service
    ↓
Ingestion Service:
  - Parse file (PDF/TXT/MD)
  - Split thành section-aware parent-child chunks
  - Return chunk list
    ↓
Backend queue background job: POST /v1/index/upsert → ingestion_service
    ↓
Ingestion Service:
  - Embed child chunks qua ollama_service
  - Return embeddings
    ↓
Backend queue background job: POST /v1/index/chunks → retrieval_service
    ↓
Retrieval Service:
  - Store vectors + metadata → Qdrant
    ↓
Backend update document status: uploaded → indexing → embedded
(or index_failed nếu lỗi)
```

### Chat Query Flow

```
Frontend send question
    ↓
Backend nhận question
    ↓
Backend POST /v1/search/hybrid → retrieval_service
    ↓
Retrieval Service:
  - Embed query qua ollama_service
  - Vector search + keyword search
  - Hybrid ranking (RRF)
  - Return top_k contexts
    ↓
Backend build prompt:
  [system_prompt] + [contexts] + [question]
    ↓
Backend POST /api/chat → ollama_service
    ↓
Ollama Service:
  - Call upstream Ollama
  - Return answer
    ↓
Backend format response:
  { answer, sources (với citations) }
    ↓
Backend save chat history (user_msg, assistant_msg, sources) → SQLite
    ↓
Frontend receive answer + sources
```

## Async Indexing Behavior

Backend dùng `BackgroundTasks` để xử lý indexing không đồng bộ:

- `POST /api/documents/{document_id}/process` - Parse + embed + index (recommended)
  - Queues background job, returns `202 Accepted` immediately
  - `status` transitions: `uploaded` → `indexing` → `embedded`

- `POST /api/documents/{document_id}/embed` - Embed + index only
  - Queues background job, returns `202 Accepted` immediately

- `POST /api/documents/reindex` - Reindex pending documents
  - Queues pending documents lại, useful sau restart/failure

**Status transitions:**
- `uploaded` → `indexing` (job started)
- `indexing` → `embedded` (success)
- `indexing` → `index_failed` (error)

**Behavior:**
- Qdrant upsert dùng async mode (`wait=false`) để tăng throughput
- Incremental updates scoped per `document_id` (không recreate collection)
- Cached counts trả về nếu file hash không thay đổi

## Indexing Strategy

**Section-Parent-Child Model:**

- **Parent Chunks**: Markdown heading sections (toàn bộ section text)
- **Child Chunks**: Sub-sections của parent (chunk nhỏ hơn)
- **Vectors**: Only child chunks được embed + indexed
- **Retrieval**: Return child contexts, nhưng có thể map về parent context

**Implementation:**
- Ingestion service xử lý chunking strategy
- Backend không embed, chỉ orchestrate
- Không có per-chunk LLM metadata indexing (disabled)

## Operational Notes

**BackgroundTasks:**
- In-process jobs, nếu backend restart → in-flight jobs bị dừng
- `POST /api/documents/reindex` dùng để queue lại pending documents sau restart

**Logging:**
- File logs: `backend/storage/logs/backend_service_*.log` (rotating)
- Query/RAG logs: mỗi request log step timing
- Debug markdown: `ingestion_service/storage/markdown_logs/marker/` (nếu dùng marker)

**Performance Tuning:**
- `CHUNK_SIZE`: Tăng → ít chunks hơn, nhưng context dài hơn
- `CHUNK_OVERLAP`: Overlap để tránh mất context ở boundaries
- `RERANKER_ENABLED`: Rerank top_k results (chậm hơn nhưng kết quả tốt hơn)

**Scaling:**
- Ingestion: scale riêng nếu có nhiều documents
- Retrieval: scale Qdrant nếu có hàng triệu vectors
- Ollama: scale nếu có nhiều concurrent requests

## Service Dependencies & Fallbacks

| Service | Yêu cầu | Khi không có URL | Khi không khả dụng |
|---------|--------|-----------------|-------------------|
| `ollama_service` | ✓ REQUIRED | N/A | ✗ Fail (503) |
| `ingestion_service` | ○ Recommended | Parse local | ✗ Fail (503) |
| `retrieval_service` | ○ Recommended | Use local Qdrant fallback | ✗ Use local fallback |
| Qdrant | ○ Optional | Embedded local (`QDRANT_PATH`) | N/A |

**Recommended Production Setup:**
- All services deployed separately
- Remote Qdrant server
- Remote ingestion service
- Load balancers cho high availability

## Troubleshooting

### Backend not starting

**Error: `OLLAMA_BASE_URL` or `OLLAMA_API_KEY` missing**
```bash
# Fix: Set environment variables
export OLLAMA_BASE_URL=http://localhost:8200
export OLLAMA_API_KEY=change-this-key
```

**Error: `RuntimeError: database is locked`**
```bash
# SQLite lock, try:
rm backend/storage/app.db
# Then restart
```

### Ingestion/Upload failing

**Status shows `index_failed`**
- Check backend logs: `tail -f backend/storage/logs/backend_service_*.log`
- Verify `INGESTION_SERVICE_URL` is set and reachable
- Check ingestion service logs: `tail -f ingestion_service/storage/logs/*`

**File upload too slow**
- PDF parsing chậm → use `marker` mode (cần GPU)
- Or: reduce `CHUNK_SIZE`, increase `INGESTION_TIMEOUT_SECONDS`

### Chat queries returning no results

**No contexts retrieved**
- Check `RETRIEVAL_SERVICE_URL` is set
- Verify documents indexed successfully (status = `embedded`)
- Try: `curl http://localhost:8300/ready` (retrieval_service health)
- Check hybrid search parameters (top_k, filters)

**LLM not responding**
- Check `OLLAMA_BASE_URL` is reachable
- Verify `OLLAMA_CHAT_MODEL` exists: `ollama list`
- Check ollama_service logs: `tail -f ollama_service/storage/logs/*`

### Vector storage issues

**Qdrant embedded mode slow**
- Embedded SQLite not optimized, use remote Qdrant:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-key
```

**Vectors not indexed**
- Check `retrieval_service` is running
- Verify `RETRIEVAL_SERVICE_URL` in backend
- Check Qdrant is writable: `curl http://localhost:6333/health`

## Quick Reference

### Common Commands

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8000/api/ready
curl http://localhost:8200/health    # ollama_service
curl http://localhost:8100/health    # ingestion_service
curl http://localhost:8300/health    # retrieval_service

# View logs
tail -f backend/storage/logs/backend_service_*.log
tail -f ollama_service/storage/logs/ollama_service_*.log
tail -f ingestion_service/storage/logs/*

# API documentation
http://localhost:8000/docs            # Swagger UI
http://localhost:8100/docs            # Ingestion service
http://localhost:8200/docs            # Ollama service
http://localhost:8300/docs            # Retrieval service

# Database inspection (SQLite)
sqlite3 backend/storage/app.db
  SELECT * FROM documents;
  SELECT * FROM chat_sessions;
  SELECT COUNT(*) FROM document_chunks;
```

### Environment Variable Checklist

**Development (Minimal):**
```env
✓ OLLAMA_BASE_URL
✓ OLLAMA_API_KEY
✓ OLLAMA_CHAT_MODEL
✓ OLLAMA_EMBEDDING_MODEL
✓ INGESTION_SERVICE_URL
✓ RETRIEVAL_SERVICE_URL
```

**Production:**
```env
✓ APP_NAME
✓ SECRET_KEY (strong)
✓ ADMIN_DEFAULT_PASSWORD (strong)
✓ OLLAMA_BASE_URL (remote)
✓ OLLAMA_API_KEY (strong)
✓ INGESTION_SERVICE_URL (remote)
✓ RETRIEVAL_SERVICE_URL (remote)
✓ QDRANT_URL (remote)
✓ QDRANT_API_KEY
✓ CORS_ALLOW_ORIGINS (specific domains)
```

## Related Services

- [ingestion_service/README.md](../ingestion_service/README.md) - Document parsing & chunking
- [ollama_service/README.md](../ollama_service/README.md) - Ollama proxy & rate limiting
- [retrieval_service/README.md](../retrieval_service/README.md) - Vector search & indexing
- [root README.md](../README.md) - Overall architecture & setup guide
