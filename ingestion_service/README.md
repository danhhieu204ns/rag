# Ingestion Service

Service indexing được tách riêng khỏi backend RAG. Nhiệm vụ chính:

- Nhận file `.pdf`, `.txt`, `.md`, render Markdown và tạo section-aware parent-child chunks qua `POST /v1/index/build`.
- Nhận child rows đã map `parent_chunk_id` để embed + index qua `POST /v1/index/upsert`.
- Duy trì API parse/split cũ (`/v1/parse`, `/v1/split`) để debug.

## Run

```bash
cd ingestion_service
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

URL mặc định: `http://localhost:8100`

- Health: `GET /health`
- Swagger UI: `http://localhost:8100/docs`

## Environment

```env
# Core settings
INGESTION_APP_NAME=RAG Ingestion Service
INGESTION_STORAGE_DIR=ingestion_service/storage
PDF_PARSER_MODE=legacy

# Chunking configuration
CHUNKING_STRATEGY=section_parent_child
INDEXING_INDEX_TYPE=section_parent_child
PARENT_MAX_TOKENS=2500
CHILD_CHUNK_SIZE=500
CHILD_CHUNK_OVERLAP=100
PREPEND_HEADING_PATH=true

# Retrieval service integration
RETRIEVAL_SERVICE_URL=http://localhost:8200
RETRIEVAL_TIMEOUT_SECONDS=180

# API authentication
OLLAMA_API_KEY=
```

### Environment Variables Explanation

- **`PDF_PARSER_MODE`**: PDF parsing backend
  - `legacy`: dùng `PyPDFLoader` cho PDF, `TextLoader` cho text/markdown.
  - `marker`: dùng Marker để parse PDF thành markdown theo layout. Chế độ này cần dependencies Marker và có thể cần GPU/VRAM tùy tài liệu.

- **`CHUNKING_STRATEGY` & `INDEXING_INDEX_TYPE`**: Chiến lược chunking hiện tại là `section_parent_child` - tạo parent chunks theo section/heading và child chunks từ parent.

- **`PARENT_MAX_TOKENS`**: Độ dài tối đa (token) của parent chunk. Default: 2500.

- **`CHILD_CHUNK_SIZE`**: Số ký tự tối đa mỗi child chunk. Default: 500.

- **`CHILD_CHUNK_OVERLAP`**: Độ overlap giữa các child chunks. Default: 100.

- **`PREPEND_HEADING_PATH`**: Nếu `true`, thêm heading path vào đầu child text để cung cấp context. Default: true.

- **`RETRIEVAL_SERVICE_URL`**: URL của retrieval service để upsert indexed chunks. Bắt buộc để `/v1/index/upsert` hoạt động.

- **`RETRIEVAL_TIMEOUT_SECONDS`**: Timeout cho request tới retrieval service. Default: 180s.

- **`OLLAMA_API_KEY`**: Optional API key cho upstream services (nếu cần).

## API

### `GET /health`

Health check endpoint.

Response:

```json
{
  "status": "ok",
  "service": "ingestion-service",
  "version": "1.0.0"
}
```

### `GET /ready`

Readiness check endpoint (kiểm tra service sẵn sàng, bao gồm cấu hình parser mode).

Response:

```json
{
  "status": "ok",
  "service": "ingestion-service",
  "parser_mode": "legacy"
}
```

### `POST /v1/index/build`

Multipart form:

- `file`: required, hỗ trợ `.pdf`, `.txt`, `.md`.

Response:

```json
{
  "parent_chunks": [
    {
      "parent_id": "sec-0001",
      "section_id": "sec-0001",
      "chunk_index": 0,
      "content": "A\n\nchunk text",
      "title": "A",
      "heading_path": ["A"],
      "token_count": 3,
      "page_start": 1,
      "page_end": 1,
      "source_page": 1,
      "source_kind": "pdf_marker_section",
      "source_metadata": {
        "index_type": "section_parent_child"
      }
    }
  ],
  "child_rows": [
    {
      "chunk_index": 0,
      "chunk_id": "sec-0001-child-0000",
      "parent_id": "sec-0001",
      "section_id": "sec-0001",
      "child_type": "section_child",
      "child_index": 0,
      "child_text": "chunk text",
      "embedding_text": "A\n\nchunk text",
      "token_count": 2,
      "section_title": "A",
      "heading_path": ["A"],
      "page_start": 1,
      "page_end": 1,
      "source_page": 1,
      "source_kind": "pdf_marker_section",
      "index_type": "section_parent_child",
      "source_metadata": {
        "index_type": "section_parent_child"
      }
    }
  ]
}
```

### `POST /v1/index/build-from-markdown`

Giống như `/v1/index/build` nhưng nhận markdown content trực tiếp thay vì file.

Request body:

```json
{
  "markdown": "# Heading\n\nContent here",
  "source_file_path": "document.pdf",
  "source_parser": "legacy",
  "source_type": "pdf"
}
```

Response: (giống như `/v1/index/build`)

```json
{
  "parent_chunks": [...],
  "child_rows": [...]
}
```

### `POST /v1/index/upsert`

Upsert child rows vào retrieval service để embedding và indexing.

**Yêu cầu:** `RETRIEVAL_SERVICE_URL` phải được cấu hình.

JSON body:

```json
{
  "document_id": 1,
  "child_rows": [
    {
      "document_id": 1,
      "parent_chunk_id": 101,
      "chunk_id": "sec-0001-child-0000",
      "parent_id": "sec-0001",
      "section_id": "sec-0001",
      "source_page": 2,
      "page_start": 2,
      "page_end": 2,
      "child_type": "section_child",
      "child_index": 0,
      "child_text": "raw child text",
      "embedding_text": "Heading > Path\n\nraw child text",
      "token_count": 3,
      "section_title": "Heading",
      "heading_path": ["Heading", "Path"],
      "index_type": "section_parent_child",
      "source_metadata": {}
    }
  ]
}
```

Response:

```json
{
  "indexed_chunks": 1
}
```

**Details:**
- Service sẽ gửi các chunks tới `RETRIEVAL_SERVICE_URL/v1/index/chunks` theo batch (mặc định 100 chunks/batch).
- Batch đầu tiên sẽ include `purge_document_ids` để xóa old chunks của document.
- Nếu `RETRIEVAL_SERVICE_URL` không được cấu hình, endpoint trả về 500 error.
- Nếu retrieval service request thất bại, endpoint trả về 502 error.

### `POST /v1/parse`

Multipart form:

- `file`: required, hỗ trợ `.pdf`, `.txt`, `.md`.

Endpoint này dùng để parse file thành markdown mà không tạo chunks (debug).

Response:

```json
{
  "markdown": "# Parsed content",
  "source_parser": "legacy",
  "source_type": "pdf"
}
```

Ví dụ:

```bash
curl -F "file=@sample.pdf" http://localhost:8100/v1/parse
```

### `POST /v1/split`

JSON body:

```json
{
  "markdown": "# Parsed content",
  "source_file_path": "sample.pdf",
  "source_parser": "legacy",
  "source_type": "pdf",
  "chunk_size": 1000,
  "chunk_overlap": 150
}
```

Endpoint này dùng để split markdown thành chunks (debug).

Response:

```json
{
  "chunks": [
    {
      "page_content": "chunk text",
      "metadata": {
        "source": "sample.pdf",
        "source_parser": "legacy",
        "source_type": "pdf",
        "source_page": 1
      }
    }
  ]
}
```

## Logging

Ingestion service tự động log chi tiết tất cả requests vào:

```
ingestion_service/storage/logs/ingestion_service_YYYYMMDD_HHMMSS.log
```

Logs bao gồm:
- Timing của mỗi step trong request
- Số lượng chunks được tạo
- Lỗi và exceptions chi tiết
- Request metadata (filename, markdown size, etc.)

## Backend integration

Trong `.env` của repo/backend, đặt:

```env
INGESTION_SERVICE_URL=http://localhost:8100
INGESTION_TIMEOUT_SECONDS=300
```

Khi `INGESTION_SERVICE_URL` có giá trị, backend sẽ gọi ingestion service để parse/split/index. Khi để rỗng, backend có thể dùng fallback local.

## Internal Flow

1. **File Upload (`/v1/index/build`)**
   - Parse file → markdown (dùng PDF_PARSER_MODE)
   - Build section-aware parent-child chunks
   - Return parent chunks + child rows

2. **Upsert to Index (`/v1/index/upsert`)**
   - Nhận child rows với metadata
   - Upsert tới retrieval service theo batch
   - Batch đầu tiên xóa old chunks của document

3. **Debug Endpoints (`/v1/parse`, `/v1/split`)**
   - `/v1/parse`: chỉ parse file → markdown
   - `/v1/split`: chỉ split markdown → chunks (cũ, không dùng parent-child)
