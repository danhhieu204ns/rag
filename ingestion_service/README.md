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
PDF_PARSER_MODE=legacy
INGESTION_APP_NAME=RAG Ingestion Service
INGESTION_STORAGE_DIR=ingestion_service/storage
```

`PDF_PARSER_MODE`:

- `legacy`: dùng `PyPDFLoader` cho PDF, `TextLoader` cho text/markdown.
- `marker`: dùng Marker để parse PDF thành markdown theo layout. Chế độ này cần dependencies Marker và có thể cần GPU/VRAM tùy tài liệu.

## API

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

### `POST /v1/index/upsert`

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

### `POST /v1/parse`

Multipart form:

- `file`: required, hỗ trợ `.pdf`, `.txt`, `.md`.

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

## Backend integration

Trong `.env` của repo/backend, đặt:

```env
INGESTION_SERVICE_URL=http://localhost:8100
INGESTION_TIMEOUT_SECONDS=300
PDF_PARSER_MODE=marker
```

Khi `INGESTION_SERVICE_URL` có giá trị, backend sẽ gọi service này để parse/split. Khi để rỗng, backend dùng fallback local để giữ tương thích môi trường cũ.
