# Ingestion Service

Service indexing được tách riêng khỏi backend RAG. Nhiệm vụ chính:

- Nhận file `.pdf`, `.txt`, `.md` và parse/split/enrich qua `POST /v1/index/build`.
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
      "chunk_index": 0,
      "content": "chunk text",
      "source_page": 1,
      "source_kind": "pdf_marker_page",
      "source_metadata": {}
    }
  ],
  "child_rows": [
    {
      "chunk_index": 0,
      "child_type": "summary",
      "child_index": 0,
      "child_text": "Tóm tắt: ...",
      "source_page": 1,
      "source_kind": "pdf_marker_page",
      "source_metadata": {}
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
      "source_page": 2,
      "child_type": "summary",
      "child_index": 0,
      "child_text": "Tóm tắt: ...",
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
