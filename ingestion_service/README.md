# Ingestion Service

Service xử lý tài liệu được tách riêng khỏi backend RAG. Nhiệm vụ chính:

- Nhận file `.pdf`, `.txt`, `.md` và trả về markdown qua `POST /v1/parse`.
- Nhận markdown đã parse và trả về chunks + metadata qua `POST /v1/split`.
- Chạy parser nặng như Marker/OCR trong process riêng, phù hợp đặt trên GPU cloud.

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
