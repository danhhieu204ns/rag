# Ingestion Service

Dich vu parse + split tai lieu duoc tach rieng khoi backend.

## Run

```bash
cd ingestion_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8100
```

## API

- `GET /health`
- `POST /v1/parse` (multipart file)
- `POST /v1/split` (JSON markdown + metadata + chunk params)

## Env

- `PDF_PARSER_MODE=legacy|marker`
- `INGESTION_STORAGE_DIR=...` (optional)
- `INGESTION_APP_NAME=...` (optional)
