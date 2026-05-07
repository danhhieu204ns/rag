# RAG Retrieval Service

Standalone FastAPI service for retrieval and vector indexing.

It owns direct access to Qdrant and calls only `ollama_service` for embeddings.
Other services should use this API instead of touching Qdrant directly.

## Run

```powershell
cd retrieval_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8030
```

Health check:

```text
GET http://localhost:8030/health
```

## API

```text
POST   /v1/retrieve
POST   /v1/search/vector
POST   /v1/search/hybrid
POST   /v1/index/chunks
DELETE /v1/index/document/{document_id}
GET    /health
```

Example retrieve:

```json
{
  "query": "Quy trinh dao tao noi bo gom nhung buoc nao?",
  "collection": "training_documents",
  "top_k": 5,
  "filters": {
    "document_ids": [1, 2],
    "metadata": {
      "department": "VTAca"
    }
  }
}
```

`/v1/search/hybrid` currently keeps the same contract as `/v1/retrieve` and uses
vector retrieval. BM25/rerank can be added behind this route without changing the
RAG Orchestrator contract.
