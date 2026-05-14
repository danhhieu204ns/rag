# RAG Retrieval Service

Standalone FastAPI service for retrieval and vector indexing.

It owns direct access to Qdrant and calls only `ollama_service` for embeddings.
Core role: query embedding + vector search + filter + rerank + context response.
Other services should use this API instead of touching Qdrant directly.

## Run

```powershell
cd retrieval_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8300
```

Health check:

```text
GET http://localhost:8300/health
```

## API

```text
POST   /v1/retrieve
POST   /v1/search/vector
POST   /v1/search/hybrid
POST   /v1/index/chunks
DELETE /v1/index/document/{document_id}
GET    /health
GET    /ready
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

`/v1/search/hybrid` combines vector retrieval from Qdrant with BM25 keyword
retrieval over `document_chunks` in the shared metadata database, then merges
candidates with reciprocal-rank fusion. The route keeps the same contract as
`/v1/retrieve` so the Orchestrator does not need to know the retrieval strategy.

BM25 uses SQLite FTS5 and indexes parent chunk content plus selected metadata
fields: title, heading path, `search_optimization` keywords, document codes,
dates, source info, and a bilingual alias field. Vietnamese-English expansion
is enabled by default so Vietnamese queries can match common English terms such
as `annual leave`, `policy`, `contract`, and `performance review`. Add
domain-specific terms through `BM25_EXTRA_SYNONYMS`.

## Configuration

- See `.env.example` for supported environment variables and defaults.
- Important vars: `OLLAMA_SERVICE_URL`, `OLLAMA_API_KEY`, `QDRANT_URL` / `QDRANT_PATH`,
  `QDRANT_COLLECTION_NAME`, and database path `RETRIEVAL_DATABASE_PATH`.
- Hybrid vars: `HYBRID_PROBE_MULTIPLIER`, `HYBRID_RRF_K`,
  `HYBRID_VECTOR_RRF_WEIGHT`, `HYBRID_KEYWORD_RRF_WEIGHT`.
- Reranker vars: `RERANKER_ENABLED`, `RERANKER_MODEL`,
  `RERANKER_CANDIDATE_POOL`.
- BM25 vars: `BM25_ENABLED`, `BM25_BILINGUAL_EXPANSION`,
  `BM25_CANDIDATE_LIMIT_MULTIPLIER`, `BM25_EXTRA_SYNONYMS`.

## Dependencies

- See `requirements.txt` for Python dependencies (FastAPI, qdrant-client, httpx,
  sentence-transformers for reranking, etc.).

## Notes

- This service exposes both `/health` (basic status) and `/ready` (readiness checks
  that validate Qdrant and the Ollama embedding service). The `/ready` endpoint
  returns HTTP 503 when dependencies are not ready.
