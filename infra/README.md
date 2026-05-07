# Local Operations

This folder contains the Docker Compose entrypoint for running the current
RAG platform as a managed set of services.

## Commands

From the repository root:

```bash
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml down
docker compose -f infra/docker-compose.yml logs
docker compose -f infra/docker-compose.yml logs api-gateway
docker compose -f infra/docker-compose.yml restart ollama-service
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml build
```

## Public Ports

Only these services are published to the host by default:

- `api-gateway`: `http://localhost:8000`
- `frontend`: `http://localhost:3000`

Internal services are reachable only inside the Compose network:

- `ollama-service:8020`
- `retrieval-service:8030`
- `ingestion-service:8100`
- `qdrant:6333`
- `ollama:11434`

## Health And Readiness

Each FastAPI service exposes:

- `GET /health`
- `GET /ready`

Compose health checks use `/ready` so dependencies are checked before the
public API and frontend are considered healthy.

## Future Infrastructure

Postgres and Redis are included under the `future` profile because the current
code still uses SQLite and FastAPI background tasks.

Start them only when the application code is wired to use them:

```bash
docker compose -f infra/docker-compose.yml --profile future up -d --build
```
