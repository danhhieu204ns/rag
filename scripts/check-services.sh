#!/usr/bin/env bash
set -euo pipefail

check_url() {
  local name="$1"
  local url="$2"
  if curl -fsS "$url" >/dev/null 2>&1; then
    printf "%-18s OK      %s\n" "$name" "$url"
  else
    printf "%-18s FAIL    %s\n" "$name" "$url"
  fi
}

echo "== Processes =="
pgrep -af 'uvicorn|ollama|qdrant|npm|vite' || true

echo ""
echo "== Ports =="
ss -ltnp | grep -E ':5173|:8000|:8100|:8200|:8300|:6333|:11434' || true

echo ""
echo "== Health checks =="
check_url "frontend" "http://localhost:5173"
check_url "backend" "http://localhost:8000/ready"
check_url "ingestion" "http://localhost:8100/ready"
check_url "ollama-service" "http://localhost:8200/ready"
check_url "retrieval" "http://localhost:8300/ready"
check_url "retrieval-health" "http://localhost:8300/health"
check_url "ollama" "http://localhost:11434/api/tags"
