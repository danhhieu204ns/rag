#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required" >&2
  exit 1
fi

# Start services in the background and stop them on exit.
pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT

run_service() {
  local name="$1"
  local workdir="$2"
  local env_file="$3"
  shift 3

  echo "Starting ${name}..."
  (
    if [ -n "$env_file" ] && [ -f "$env_file" ]; then
      set -a
      # shellcheck source=/dev/null
      source "$env_file"
      set +a
    elif [ -n "$env_file" ]; then
      echo "Warning: env file not found: ${env_file}" >&2
    fi
    cd "$workdir"
    "$@"
  ) &
  pids+=("$!")
}

wait_ready() {
  local name="$1"
  local url="$2"
  local timeout_s="${3:-60}"
  local interval_s="${4:-2}"
  local start_ts

  start_ts="$(date +%s)"
  echo "Waiting for ${name} at ${url}..."
  while true; do
    if python - <<PY
import sys
import urllib.request
try:
  urllib.request.urlopen("${url}", timeout=2)
  sys.exit(0)
except Exception:
  sys.exit(1)
PY
    then
      echo "${name} is ready."
      break
    fi
    if [ $(("$(date +%s)" - start_ts)) -ge "$timeout_s" ]; then
      echo "${name} did not become ready in ${timeout_s}s: ${url}" >&2
      exit 1
    fi
    sleep "$interval_s"
  done
}

start_optional_service() {
  local name="$1"
  local cmd="$2"
  local url="$3"
  local env_file="$4"

  if [ -n "$cmd" ] && command -v "$cmd" >/dev/null 2>&1; then
    run_service "$name" "$ROOT_DIR" "$env_file" "$cmd" ${5:-}
    wait_ready "$name" "$url"
    return 0
  fi

  echo "${name} is not running and '${cmd}' was not found in PATH." >&2
  echo "Start ${name} manually, then re-run this script." >&2
  exit 1
}

# Native dependencies (no Docker)
start_optional_service "ollama" "ollama" "http://127.0.0.1:11434/api/tags" ""
start_optional_service "qdrant" "qdrant" "http://127.0.0.1:6333/readyz" ""

# Backend services
run_service "ollama_service" "$ROOT_DIR/ollama_service" "$ROOT_DIR/ollama_service/.env" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8200
wait_ready "ollama_service" "http://127.0.0.1:8200/ready"

run_service "ingestion_service" "$ROOT_DIR/ingestion_service" "$ROOT_DIR/ingestion_service/.env" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8100
wait_ready "ingestion_service" "http://127.0.0.1:8100/ready"

run_service "retrieval_service" "$ROOT_DIR/retrieval_service" "$ROOT_DIR/retrieval_service/.env" \
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8030
wait_ready "retrieval_service" "http://127.0.0.1:8030/ready"

run_service "backend" "$ROOT_DIR/backend" "$ROOT_DIR/backend/.env" \
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
wait_ready "backend" "http://127.0.0.1:8000/ready"

# Frontend
if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd "$ROOT_DIR/frontend" && npm install)
fi

run_service "frontend" "$ROOT_DIR/frontend" "$ROOT_DIR/frontend/.env" npm run dev

wait
