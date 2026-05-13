#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE="${1:-}"

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo python
  fi
}

ensure_python_deps() {
  local dir="$1"
  local py
  py="$(python_cmd)"

  if [ ! -d "$dir/.venv" ]; then
    "$py" -m venv "$dir/.venv"
  fi

  # shellcheck source=/dev/null
  source "$dir/.venv/bin/activate"
  python -m pip install -q -U pip wheel setuptools
  if [ -f "$dir/requirements.txt" ]; then
    python -m pip install -q -r "$dir/requirements.txt"
  fi
}

run_python_service() {
  local dir="$1"
  local port="$2"
  cd "$dir"
  ensure_python_deps "$dir"
  set -a
  [ -f .env ] && source .env
  set +a
  python -m uvicorn app.main:app --host 0.0.0.0 --port "$port"
}

case "$SERVICE" in
  ollama)
    ollama serve
    ;;
  qdrant)
    mkdir -p "$ROOT_DIR/.qdrant"
    QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__STORAGE__STORAGE_PATH="$ROOT_DIR/.qdrant" qdrant
    ;;
  ollama-service)
    run_python_service "$ROOT_DIR/ollama_service" 8200
    ;;
  ingestion-service)
    run_python_service "$ROOT_DIR/ingestion_service" 8100
    ;;
  retrieval-service)
    run_python_service "$ROOT_DIR/retrieval_service" 8300
    ;;
  backend)
    run_python_service "$ROOT_DIR/backend" 8000
    ;;
  frontend)
    cd "$ROOT_DIR/frontend"
    [ -d node_modules ] || npm install
    npm run dev
    ;;
  *)
    echo "Usage: $0 {ollama|qdrant|ollama-service|ingestion-service|retrieval-service|backend|frontend}" >&2
    exit 1
    ;;
esac
