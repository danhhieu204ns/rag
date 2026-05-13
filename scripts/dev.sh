#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo python
  fi
}

wait_url() {
  local name="$1"
  local url="$2"
  local timeout_s="${3:-90}"
  local py
  py="$(python_cmd)"

  echo "Waiting for ${name}: ${url}"
  for _ in $(seq 1 "$timeout_s"); do
    if "$py" - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
try:
    urllib.request.urlopen(sys.argv[1], timeout=2)
except Exception:
    sys.exit(1)
PY
    then
      echo "${name} is ready"
      return 0
    fi
    sleep 1
  done

  echo "Timeout waiting for ${name}" >&2
  return 1
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
  deactivate
}

run_bg() {
  local name="$1"
  local workdir="$2"
  shift 2

  echo "Starting ${name}..."
  (
    cd "$workdir"
    "$@"
  ) &
  PIDS+=("$!")
}

run_python_service() {
  local name="$1"
  local dir="$2"
  local port="$3"

  ensure_python_deps "$dir"
  run_bg "$name" "$dir" bash -lc "set -a; [ -f .env ] && source .env; set +a; source .venv/bin/activate; python -m uvicorn app.main:app --host 0.0.0.0 --port ${port}"
}

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  command -v ollama >/dev/null 2>&1 || { echo "ollama command not found" >&2; exit 1; }
  run_bg "ollama" "$ROOT_DIR" ollama serve
  wait_url "ollama" "http://127.0.0.1:11434/api/tags" 120
fi

if ! curl -fsS http://127.0.0.1:6333/readyz >/dev/null 2>&1; then
  command -v qdrant >/dev/null 2>&1 || { echo "qdrant command not found" >&2; exit 1; }
  mkdir -p "$ROOT_DIR/.qdrant"
  run_bg "qdrant" "$ROOT_DIR" env QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__STORAGE__STORAGE_PATH="$ROOT_DIR/.qdrant" qdrant
  wait_url "qdrant" "http://127.0.0.1:6333/readyz" 120
fi

run_python_service "ollama_service" "$ROOT_DIR/ollama_service" 8200
wait_url "ollama_service" "http://127.0.0.1:8200/ready"

run_python_service "ingestion_service" "$ROOT_DIR/ingestion_service" 8100
wait_url "ingestion_service" "http://127.0.0.1:8100/ready"

run_python_service "retrieval_service" "$ROOT_DIR/retrieval_service" 8300
wait_url "retrieval_service" "http://127.0.0.1:8300/ready"

run_python_service "backend" "$ROOT_DIR/backend" 8000
wait_url "backend" "http://127.0.0.1:8000/ready"

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  (cd "$ROOT_DIR/frontend" && npm install)
fi
run_bg "frontend" "$ROOT_DIR/frontend" npm run dev

echo ""
echo "RAG app is running"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"
echo "Stop:     Ctrl+C"
wait
