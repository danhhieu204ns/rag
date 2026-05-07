#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/runtime"
DATA_DIR="$ROOT_DIR/storage"

mkdir -p "$LOG_DIR" "$PID_DIR" \
  "$DATA_DIR/backend" \
  "$DATA_DIR/ingestion" \
  "$DATA_DIR/qdrant" \
  "$DATA_DIR/ollama"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.env"
  set +a
fi

API_GATEWAY_PORT="${API_GATEWAY_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
OLLAMA_SERVICE_PORT="${OLLAMA_SERVICE_PORT:-8020}"
SHIELD_API_KEY="${SHIELD_API_KEY:-change-this-key}"
OLLAMA_API_KEY="${OLLAMA_API_KEY:-$SHIELD_API_KEY}"

wait_url() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-120}"

  echo "Waiting for $name: $url"

  for _ in $(seq 1 "$timeout_s"); do
    if python - "$url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request
url = sys.argv[1]
urllib.request.urlopen(url, timeout=3)
PY
    then
      echo "$name is ready"
      return 0
    fi

    sleep 1
  done

  echo "Timeout waiting for $name"
  return 1
}

wait_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout_s="${4:-120}"

  echo "Waiting for $name: $host:$port"

  for _ in $(seq 1 "$timeout_s"); do
    if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys
host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=3):
    pass
PY
    then
      echo "$name is ready"
      return 0
    fi

    sleep 1
  done

  echo "Timeout waiting for $name"
  return 1
}

start_service() {
  local name="$1"
  local workdir="$2"
  local cmd="$3"
  local logfile="$LOG_DIR/$name.log"
  local pidfile="$PID_DIR/$name.pid"

  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name is already running, pid=$(cat "$pidfile")"
    return 0
  fi

  echo "Starting $name..."
  (
    cd "$workdir"
    nohup bash -lc "$cmd" > "$logfile" 2>&1 &
    echo $! > "$pidfile"
  )

  echo "$name started, log=$logfile"
}

install_python_service() {
  local dir="$1"
  local name="$2"

  echo "Installing Python dependencies for $name..."

  cd "$dir"

  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi

  # shellcheck source=/dev/null
  source .venv/bin/activate
  python -m pip install -U pip wheel setuptools

  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
  elif [ -f "pyproject.toml" ]; then
    pip install -e .
  else
    echo "No requirements.txt or pyproject.toml found in $dir, skip install."
  fi

  deactivate
}

install_frontend() {
  local dir="$1"

  echo "Installing frontend dependencies..."

  cd "$dir"

  if [ -f "package-lock.json" ]; then
    npm ci
  else
    npm install
  fi
}

up() {
  echo "Starting local compose for RAG project..."

  if command -v ollama >/dev/null 2>&1; then
    start_service "ollama" "$ROOT_DIR" \
      "OLLAMA_HOST=0.0.0.0:11434 OLLAMA_MODELS='$DATA_DIR/ollama' ollama serve"

    wait_port "127.0.0.1" "11434" "ollama" 120
  else
    echo "ollama command not found. Please install/start Ollama manually."
    exit 1
  fi

  if command -v qdrant >/dev/null 2>&1; then
    start_service "qdrant" "$ROOT_DIR" \
      "QDRANT__SERVICE__HTTP_PORT=6333 QDRANT__STORAGE__STORAGE_PATH='$DATA_DIR/qdrant' qdrant"

    wait_port "127.0.0.1" "6333" "qdrant" 120
  else
    echo "qdrant command not found."
    echo "Option 1: install Qdrant binary locally."
    echo "Option 2: use Qdrant Cloud/remote Qdrant and set QDRANT_URL in .env."
    echo "Option 3: if Qdrant is already running elsewhere, ignore this warning and ensure QDRANT_URL is correct."
  fi

  install_python_service "$ROOT_DIR/ollama_service" "ollama-service"
  install_python_service "$ROOT_DIR/ingestion_service" "ingestion-service"
  install_python_service "$ROOT_DIR/retrieval_service" "retrieval-service"
  install_python_service "$ROOT_DIR/backend" "api-gateway"

  start_service "ollama-service" "$ROOT_DIR/ollama_service" \
    "source .venv/bin/activate && \
     UPSTREAM_OLLAMA_BASE_URL='${UPSTREAM_OLLAMA_BASE_URL:-http://127.0.0.1:11434}' \
     SHIELD_API_KEY='$SHIELD_API_KEY' \
     python -m uvicorn app.main:app --host 0.0.0.0 --port $OLLAMA_SERVICE_PORT"

  wait_url "http://127.0.0.1:$OLLAMA_SERVICE_PORT/ready" "ollama-service" 120

  start_service "ingestion-service" "$ROOT_DIR/ingestion_service" \
    "source .venv/bin/activate && \
     python -m uvicorn app.main:app --host 0.0.0.0 --port 8100"

  wait_url "http://127.0.0.1:8100/ready" "ingestion-service" 120

  start_service "retrieval-service" "$ROOT_DIR/retrieval_service" \
    "source .venv/bin/activate && \
     OLLAMA_SERVICE_URL='${OLLAMA_SERVICE_URL:-http://127.0.0.1:$OLLAMA_SERVICE_PORT}' \
     OLLAMA_API_KEY='$OLLAMA_API_KEY' \
     QDRANT_URL='${QDRANT_URL:-http://127.0.0.1:6333}' \
     RETRIEVAL_DATABASE_PATH='${RETRIEVAL_DATABASE_PATH:-$ROOT_DIR/backend/storage/app.db}' \
     python -m uvicorn app.main:app --host 0.0.0.0 --port 8030"

  wait_url "http://127.0.0.1:8030/ready" "retrieval-service" 120

  start_service "api-gateway" "$ROOT_DIR/backend" \
    "source .venv/bin/activate && \
     APP_NAME='RAG API Gateway' \
     OLLAMA_BASE_URL='${OLLAMA_BASE_URL:-http://127.0.0.1:$OLLAMA_SERVICE_PORT}' \
     OLLAMA_API_KEY='$OLLAMA_API_KEY' \
     INGESTION_SERVICE_URL='${INGESTION_SERVICE_URL:-http://127.0.0.1:8100}' \
     RETRIEVAL_SERVICE_URL='${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:8030}' \
     QDRANT_URL='${QDRANT_URL:-http://127.0.0.1:6333}' \
     python -m uvicorn app.main:app --host 0.0.0.0 --port $API_GATEWAY_PORT"

  wait_url "http://127.0.0.1:$API_GATEWAY_PORT/ready" "api-gateway" 120

  if [ -d "$ROOT_DIR/frontend" ]; then
    install_frontend "$ROOT_DIR/frontend"

    start_service "frontend" "$ROOT_DIR/frontend" \
      "VITE_API_BASE_URL='${VITE_API_BASE_URL:-http://localhost:$API_GATEWAY_PORT/api}' \
       npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT"

    wait_port "127.0.0.1" "$FRONTEND_PORT" "frontend" 120
  fi

  echo ""
  echo "Local compose started successfully."
  echo "API Gateway: http://127.0.0.1:$API_GATEWAY_PORT"
  echo "Frontend:    http://127.0.0.1:$FRONTEND_PORT"
  echo ""
  echo "Logs:        ./logs/*.log"
  echo "Stop:        ./scripts/local-compose.sh down"
}

down() {
  echo "Stopping local compose..."

  for pidfile in "$PID_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue

    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"

    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name, pid=$pid"
      kill "$pid" || true
    fi

    rm -f "$pidfile"
  done

  echo "Stopped."
}

status() {
  echo "Service status:"

  for pidfile in "$PID_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue

    name="$(basename "$pidfile" .pid)"
    pid="$(cat "$pidfile")"

    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: running, pid=$pid"
    else
      echo "$name: stopped"
    fi
  done
}

logs() {
  local name="${1:-}"

  if [ -z "$name" ]; then
    ls -1 "$LOG_DIR"/*.log 2>/dev/null || true
    echo ""
    echo "Use: $0 logs api-gateway"
    return 0
  fi

  tail -f "$LOG_DIR/$name.log"
}

case "${1:-up}" in
  up)
    up
    ;;
  down|stop)
    down
    ;;
  restart)
    down
    up
    ;;
  status)
    status
    ;;
  logs)
    logs "${2:-}"
    ;;
  *)
    echo "Usage: $0 {up|down|restart|status|logs [service-name]}"
    exit 1
    ;;
esac
