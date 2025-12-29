#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting E-commerce Demand Forecasting GUI..."
echo ""

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BE_DIR="$ROOT_DIR/be"
GUI_DIR="$ROOT_DIR/gui"
BE_VENV="$BE_DIR/.venv"
BACKEND_LOG="$ROOT_DIR/backend.log"
FRONTEND_LOG="$ROOT_DIR/frontend.log"
BACKEND_PID=""
FRONTEND_PID=""
TAIL_PID=""

# Load project-wide .env if present so dev envs (e.g. PUBLIC_HOST/HMR_*) are applied
if [[ -f "$ROOT_DIR/.env" ]]; then
  echo "🔐 Loading environment variables from $ROOT_DIR/.env"
  # export all variables defined in the .env so child processes inherit them
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PY_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PY_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PY_BIN=python
else
    echo "❌ Error: python3 not found. Install Python 3.11 or newer."
    exit 1
fi

if [ ! -d "$BE_VENV" ]; then
    echo "🛠 Creating virtual environment for backend at $BE_VENV..."
    "$PY_BIN" -m venv "$BE_VENV"
fi

source "$BE_VENV/bin/activate"

echo "📦 Checking backend Python requirements..."

# Compute a hash of the requirements file to avoid reinstalling if nothing changed
_requirements_file="$BE_DIR/requirements.txt"
_requirements_hash_file="$BE_VENV/.requirements_hash"

# Helper: compute SHA256 of a file in a portable way
compute_hash() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        python - <<PY -c "import sys,hashlib
p=sys.argv[1]
print(hashlib.sha256(open(p,'rb').read()).hexdigest())" "$1"
PY
    fi
}

python -m pip install --upgrade pip setuptools wheel >/dev/null
if [ -f "$_requirements_file" ]; then
    new_hash=$(compute_hash "$_requirements_file")
    old_hash=""
    if [ -f "$_requirements_hash_file" ]; then
        old_hash=$(cat "$_requirements_hash_file")
    fi
    if [ "$new_hash" = "$old_hash" ]; then
        echo "✔ Backend requirements unchanged — skipping pip install"
    else
        echo "🔄 Installing/updating backend Python requirements from $_requirements_file..."
        python -m pip install -r "$_requirements_file"
        echo "$new_hash" > "$_requirements_hash_file"
    fi
else
    echo "⚠️  No requirements.txt found in $BE_DIR — skipping pip install"
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "❌ Error: npm not found. Install Node.js/npm (Node >= 20 recommended)."
    exit 1
fi

# Install frontend deps only if package-lock.json/package.json changed or node_modules missing
lock_file=""
if [ -f "$GUI_DIR/package-lock.json" ]; then
    lock_file="$GUI_DIR/package-lock.json"
elif [ -f "$GUI_DIR/package.json" ]; then
    lock_file="$GUI_DIR/package.json"
fi

_frontend_hash_file="$GUI_DIR/.node_deps_hash"
install_frontend_deps() {
    echo "📥 Installing frontend dependencies (npm i)..."
    (cd "$GUI_DIR" && npm i)
}

if [ ! -d "$GUI_DIR/node_modules" ]; then
    echo "📥 node_modules missing — installing frontend dependencies"
    install_frontend_deps
else
    if [ -n "$lock_file" ]; then
        new_hash=$(compute_hash "$lock_file")
        old_hash=""
        if [ -f "$_frontend_hash_file" ]; then
            old_hash=$(cat "$_frontend_hash_file")
        fi
        if [ "$new_hash" = "$old_hash" ]; then
            echo "✔ Frontend dependencies appear up-to-date — skipping npm ci"
        else
            echo "🔄 Package file changed — reinstalling frontend dependencies"
            install_frontend_deps
            echo "$new_hash" > "$_frontend_hash_file"
        fi
    else
        echo "⚠️  No package.json or package-lock.json found — skipping frontend install"
    fi
fi

stop_servers() {
    echo "🛑 Shutting down servers..."
    if [ -n "${TAIL_PID:-}" ]; then pkill -P "$TAIL_PID" &>/dev/null || true; kill "$TAIL_PID" &>/dev/null || true; fi
    if [ -n "${FRONTEND_PID:-}" ]; then pkill -P "$FRONTEND_PID" &>/dev/null || true; kill "$FRONTEND_PID" &>/dev/null || true; fi
    if [ -n "${BACKEND_PID:-}" ]; then pkill -P "$BACKEND_PID" &>/dev/null || true; kill "$BACKEND_PID" &>/dev/null || true; fi
    TAIL_PID=""
    FRONTEND_PID=""
    BACKEND_PID=""
}

cleanup() {
    echo ""
    stop_servers
    deactivate >/dev/null 2>&1 || true
    exit 0
}

trap cleanup INT TERM

start_servers() {
    clear
    # Truncate log files
    > "$BACKEND_LOG"
    > "$FRONTEND_LOG"

    echo "📡 Starting FastAPI backend on http://localhost:8000..."
    cd "$BE_DIR"
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    cd "$ROOT_DIR"

    echo "⏳ Waiting for backend to become healthy..."
    for i in {1..20}; do
        if curl -fsS --max-time 1 http://127.0.0.1:8000/health >/dev/null 2>&1; then
            echo "✅ Backend is healthy"
            break
        fi
        if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
            echo "❌ Backend failed to start. See $BACKEND_LOG"
            exit 1
        fi
        sleep 0.5
    done

    echo "🎨 Starting Svelte frontend on http://localhost:5173..."
    cd "$GUI_DIR"
    npm run dev > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    cd "$ROOT_DIR"

    sleep 2

    echo ""
    echo "✅ Both servers are running!"
    echo "   Frontend: http://localhost:5173"
    echo "   Backend:  http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "Press 'r' to restart, 'c' to clear screen, 'q' to quit"
    echo ""

    tail -f "$BACKEND_LOG" "$FRONTEND_LOG" &
    TAIL_PID=$!
}

start_servers

# Interactive loop
while true; do
    read -rsn1 input
    case "$input" in
        r|R)
            echo; echo "🔄 Restarting servers..."
            stop_servers
            start_servers
            ;;
        c|C)
            clear
            echo "✅ Screen cleared."
            echo "Press 'r' to restart, 'c' to clear screen, 'q' to quit"
            ;;
        q|Q)
            cleanup
            ;;
    esac
    sleep 0.1
done
