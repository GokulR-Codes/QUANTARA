#!/usr/bin/env bash
# Run the FastAPI backend (Linux/macOS equivalent of scripts/run_backend.ps1).
# Usage: ./scripts/run_backend.sh        (from anywhere; activates nothing — use your venv's python)
set -euo pipefail
cd "$(dirname "$0")/.."
echo "API: http://127.0.0.1:8000   docs: http://127.0.0.1:8000/api/docs"
exec python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --log-level info
