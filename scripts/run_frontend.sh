#!/usr/bin/env bash
# Run the React dev server (Linux/macOS equivalent of scripts/run_frontend.ps1).
set -euo pipefail
cd "$(dirname "$0")/../frontend"
echo "Frontend: http://localhost:5173"
exec npm run dev
