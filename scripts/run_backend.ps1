# Run the FastAPI backend
# Usage: .\scripts\run_backend.ps1

$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."

Write-Host "Starting Quantum-Centric Federated Learning Backend..." -ForegroundColor Cyan
Write-Host "API will be available at: http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "API docs: http://127.0.0.1:8000/api/docs" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""

python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload --log-level info
