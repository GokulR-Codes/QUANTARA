# Run the React frontend dev server
# Usage: .\scripts\run_frontend.ps1

$ErrorActionPreference = "Stop"
$FrontendDir = "$PSScriptRoot\..\frontend"

Write-Host "Starting frontend dev server..." -ForegroundColor Cyan
Write-Host "Frontend will be available at: http://localhost:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow

Set-Location $FrontendDir
cmd /c "npm run dev"
