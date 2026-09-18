# Quick training status checker

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "Nanochat Training Status Check" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan

# Check container status
$status = docker ps --filter name=nanochat-base --format "{{.Status}}"
if ($status) {
    Write-Host "✓ Container: nanochat-base" -ForegroundColor Green
    Write-Host "  Status: $status" -ForegroundColor Green
} else {
    Write-Host "✗ Container not running" -ForegroundColor Red
    Write-Host "`nTry: docker ps -a --filter name=nanochat" -ForegroundColor Yellow
    exit
}

# Check for training steps
Write-Host "`nLatest training steps:" -ForegroundColor Yellow
docker logs nanochat-base 2>&1 | Select-String -Pattern "step \d{5}" | Select-Object -Last 5

# Show last few log lines
Write-Host "`nLast 10 log lines:" -ForegroundColor Yellow
docker logs nanochat-base 2>&1 | Select-Object -Last 10

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "To follow logs in real-time, run:" -ForegroundColor Yellow
Write-Host "  docker logs -f nanochat-base" -ForegroundColor White
Write-Host "========================================================================`n" -ForegroundColor Cyan
