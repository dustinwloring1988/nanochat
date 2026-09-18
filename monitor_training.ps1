# Monitor Docker training progress
# Run this script to watch training in real-time

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Nanochat Protocol v1.0.0 Training Monitor" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if container is running
$status = docker ps --filter name=nanochat-protocol-v1 --format "{{.Status}}"
if ($status) {
    Write-Host "✓ Container Status: $status" -ForegroundColor Green
} else {
    Write-Host "✗ Container not running" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Streaming logs (Ctrl+C to stop)..." -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Follow logs
docker logs -f --tail 50 nanochat-protocol-v1
