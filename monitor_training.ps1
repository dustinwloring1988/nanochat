# NanoChat Training Monitor Script
# Monitors the automated training pipeline and provides status updates

param(
    [int]$RefreshSeconds = 30,
    [switch]$ShowLogs = $false
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NanoChat Training Monitor" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitoring: docker-compose logs nanochat-train-auto" -ForegroundColor Yellow
Write-Host "Refresh interval: $RefreshSeconds seconds" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow
Write-Host ""

$startTime = Get-Date

while ($true) {
    Clear-Host
    
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "NanoChat Training Status" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    Write-Host "Elapsed: $((New-TimeSpan -Start $startTime -End (Get-Date)).ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    Write-Host "Updated: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # Check container status
    $containerStatus = docker ps --filter "name=nanochat-train-auto" --format "{{.Status}}" 2>$null
    
    if ($containerStatus) {
        Write-Host "Container Status: " -NoNewline -ForegroundColor Green
        Write-Host "$containerStatus" -ForegroundColor White
    } else {
        Write-Host "Container Status: " -NoNewline -ForegroundColor Red
        Write-Host "Not Running" -ForegroundColor White
        Write-Host ""
        Write-Host "Training may have completed or failed. Check logs with:" -ForegroundColor Yellow
        Write-Host "  docker-compose logs nanochat-train-auto" -ForegroundColor White
        break
    }
    
    Write-Host ""
    Write-Host "Recent Training Output:" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
    # Get last 15 lines of logs
    $logs = docker-compose logs --tail 15 nanochat-train-auto 2>$null | Select-Object -Last 15
    
    foreach ($line in $logs) {
        # Color code different types of output
        if ($line -match "STEP \d+:|Stage \d+:|==================") {
            Write-Host $line -ForegroundColor Yellow
        } elseif ($line -match "error|failed|Error|Failed") {
            Write-Host $line -ForegroundColor Red
        } elseif ($line -match "complete|Complete|✓|SUCCESS") {
            Write-Host $line -ForegroundColor Green
        } elseif ($line -match "training|Training|loss|Loss") {
            Write-Host $line -ForegroundColor Cyan
        } else {
            Write-Host $line -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
    # Check for checkpoint files
    $checkpointPath = ".\checkpoints"
    if (Test-Path $checkpointPath) {
        $checkpoints = Get-ChildItem -Path $checkpointPath -Recurse -File | Measure-Object -Property Length -Sum
        if ($checkpoints.Count -gt 0) {
            $sizeGB = [math]::Round($checkpoints.Sum / 1GB, 2)
            Write-Host "Checkpoints: $($checkpoints.Count) files, ${sizeGB}GB" -ForegroundColor Green
        }
    }
    
    Write-Host ""
    Write-Host "Monitoring... (refresh in $RefreshSeconds seconds)" -ForegroundColor Gray
    Start-Sleep -Seconds $RefreshSeconds
}

Write-Host ""
Write-Host "Monitoring stopped." -ForegroundColor Yellow
