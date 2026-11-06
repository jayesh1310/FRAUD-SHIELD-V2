@echo off
setlocal
title Stop FraudShield 2.0
cd /d "%~dp0"
set "FRAUDSHIELD_ROOT=%~dp0"
echo Stopping FraudShield 2.0...
set "FAILED=0"
docker compose -f "%~dp0fraud-detection-pipeline-main\docker-compose.yml" down
if errorlevel 1 (echo ERROR: Docker Compose could not stop the services. & set "FAILED=1")

if exist ".fraudshield-dashboard.pid" (
    powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $pidFile=Join-Path $env:FRAUDSHIELD_ROOT '.fraudshield-dashboard.pid'; $savedPid=0; if (-not [int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$savedPid)) { Write-Host 'ERROR: Invalid dashboard PID file.'; exit 1 }; $p=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $savedPid); if ($p) { $app=Join-Path $env:FRAUDSHIELD_ROOT 'dashboard\app.py'; if ($p.CommandLine -and $p.CommandLine.IndexOf($app, [StringComparison]::OrdinalIgnoreCase) -ge 0) { Stop-Process -Id $savedPid -ErrorAction Stop; Write-Host 'Dashboard stopped.' } else { Write-Host 'Dashboard PID belongs to another process; leaving it alone.' } } else { Write-Host 'Dashboard is already stopped.' }; Remove-Item -LiteralPath $pidFile"
    if errorlevel 1 (echo ERROR: Could not stop the dashboard. & set "FAILED=1")
) else (
    echo No dashboard started by start_fraudshield_2.bat was found.
)
echo Prediction data and checkpoints are preserved.
if "%FAILED%"=="1" (pause & exit /b 1)
echo FraudShield 2.0 stopped.
pause
exit /b 0
