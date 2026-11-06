@echo off
setlocal
title FraudShield 2.0 - Stop System
cd /d "%~dp0"
set "ROOT_DIR=%~dp0"
set "COMPOSE_FILE=%ROOT_DIR%fraud-detection-pipeline-main\docker-compose.yml"

echo ======================================================================
echo   Stopping FraudShield 2.0 [Containers, Streaming, and Dashboard]
echo ======================================================================
echo.

:: 1. Stop Docker Compose services
echo [Step 1/3] Stopping Docker Compose services...
docker compose -f "%COMPOSE_FILE%" down >nul 2>&1
docker stop fraud-consumer fraud-producer kafka zookeeper spark-master spark-worker >nul 2>&1
echo   [OK] Docker containers stopped.
echo.

:: 2. Terminate Dashboard process
echo [Step 2/3] Terminating Dashboard backend process...
if exist "%ROOT_DIR%.fraudshield-dashboard.pid" (
    powershell -NoProfile -Command "$pidFile = Join-Path $env:ROOT_DIR '.fraudshield-dashboard.pid'; if (Test-Path $pidFile) { $pId = (Get-Content $pidFile -Raw).Trim(); if ($pId) { Stop-Process -Id $pId -Force -ErrorAction SilentlyContinue }; Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }" >nul 2>&1
)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'dashboard[/\\]app\.py|transaction_producer|spark_consumer' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo   [OK] Host Python processes stopped.
echo.

:: 3. Free network ports
echo [Step 3/3] Ensuring network ports are released (5050, 9092, 4040, 8080, 8081, 7077)...
powershell -NoProfile -Command "5050, 9092, 4040, 8080, 8081, 7077 | ForEach-Object { $p = $_; Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }" >nul 2>&1
echo   [OK] Network ports released.
echo.

echo ======================================================================
echo   [SUCCESS] FraudShield 2.0 has been completely stopped!
echo ======================================================================
echo.
pause
exit /b 0
