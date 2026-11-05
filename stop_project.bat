@echo off
title Stop Fraud Detection Project
echo ======================================================================
echo   Stopping FraudShield Project [Containers, Streaming, and Dashboard]
echo ======================================================================
echo.

:: 1. Stop all Docker Compose containers
echo [Step 1/3] Stopping Docker containers...
cd /d "%~dp0fraud-detection-pipeline-main"
docker compose down >nul 2>&1
docker stop kafka zookeeper spark-master spark-worker fraud-producer fraud-consumer >nul 2>&1
cd /d "%~dp0"
echo   [OK] Docker containers stopped.
echo.

:: 2. Terminate host Python streaming and dashboard processes
echo [Step 2/3] Stopping Python streaming and dashboard processes...
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'stream_real_producer|stream_consumer_xgb|dashboard[/\\]app.py|transaction_producer' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo   [OK] Host Python processes terminated.
echo.

:: 3. Free ports 5050 (Dashboard) and 9092 (Kafka)
echo [Step 3/3] Ensuring ports 5050 and 9092 are released...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1
echo   [OK] Network ports released.
echo.

echo ======================================================================
echo   [SUCCESS] Whole project has been completely stopped!
echo ======================================================================
echo.
pause
