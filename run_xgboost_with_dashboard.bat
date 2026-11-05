@echo off
title Fraud Detection - XGBoost + Dashboard
echo ======================================================================
echo   Launching Real-Time Fraud Detection with XGBoost ^& Dashboard
echo ======================================================================
echo.

:: 1. Ensure Docker Desktop is running
echo [Step 1/5] Checking Docker status...
docker info >nul 2>&1
if %ERRORLEVEL% equ 0 goto docker_ready

echo Docker Desktop is not running! Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo Waiting for Docker engine to initialize [approx 30-45 seconds]...

:wait_docker
powershell -NoProfile -Command "Start-Sleep -Seconds 4"
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 goto wait_docker

:docker_ready
echo [OK] Docker engine is ready!
echo.

:: 2. Start all 6 Docker containers
echo [Step 2/5] Starting all 6 Docker containers (Zookeeper, Kafka, Spark Master, Spark Worker, Producer, Consumer)...
cd /d "%~dp0fraud-detection-pipeline-main"
docker compose up -d
cd /d "%~dp0"
echo [OK] All 6 Docker containers are active and running!
echo.

:: 3. Reset session logs to ensure project starts at 0 transactions
echo [Step 3/5] Initializing fresh session (resetting counters to 0)...
del /f /q "%~dp0producer_stream.log" >nul 2>&1
del /f /q "%~dp0consumer_stream.log" >nul 2>&1
echo [OK] Clean session initialized!
echo.

:: 4. Launch Dashboard backend in a separate window
echo [Step 4/5] Launching Dashboard Backend on http://localhost:5050 ...
start "FraudShield Dashboard" cmd /k "cd /d "%~dp0dashboard" && python app.py"
echo [OK] Dashboard launched!
start http://localhost:5050
echo.

:: 5. Launch XGBoost Stream Consumer in this window
echo [Step 5/5] Launching XGBoost Stream Consumer...
echo ======================================================================
echo   Starting XGBoost Live Consumer (Press Ctrl+C to stop)
echo ======================================================================
python stream_consumer_xgb.py

pause
