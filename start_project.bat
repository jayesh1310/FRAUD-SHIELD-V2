@echo off
setlocal enabledelayedexpansion
title FraudShield 2.0 - Start System (From Scratch)
cd /d "%~dp0"
set "ROOT_DIR=%~dp0"
set "COMPOSE_FILE=%ROOT_DIR%fraud-detection-pipeline-main\docker-compose.yml"

echo ======================================================================
echo   FraudShield 2.0: Financial Fraud Intelligence System
echo   Start All 6 Docker Services + Flask Dashboard (From Scratch)
echo ======================================================================
echo.

:: 1. Validate dataset
if not exist "%ROOT_DIR%fraudTest.csv" (
    echo [ERROR] 'fraudTest.csv' not found in: %ROOT_DIR%
    echo Please place 'fraudTest.csv' in the project directory before starting.
    goto :fail
)

:: 2. Check Python installation & dependencies
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found on PATH. Please install Python 3.8+ and retry.
    goto :fail
)
python -c "import flask, flask_cors, pandas, pyarrow" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required dashboard dependencies...
    python -m pip install -r "%ROOT_DIR%dashboard\requirements.txt"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies.
        goto :fail
    )
)

:: 3. Check Docker status
echo [Step 1/5] Checking Docker Desktop status...
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker command not found on PATH. Install Docker Desktop first.
    goto :fail
)

docker info >nul 2>&1
if %ERRORLEVEL% equ 0 goto :docker_ready

echo [INFO] Docker Desktop is not running. Launching Docker Desktop...
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
    start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
) else (
    echo [ERROR] Docker Desktop not found at standard path. Please launch Docker Desktop manually.
    goto :fail
)

echo [INFO] Waiting for Docker engine to initialize (up to 90 seconds)...
set /a DOCKER_WAIT=0
:wait_docker
timeout /t 5 /nobreak >nul
set /a DOCKER_WAIT+=5
docker info >nul 2>&1
if %ERRORLEVEL% equ 0 goto :docker_ready
if !DOCKER_WAIT! geq 90 (
    echo [ERROR] Docker daemon did not respond within 90 seconds. Please check Docker Desktop.
    goto :fail
)
echo   ... waiting for Docker daemon (!DOCKER_WAIT!s / 90s) ...
goto :wait_docker

:docker_ready
echo [OK] Docker engine is ready!
echo.

:: 4. Clean previous session (From Scratch)
echo [Step 2/5] Initializing clean state (from scratch)...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1

if exist "%ROOT_DIR%fraud-detection-pipeline-main\storage\checkpoints\consumer_v2" (
    echo   [INFO] Resetting consumer checkpoint for fresh streaming session...
    rmdir /s /q "%ROOT_DIR%fraud-detection-pipeline-main\storage\checkpoints\consumer_v2" >nul 2>&1
)
del /f /q "%ROOT_DIR%producer_stream.log" >nul 2>&1
del /f /q "%ROOT_DIR%consumer_stream.log" >nul 2>&1
del /f /q "%ROOT_DIR%.fraudshield-dashboard.pid" >nul 2>&1
echo [OK] Clean state initialized!
echo.

:: 5. Start all 6 Docker services
echo [Step 3/5] Starting 6-service Docker pipeline:
echo   - zookeeper       (Coordination)
echo   - kafka           (Distributed Event Broker)
echo   - spark-master    (Cluster Orchestration)
echo   - spark-worker    (Executor Compute)
echo   - fraud-producer  (Replaying fraudTest.csv)
echo   - fraud-consumer  (Spark Structured Streaming: LR baseline, RF, XGBoost)
echo.

docker compose -f "%COMPOSE_FILE%" down >nul 2>&1
docker compose -f "%COMPOSE_FILE%" up -d
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to start Docker Compose services!
    goto :fail
)
echo [OK] All 6 Docker services started in background!
echo.

:: 6. Launch Flask Dashboard
echo [Step 4/5] Launching Flask Dashboard Backend on http://localhost:5050 ...
powershell -NoProfile -Command "$p = Start-Process -FilePath python -ArgumentList 'app.py' -WorkingDirectory '%ROOT_DIR%dashboard' -PassThru; Set-Content -LiteralPath '%ROOT_DIR%.fraudshield-dashboard.pid' -Value $p.Id"

echo [INFO] Waiting for Dashboard to respond on port 5050...
powershell -NoProfile -Command "for ($i = 0; $i -lt 25; $i++) { Start-Sleep -Seconds 1; try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5050/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {} }; exit 1"
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Dashboard healthcheck timed out, but process is starting.
) else (
    echo [OK] Dashboard is responsive!
)
echo.

:: 7. Open Browser & Display Status
echo [Step 5/5] Opening Dashboard in default browser...
start "" "http://localhost:5050"
echo.
echo ======================================================================
echo   [SUCCESS] FraudShield 2.0 is fully operational!
echo ======================================================================
echo   Web Dashboard    : http://localhost:5050
echo   Spark Master UI  : http://localhost:8080
echo   Spark Worker UI  : http://localhost:8081
echo   Spark Driver UI  : http://localhost:4040
echo   Kafka Broker     : localhost:9092
echo ======================================================================
echo.
docker compose -f "%COMPOSE_FILE%" ps
echo.
echo ----------------------------------------------------------------------
echo   Streaming live scoring logs below.
echo   Press Ctrl+C to stop following logs (containers remain active).
echo   To completely stop all services, run: stop_project.bat
echo ----------------------------------------------------------------------
echo.
docker compose -f "%COMPOSE_FILE%" logs --follow --tail 25 consumer
exit /b 0

:fail
echo.
echo [FAILED] Startup encountered an error.
pause
exit /b 1
