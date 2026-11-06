@echo off
setlocal
title Start FraudShield 2.0
cd /d "%~dp0"
set "FRAUDSHIELD_ROOT=%~dp0"
echo ======================================================================
echo   FraudShield 2.0 - Real Dataset Streaming and Dashboard
echo ======================================================================
echo.

where docker >nul 2>&1
if errorlevel 1 (echo ERROR: Install Docker Desktop and make sure docker is on PATH. & goto :fail)
docker compose version >nul 2>&1
if errorlevel 1 (echo ERROR: Docker Compose is not available. & goto :fail)
where python >nul 2>&1
if errorlevel 1 (echo ERROR: Python is not on PATH. Install Python and try again. & goto :fail)
python -c "import flask, flask_cors, pandas, pyarrow" >nul 2>&1
if errorlevel 1 (echo ERROR: Dashboard dependencies are missing. Run: python -m pip install flask flask-cors pandas pyarrow & goto :fail)
if not exist "fraudTest.csv" (echo ERROR: fraudTest.csv is missing from the project folder. & goto :fail)
if not exist "fraud-detection-pipeline-main\models\xgboost\xgb_model.json" (echo ERROR: The 2.0 XGBoost model is missing. & goto :fail)
if not exist "fraud-detection-pipeline-main\models\linear_regression\data\*.parquet" (echo ERROR: Linear regression model data is missing. Train or restore the 2.0 models. & goto :fail)
if not exist "fraud-detection-pipeline-main\models\random_forest\data\*.parquet" (echo ERROR: Random forest model data is missing. Train or restore the 2.0 models. & goto :fail)
if not exist "fraud-detection-pipeline-main\models\preprocessor\stages\*" (echo ERROR: The fitted preprocessor is missing. Train or restore the 2.0 models. & goto :fail)

rem Avoid starting another dashboard or taking over an unrelated service.
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue) { exit 1 }" >nul 2>&1
if errorlevel 1 (echo ERROR: Port 5050 is already in use. Stop the existing dashboard or free the port. & goto :fail)

echo [Step 1/4] Checking Docker Desktop...
docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready
if not exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (echo ERROR: Docker Desktop was not found in its usual location. Start it manually and retry. & goto :fail)
echo Starting Docker Desktop and waiting for the engine...
start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
set /a DOCKER_ATTEMPTS=0
:wait_docker
timeout /t 4 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto :docker_ready
set /a DOCKER_ATTEMPTS+=1
if %DOCKER_ATTEMPTS% lss 90 goto :wait_docker
echo ERROR: Docker did not become ready within 6 minutes. Check Docker Desktop and retry.
goto :fail

:docker_ready
echo [OK] Docker engine is ready.
echo.
echo [Step 2/4] Building and starting the six Docker services...
echo The first build can take several minutes.
docker compose -f "%~dp0fraud-detection-pipeline-main\docker-compose.yml" up -d --build
if errorlevel 1 (echo ERROR: Docker Compose failed. See the output above. & goto :fail)

echo.
echo [Step 3/4] Starting the dashboard...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $p = Start-Process -FilePath python -ArgumentList ([char]34 + (Join-Path $env:FRAUDSHIELD_ROOT 'dashboard\app.py') + [char]34) -WorkingDirectory (Join-Path $env:FRAUDSHIELD_ROOT 'dashboard') -WindowStyle Hidden -PassThru; Set-Content -LiteralPath (Join-Path $env:FRAUDSHIELD_ROOT '.fraudshield-dashboard.pid') -Value $p.Id; for ($i=0; $i -lt 20; $i++) { Start-Sleep -Seconds 1; try { $r=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5050/api/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; if ($p.HasExited) { break } }; exit 1"
if errorlevel 1 (echo ERROR: Dashboard did not become ready. Run python dashboard\app.py to diagnose; Docker services may still be running. & goto :fail)
start "" "http://localhost:5050"
echo Dashboard is ready at http://localhost:5050
echo Existing predictions and checkpoints are preserved.
echo.
echo [Step 4/4] Showing producer and consumer logs. Press Ctrl+C to stop viewing logs.
echo Use stop_fraudshield_2.bat to stop the project.
docker compose -f "%~dp0fraud-detection-pipeline-main\docker-compose.yml" logs --follow --tail 20 producer consumer
exit /b 0

:fail
pause
exit /b 1
