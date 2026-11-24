@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DiffSight - launcher
if not exist "logs" mkdir "logs"
set "LOG=%~dp0logs\start.log"
rem A new start overwrites the previous start log. The running product writes its own log: logs\nexascan.log
> "%LOG%" echo DiffSight start log
call :log "Start requested in %~dp0"

if not exist "venv\Scripts\python.exe" (
  call :log "ERROR: not set up yet. Run setup.bat first."
  goto :fail
)
if not exist "frontend\node_modules" (
  call :log "ERROR: web UI not installed. Run setup.bat first."
  goto :fail
)

rem Ports (rarely used; nothing else is stopped or changed)
set "NEXASCAN_PORT=38219"
set "NEXASCAN_BACKEND=http://127.0.0.1:%NEXASCAN_PORT%"
set "NEXASCAN_LOG_DIR=%~dp0logs"

netstat -ano | findstr /R /C:":%NEXASCAN_PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  call :log "ERROR: port %NEXASCAN_PORT% is already in use (maybe DiffSight is already running). Close that program or change NEXASCAN_PORT in start.bat."
  goto :fail
)
call :log "Port %NEXASCAN_PORT% is free."

call :log "Starting the backend window (application log: logs\nexascan.log) ..."
start "DiffSight BACKEND (port %NEXASCAN_PORT%)" cmd /k ""%~dp0venv\Scripts\python.exe" "%~dp0run.py""
call :log "Starting the frontend window (port 34517) ..."
start "DiffSight FRONTEND (port 34517)" cmd /k "cd /d "%~dp0frontend" && npm run dev"

rem Wait (up to 40 s) until the backend answers, and record the result
set /a TRIES=0
:wait
set /a TRIES+=1
set "CODE="
curl -s -o nul -w "%%{http_code}" "http://127.0.0.1:%NEXASCAN_PORT%/api/health" > "%TEMP%\nexascan_health.txt" 2>nul
set /p CODE=<"%TEMP%\nexascan_health.txt"
if "%CODE%"=="200" goto :up
if %TRIES% GEQ 40 goto :down
timeout /t 1 /nobreak >nul
goto :wait

:up
call :log "Backend is up (HTTP 200 after %TRIES% s): http://127.0.0.1:%NEXASCAN_PORT%"
call :log "Opening http://localhost:34517 in the browser."
start "" "http://localhost:34517"
call :log "START OK. Close the two windows to stop DiffSight."
echo.
echo Logs: %~dp0logs  (start.log = this launcher, nexascan.log = the running product)
timeout /t 6 >nul
exit /b 0

:down
call :log "WARNING: the backend did not answer within 40 s. See the BACKEND window and logs\nexascan.log."
goto :fail

:fail
call :log "START FAILED. Log file: %LOG%"
powershell -NoProfile -Command "Get-Content -Tail 10 -LiteralPath '%LOG%'"
pause
exit /b 1

:try_tess
if defined TESS_EXE exit /b 0
if not exist "%~1" exit /b 0
"%~1" --version >nul 2>&1
if errorlevel 1 (
  call :log "Found %~1 but it does not run - skipping it."
  exit /b 0
)
set "TESS_EXE=%~1"
exit /b 0

:log
echo %~1
>> "%LOG%" echo [%date% %time%] %~1
exit /b 0
