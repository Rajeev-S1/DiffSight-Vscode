@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title DiffSight - setup
if not exist "logs" mkdir "logs"
set "LOG=%~dp0logs\setup.log"
rem A new run overwrites the previous setup log (so a successful setup leaves a clean log).
> "%LOG%" echo DiffSight setup log
call :log "Setup started in %~dp0"
call :log "Windows: %OS% ; user: %USERNAME% ; computer: %COMPUTERNAME%"
echo.

rem ---- 1. Python ------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )
if not defined PY (
  call :log "ERROR: Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/ and tick Add python.exe to PATH."
  goto :fail
)
%PY% --version >> "%LOG%" 2>&1
call :log "Python command: %PY%"
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 call :log "WARNING: Python older than 3.11. DiffSight works, but 3.11 or newer is recommended."

rem ---- 2. Node.js -----------------------------------------------------
where npm >nul 2>&1
if errorlevel 1 (
  call :log "ERROR: Node.js was not found. Install the LTS version from https://nodejs.org/ and run setup.bat again."
  goto :fail
)
node --version >> "%LOG%" 2>&1
call npm --version >> "%LOG%" 2>&1
call :log "Node.js and npm found."

rem ---- 3b. LibreOffice (only needed for Word files)
set "LO="
where soffice >nul 2>&1 && set "LO=1"
if exist "%ProgramFiles%\LibreOffice\program\soffice.exe" set "LO=1"
if exist "%ProgramFiles(x86)%\LibreOffice\program\soffice.exe" set "LO=1"
if defined LO (
  call :log "LibreOffice found - Word files (.docx, .doc) are supported."
) else (
  call :log "WARNING: LibreOffice not found. Word files (.docx, .doc) cannot be compared until it is installed: winget install -e --id TheDocumentFoundation.LibreOffice  (PDF, JPG and PNG work without it)."
)

rem ---- 4. Virtual environment + Python packages -----------------------
if not exist "venv\Scripts\python.exe" (
  call :log "[1/3] Creating the virtual environment in .\venv ..."
  %PY% -m venv venv >> "%LOG%" 2>&1
  if errorlevel 1 ( call :log "ERROR: could not create the virtual environment." & goto :fail )
) else (
  call :log "[1/3] Virtual environment already exists - reusing it."
)
call :log "[2/3] Installing Python packages (first time can take several minutes; details are written to the log) ..."
"venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
"venv\Scripts\python.exe" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 ( call :log "ERROR: pip install failed. Check the internet connection and run setup.bat again." & goto :fail )
call :log "Python packages installed."

rem ---- 5. Web UI (React) ----------------------------------------------
call :log "[3/3] Installing and building the web UI ..."
pushd frontend
call npm install --no-audit --no-fund >> "%LOG%" 2>&1
if errorlevel 1 ( popd & call :log "ERROR: npm install failed." & goto :fail )
call npm run build >> "%LOG%" 2>&1
if errorlevel 1 ( popd & call :log "ERROR: npm run build failed." & goto :fail )
popd

call :log "SETUP COMPLETED SUCCESSFULLY. Next: double-click start.bat"
echo.
echo Log file: %LOG%
pause
exit /b 0

:fail
call :log "SETUP FAILED. Last lines of the log:"
powershell -NoProfile -Command "Get-Content -Tail 15 -LiteralPath '%LOG%'"
echo.
echo Full log: %LOG%
pause
exit /b 1

:log
echo %~1
>> "%LOG%" echo [%date% %time%] %~1
exit /b 0
