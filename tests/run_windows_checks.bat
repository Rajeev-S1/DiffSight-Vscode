@echo off
rem Run on WINDOWS after setup.bat. Executes the tests that need real Windows file locking and the real Tesseract install.
rem Result: logs\windows_checks.log  (also printed). Every line says PASSED / FAILED; nothing is skipped silently.
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
set "LOG=%CD%\logs\windows_checks.log"
> "%LOG%" echo DiffSight Windows checks - %date% %time%
if not exist "venv\Scripts\python.exe" ( echo Run setup.bat first. & exit /b 1 )
set "PY=venv\Scripts\python.exe"
set "FAILS=0"

echo [1/3] Real Windows file locking + progress writes (WinError 5 fix) ...
%PY% -m pytest tests\test_windows_real.py tests\test_progress_write.py -v -p no:cacheprovider >> "%LOG%" 2>&1
if errorlevel 1 ( echo     FAILED - see %LOG% & set /a FAILS+=1 ) else ( echo     PASSED )

echo [2/3] Tesseract found and working (PATH, common folders or tools\Tesseract-OCR) ...
%PY% -m nexascan.tesseract_locator >> "%LOG%" 2>&1
if errorlevel 1 ( echo     FAILED - see %LOG% & set /a FAILS+=1 ) else ( echo     PASSED )

echo [3/3] OCR really reads text, and a scan comparison completes ...
%PY% -m nexascan.tesseract_locator --check >> "%LOG%" 2>&1
if errorlevel 1 ( echo     FAILED - see %LOG% & set /a FAILS+=1 ) else ( echo     PASSED )
%PY% -m pytest tests\test_realpack.py -q -p no:cacheprovider >> "%LOG%" 2>&1
if errorlevel 1 ( echo     real-pack comparisons FAILED - see %LOG% & set /a FAILS+=1 ) else ( echo     real-pack comparisons PASSED )

echo.
if "%FAILS%"=="0" ( echo ALL WINDOWS CHECKS PASSED ) else ( echo %FAILS% CHECK^(S^) FAILED - send logs\windows_checks.log )
exit /b %FAILS%
