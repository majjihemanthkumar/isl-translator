@echo off
title ISL Real-Time Translator
cd /d "%~dp0"

set PORT=8000
if not "%1"=="" (
    if /i "%1"=="desktop" goto :run_desktop
    if /i "%1"=="opencv" goto :run_desktop
    set PORT=%1
)

set URL=http://localhost:%PORT%

echo =========================================================
echo            ISL Real-Time Translator
echo =========================================================
echo.
echo Starting local web server on port %PORT%...
echo Note: A local web server is required so the browser can
echo       load ONNX models and JSON labels without CORS errors.
echo.

:: Detect Python or Node.js runtime
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found Python.
    start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 800; Start-Process '%URL%'"
    echo [OK] Opening %URL% in your default browser...
    echo.
    echo ---------------------------------------------------------
    echo Server running at %URL%
    echo Press Ctrl+C to stop the server.
    echo ---------------------------------------------------------
    echo.
    if exist "%~dp0server.py" (
        python "%~dp0server.py" %PORT%
    ) else (
        python -m http.server %PORT%
    )
    goto :eof
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found Python launcher (py).
    start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 800; Start-Process '%URL%'"
    echo [OK] Opening %URL% in your default browser...
    echo.
    echo ---------------------------------------------------------
    echo Server running at %URL%
    echo Press Ctrl+C to stop the server.
    echo ---------------------------------------------------------
    echo.
    if exist "%~dp0server.py" (
        py "%~dp0server.py" %PORT%
    ) else (
        py -m http.server %PORT%
    )
    goto :eof
)

where npx >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] Found Node.js / npx.
    start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Milliseconds 800; Start-Process '%URL%'"
    echo [OK] Opening %URL% in your default browser...
    echo.
    echo ---------------------------------------------------------
    echo Server running at %URL%
    echo Press Ctrl+C to stop the server.
    echo ---------------------------------------------------------
    echo.
    npx -y http-server -p %PORT% -c-1
    goto :eof
)

echo [ERROR] Neither Python nor Node.js was found in your PATH.
echo Please install Python (https://www.python.org/) or Node.js (https://nodejs.org/) to run the local server.
echo.
pause
goto :eof

:run_desktop
echo Starting Desktop OpenCV live testing...
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python "%~dp0live_testing.py"
    goto :eof
)
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py "%~dp0live_testing.py"
    goto :eof
)
echo [ERROR] Python not found in PATH.
pause
goto :eof
