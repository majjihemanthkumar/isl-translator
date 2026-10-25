@echo off
title ISL Real-Time Translator - OpenCV Desktop
cd /d "%~dp0"

echo =========================================================
echo       ISL Real-Time Translator (OpenCV Desktop)
echo =========================================================
echo.

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

echo [ERROR] Python was not found in PATH.
echo Please ensure Python is installed and added to PATH.
echo.
pause
