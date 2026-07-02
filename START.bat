@echo off
title AI Trading Analyst - Indian Markets
echo.
echo  ===================================================
echo   AI Trading Analyst - Indian Markets
echo   Version 1.0.0 - Simulation Mode - Zero Cloud
echo  ===================================================
echo.
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

python -c "import fastapi" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting application...
python launch.py

pause
