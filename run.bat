@echo off
setlocal enabledelayedexpansion
title Attendance System Launcher

echo ============================================================
echo   Webcam Attendance System - One-Click Launcher
echo ============================================================
echo.

:: Get workspace root directory
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: 1. Check Python Virtual Environment
echo [1/3] Checking Python backend environment...
if not exist "%ROOT_DIR%venv\Scripts\activate.bat" (
    echo [WARNING] Virtual environment not found at .\venv.
    echo Creating virtual environment...
    python -m venv "%ROOT_DIR%venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo Installing backend dependencies...
    call "%ROOT_DIR%venv\Scripts\activate.bat"
    pip install -r "%ROOT_DIR%backend\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install backend dependencies.
        pause
        exit /b 1
    )
) else (
    echo [OK] Virtual environment found.
)

:: 2. Check Frontend Dependencies
echo [2/3] Checking Frontend environment...
if not exist "%ROOT_DIR%frontend\node_modules" (
    echo [INFO] Installing frontend dependencies (npm install)...
    cd /d "%ROOT_DIR%frontend"
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies. Ensure Node.js and npm are installed.
        pause
        exit /b 1
    )
    cd /d "%ROOT_DIR%"
) else (
    echo [OK] Node modules found.
)

:: 3. Launching Services
echo [3/3] Starting Backend API and Frontend UI...
echo.

:: Start Backend in dedicated window
start "Attendance System - Backend API" cmd /k "title Attendance System - Backend API && cd /d "%ROOT_DIR%" && call .\venv\Scripts\activate.bat && set PYTHONPATH=%ROOT_DIR%backend && python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload"

:: Start Frontend in dedicated window
start "Attendance System - Frontend UI" cmd /k "title Attendance System - Frontend UI && cd /d "%ROOT_DIR%frontend" && npm run dev"

:: Wait 3 seconds for initial server boot
timeout /t 3 /nobreak >nul

:: Open browser
start http://localhost:5173

echo ============================================================
echo   Attendance System is now running!
echo ============================================================
echo.
echo   * Frontend UI:    http://localhost:5173
echo   * Backend API:     http://127.0.0.1:8000
echo   * API Docs:        http://127.0.0.1:8000/docs
echo   * MJPEG Stream:    http://127.0.0.1:8000/api/stream
echo.
echo   Keep the backend and frontend terminal windows open.
echo   Close those windows or press any key here to exit launcher.
echo ============================================================
pause
