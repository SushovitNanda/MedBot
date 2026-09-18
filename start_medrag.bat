@echo off
title MedRAG Launcher
cd /d "%~dp0"

echo.
echo  MedRAG - Medical RAG Chatbot Launcher
echo  =====================================
echo.

where powershell >nul 2>&1
if errorlevel 1 (
    echo ERROR: PowerShell is required.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
if errorlevel 1 (
    echo.
    echo ERROR: Startup failed. See messages above.
    pause
    exit /b 1
)

echo.
echo This window can stay open for logs. Press any key to close.
pause >nul
