@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Crie o ambiente primeiro: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt -r requirements-build.txt
    exit /b 1
)
".venv\Scripts\python.exe" tools\build_installer.py %*
