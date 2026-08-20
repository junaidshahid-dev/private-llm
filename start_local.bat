@echo off
REM Start the local LLM security-agent UI on http://127.0.0.1:8000  (localhost only).
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start_local.py %*
) else (
  python start_local.py %*
)
pause
