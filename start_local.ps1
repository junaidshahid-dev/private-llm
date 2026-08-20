# Start the local LLM security-agent UI on http://127.0.0.1:8000 (localhost only).
Set-Location -Path $PSScriptRoot
if (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" start_local.py @args
} else {
    python start_local.py @args
}
