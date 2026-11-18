# One-time setup on Windows (PowerShell).  Run:  .\setup.ps1
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -c "import prophet, duckdb, sklearn, streamlit; print('All libraries OK')"
