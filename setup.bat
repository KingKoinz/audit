@echo off
setlocal
python -m venv .venv
call .venv\Scripts\activate
pip install -r backend\requirements.txt
python -c "from backend.db import init_db; init_db(); print('DB initialized')"
python -c "print('Next: run ingest then start server')"