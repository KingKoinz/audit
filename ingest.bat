@echo off
setlocal
call .venv\Scripts\activate
python -c "import asyncio; from backend.ingest import ingest_all; print(asyncio.run(ingest_all()))"