@echo off
setlocal
set PYTHONPATH=.
C:\Python314\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000