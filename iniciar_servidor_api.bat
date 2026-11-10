@echo off
title Dory 1.0 4B PT-BR - Chat no Navegador & Servidor API
cd /d "%~dp0"
echo ============================================================
echo Iniciando Dory 1.0 4B PT-BR
echo A interface abrira automaticamente no navegador:
echo   --^> http://localhost:8000
echo Endpoint API OpenAI (para LM Studio/Continue/Cursor):
echo   --^> http://localhost:8000/v1
echo ============================================================
echo.

set PY_BIN=python
if exist ".venv\Scripts\python.exe" set PY_BIN=.venv\Scripts\python.exe
if exist "python\.venv\Scripts\python.exe" set PY_BIN=python\.venv\Scripts\python.exe

"%PY_BIN%" python\serve_dory.py %*
pause
