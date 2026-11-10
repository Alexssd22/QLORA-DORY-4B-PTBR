@echo off
title Dory 1.0 4B PT-BR - Servidor Local OpenAI API
cd /d "%~dp0"
echo ============================================================
echo Iniciando Servidor Local OpenAI API (Porta 8000)
echo Endpoint: http://localhost:8000/v1
echo Modelo: dory-4b-ptbr
echo ============================================================
echo.

set PY_BIN=python
if exist ".venv\Scripts\python.exe" set PY_BIN=.venv\Scripts\python.exe
if exist "python\.venv\Scripts\python.exe" set PY_BIN=python\.venv\Scripts\python.exe

"%PY_BIN%" python\serve_dory.py %*
pause
