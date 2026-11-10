@echo off
title Dory 1.0 4B PT-BR - Chat Direto
cd /d "%~dp0"
echo ============================================================
echo Iniciando Chat Interativo com Dory 1.0 4B PT-BR (Uncensored)
echo ============================================================
echo.

set PY_BIN=python
if exist ".venv\Scripts\python.exe" set PY_BIN=.venv\Scripts\python.exe
if exist "python\.venv\Scripts\python.exe" set PY_BIN=python\.venv\Scripts\python.exe

"%PY_BIN%" python\chat_dory.py %*
pause
