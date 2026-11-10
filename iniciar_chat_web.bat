@echo off
chcp 65001 >nul
title Dory 1.0 4B PT-BR - Agente Autônomo & Web Chat
cd /d "%~dp0"

echo ============================================================
echo   🐟 Dory 1.0 4B PT-BR - Agente Autônomo Local & Web Chat
echo   Hardware: NVIDIA GeForce RTX 4070 (BF16 Nativo na GPU)
echo   Recursos Integrados:
echo     - Chat Visual no Navegador (http://localhost:8000)
echo     - Agente Autônomo com Manipulação de Arquivos e Pastas
echo     - Execução Segura de Comandos no Terminal (PowerShell)
echo     - Inspeção de Hardware e Sistema Operacional
echo     - API OpenAI Local (http://localhost:8000/v1)
echo     - Raciocínio com Streaming no Console e no Navegador
echo ============================================================
echo.

:: 1. Localizar Interpretador Python com as Dependências
if exist "python\.venv\Scripts\python.exe" (
    set PY_EXE=python\.venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PY_EXE=.venv\Scripts\python.exe
) else (
    set PY_EXE=python
)

:: 2. Iniciar o Servidor Unificado do Agente Dory
echo [Iniciando] Carregando o modelo Dory 4B e o Agente de Sistema...
echo O navegador abrirá automaticamente em alguns segundos.
echo.
%PY_EXE% python\serve_dory.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O servidor foi encerrado com código %ERRORLEVEL%.
    pause
)
