@echo off
setlocal
cd /d "%~dp0"
title Dory 1.0 4B PT-BR - Agente Autonomo e Web Chat

echo =================================================================
echo   Dory 1.0 4B PT-BR - Agente Autonomo Local e Web Chat
echo   Execucao 100%% Local, Direta e Sem Softwares Externos
echo =================================================================
echo.

:: 1. Definir o executavel Python
set "PY_CMD=python\venv\Scripts\python.exe"
if exist "python\.venv\Scripts\python.exe" (
    set "PY_CMD=python\.venv\Scripts\python.exe"
    goto :run_check
)

set "PY_CMD=python"
where python >nul 2>&1
if not errorlevel 1 goto :create_venv

echo [!] Python nao foi detectado no sistema.
echo Baixando instalador oficial do Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'"
echo Executando instalador do Python...
start /wait python_installer.exe
del python_installer.exe

:create_venv
if exist "python\.venv\Scripts\python.exe" goto :run_check
echo.
echo [*] Criando ambiente virtual dedicado .venv...
%PY_CMD% -m venv python\.venv
if errorlevel 1 (
    echo [ERRO] Falha ao criar o ambiente virtual Python.
    pause
    exit /b 1
)

:run_check
set "PY_CMD=python\.venv\Scripts\python.exe"
if not exist "%PY_CMD%" set "PY_CMD=python"

:: 2. Executar verificador de ambiente, dependencias e menu interativo
%PY_CMD% python\setup_check.py %*
if errorlevel 1 (
    echo.
    echo [ERRO] Falha na verificacao de dependencias do Dory.
    pause
    exit /b 1
)

:: 3. Iniciar Servidor Unificado Dory (Web Chat + Agente + Ferramentas + API)
echo.
echo [*] Iniciando o servidor do modelo e abrindo o navegador...
echo [*] Pressione Ctrl+C para encerrar o servidor e descarregar o modelo da VRAM.
echo.
%PY_CMD% python\serve_dory.py %*

:: 4. Limpeza e encerramento limpo
echo.
echo [OK] Servidor Dory encerrado. Recursos de GPU liberados.
