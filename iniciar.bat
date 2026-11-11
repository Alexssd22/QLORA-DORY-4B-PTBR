@echo off
chcp 65001 >nul
title 🐟 Dory 1.0 4B PT-BR — Agente Autônomo & Web Chat
cd /d "%~dp0"

echo =================================================================
echo   🐟 Dory 1.0 4B PT-BR — Agente Autônomo Local & Web Chat
echo   Execução 100%% Local, Direta e Sem Softwares Externos
echo =================================================================
echo.

:: 1. Verificar se o Python está presente
set PY_CMD=
if exist "python\.venv\Scripts\python.exe" (
    set PY_CMD=python\.venv\Scripts\python.exe
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=python
    ) else (
        echo [!] Python não foi detectado no sistema.
        echo Baixando e preparando instalador oficial do Python...
        powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'"
        echo Executando instalador do Python. Por favor, marque 'Add python.exe to PATH'...
        start /wait python_installer.exe
        del python_installer.exe
        set PY_CMD=python
    )
)

:: 2. Criar ambiente virtual se ainda não existir
if not exist "python\.venv\Scripts\python.exe" (
    echo.
    echo [*] Criando ambiente virtual dedicado (.venv)...
    %PY_CMD% -m venv python\.venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERRO] Falha ao criar o ambiente virtual Python.
        pause
        exit /b 1
    )
    set PY_CMD=python\.venv\Scripts\python.exe
) else (
    set PY_CMD=python\.venv\Scripts\python.exe
)

:: 3. Executar verificador de ambiente, dependências e menu interativo
%PY_CMD% python\setup_check.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Falha na verificação de dependências do Dory.
    pause
    exit /b 1
)

:: 4. Iniciar Servidor Unificado Dory (Web Chat + Agente + Ferramentas + API)
echo.
echo [*] Iniciando o servidor do modelo e abrindo o navegador...
echo [*] Pressione Ctrl+C para encerrar o servidor e descarregar o modelo da VRAM.
echo.
%PY_CMD% python\serve_dory.py %*

:: 5. Limpeza e encerramento limpo
echo.
echo [✓] Servidor Dory encerrado. Recursos de GPU liberados.
