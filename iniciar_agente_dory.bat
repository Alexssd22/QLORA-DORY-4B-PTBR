@echo off
chcp 65001 >nul
title Dory 1.0 4B - Agente Autonomo Seguro

echo ============================================================
echo      Iniciando Agente Autonomo Dory 4B (Rust Native)
echo      Ferramentas: Arquivos, Pastas e Shell Seguro (S/N)
echo ============================================================
echo.

if exist "crates\dory-agent\target\release\dory-agent.exe" (
    crates\dory-agent\target\release\dory-agent.exe
) else (
    echo Binario Rust de release nao encontrado. Executando versao Python...
    if exist "python\.venv\Scripts\python.exe" (
        python\.venv\Scripts\python.exe python\agente_dory.py
    ) else (
        python python\agente_dory.py
    )
)

pause
