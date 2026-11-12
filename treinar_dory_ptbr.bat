@echo off
setlocal
cd /d "%~dp0"
title Treinamento QLoRA Dory 4B PT-BR

echo ============================================================
echo   Iniciando Treinamento QLoRA - Dory 1.0 4B PT-BR
echo   Hardware: NVIDIA GeForce RTX 4070 (12 GB VRAM)
echo   Dataset: 19.260 amostras curadas em Portugues
echo   Matematica: Hiperesfera nGPT com Normalizacao Exata
echo   Meta: Passo 1400 (Remediacao Focada e Correcao Morfologica - 25 Passos)
echo ============================================================
echo.

set "PY_EXE=python"
if exist "python\.venv\Scripts\python.exe" (
    set "PY_EXE=python\.venv\Scripts\python.exe"
)

echo [1/2] Executando Treinamento QLoRA com Mascaramento de Prompt (-100)...
%PY_EXE% python\train_qlora_dory.py
if errorlevel 1 (
    echo.
    echo ERRO durante o treinamento!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [2/2] Treinamento concluido! Iniciando Fusao nGPT dos Pesos...
echo ============================================================
%PY_EXE% python\merge_ngpt_dory.py
if errorlevel 1 (
    echo.
    echo ERRO durante a fusao de pesos!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   SUCESSO TOTAL! O modelo Dory 4B PT-BR esta treinado e mesclado!
echo   Para iniciar, execute: iniciar.bat
echo ============================================================
pause
