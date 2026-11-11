@echo off
chcp 65001 >nul
title Treinamento QLoRA Dory 4B PT-BR (Passo 200 -> 400)

echo ============================================================
echo   Iniciando Treinamento QLoRA - Dory 1.0 4B PT-BR
echo   Hardware: NVIDIA GeForce RTX 4070 (12 GB VRAM)
echo   Dataset: 18.386 amostras curadas em Português (com Rust First)
echo   Matemática: Hiperesfera nGPT com Normalização Exata
echo   Meta: Passo 400 (Convergência com Resfriamento do LR)
echo ============================================================
echo.

if exist "python\.venv\Scripts\python.exe" (
    set PY_EXE=python\.venv\Scripts\python.exe
) else (
    set PY_EXE=python
)

echo [1/2] Executando Treinamento QLoRA com Mascaramento de Prompt (-100)...
%PY_EXE% python\train_qlora_dory.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERRO durante o treinamento!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo [2/2] Treinamento concluído! Iniciando Fusão nGPT dos Pesos...
echo ============================================================
%PY_EXE% python\merge_ngpt_dory.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERRO durante a fusão de pesos!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   SUCESSO TOTAL! O modelo Dory 4B PT-BR está treinado e mesclado!
echo   Para testar, execute: iniciar_chat_web.bat
echo ============================================================
pause
