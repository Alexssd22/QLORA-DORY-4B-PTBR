@echo off
setlocal
cd /d "%~dp0"
title Criar Atalho na Area de Trabalho - Dory 4B PT-BR

echo =================================================================
echo   Criando Atalho na Area de Trabalho para o Dory 4B PT-BR
echo =================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $desk = [System.Environment]::GetFolderPath('Desktop'); $sc = $ws.CreateShortcut((Join-Path $desk 'Dory 4B PT-BR.lnk')); $sc.TargetPath = '%~dp0iniciar.bat'; $sc.WorkingDirectory = '%~dp0'; $sc.Description = 'Dory 1.0 4B PT-BR - Agente Autonomo e Web Chat Local'; $sc.Save()"

if exist "%USERPROFILE%\Desktop\Dory 4B PT-BR.lnk" (
    echo.
    echo [OK] Atalho 'Dory 4B PT-BR' criado com sucesso na sua Area de Trabalho!
    echo Agora voce pode abrir o Dory a qualquer momento direto pelo Desktop.
) else (
    echo.
    echo [!] Nao foi possivel criar o atalho automaticamente via PowerShell.
)

echo.
pause
