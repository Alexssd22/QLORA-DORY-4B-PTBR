@echo off
chcp 65001 >nul
title Criar Atalho na Área de Trabalho - Dory 4B PT-BR
cd /d "%~dp0"

echo =================================================================
echo   🐟 Criando Atalho na Área de Trabalho para o Dory 4B PT-BR
echo =================================================================
echo.

powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $desktop = [System.Environment]::GetFolderPath('Desktop'); $sc = $ws.CreateShortcut([System.IO.Path]::Combine($desktop, 'Dory 4B PT-BR.lnk')); $sc.TargetPath = '%~dp0iniciar.bat'; $sc.WorkingDirectory = '%~dp0'; $sc.Description = 'Dory 1.0 4B PT-BR - Agente Autônomo e Web Chat Local'; $sc.Save()"

if %ERRORLEVEL% EQU 0 (
    echo [✓] Atalho 'Dory 4B PT-BR.lnk' criado com sucesso na sua Área de Trabalho!
    echo Agora você pode abrir o Dory a qualquer momento direto pelo Desktop.
) else (
    echo [!] Não foi possível criar o atalho automaticamente via PowerShell.
)

echo.
pause
