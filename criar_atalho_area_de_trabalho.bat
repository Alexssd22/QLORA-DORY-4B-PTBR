@echo off
setlocal
cd /d "%~dp0"
title Criar Atalho na Area de Trabalho - Dory 4B PT-BR

echo =================================================================
echo   Criando Atalho na Area de Trabalho para o Dory 4B PT-BR
echo =================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$desk = [System.Environment]::GetFolderPath('Desktop'); $lnkPath = Join-Path $desk 'Dory 4B PT-BR.lnk'; $ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut($lnkPath); $sc.TargetPath = (Join-Path $pwd.Path 'iniciar.bat'); $sc.WorkingDirectory = $pwd.Path; $sc.Description = 'Dory 1.0 4B PT-BR - Agente Autonomo e Web Chat Local'; $sc.Save(); if (Test-Path $lnkPath) { Write-Host '[OK] Atalho criado com sucesso em:' $lnkPath -ForegroundColor Green; Write-Host 'Agora voce pode abrir o Dory a qualquer momento direto pelo Desktop.' -ForegroundColor Cyan; } else { Write-Host '[!] Nao foi possivel criar o atalho automaticamente.' -ForegroundColor Yellow; }"

echo.
pause
