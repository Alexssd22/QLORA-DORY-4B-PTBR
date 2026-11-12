@echo off
setlocal
cd /d "%~dp0"
title Criar Atalho na Area de Trabalho - Dory 4B PT-BR

echo =================================================================
echo   Criando Atalho na Area de Trabalho para o Dory 4B PT-BR
echo =================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $desk = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop); $target = Join-Path (Get-Location) 'iniciar.bat'; $lnk = Join-Path $desk 'Dory 4B PT-BR.lnk'; $sc = $ws.CreateShortcut($lnk); $sc.TargetPath = $target; $sc.WorkingDirectory = (Get-Location).Path; $sc.Description = 'Dory 1.0 4B PT-BR - Agente Autonomo e Web Chat Local'; $sc.Save(); if (Test-Path $lnk) { Write-Host '[OK] Atalho criado com sucesso na sua Area de Trabalho!' -ForegroundColor Green; Write-Host ('Caminho: ' + $lnk) -ForegroundColor Cyan; Write-Host 'Agora voce pode abrir o Dory a qualquer momento direto pelo Desktop.' -ForegroundColor White; } else { Write-Host '[!] Falha ao criar atalho automaticamente.' -ForegroundColor Yellow; }"

echo.
pause
