@echo off
echo [Skill] Limpando pastas __pycache__ e arquivos .pyc temporarios...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo [Skill] Concluido com sucesso!
