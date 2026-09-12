@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
if errorlevel 1 echo O benchmark encontrou um erro. Consulte a mensagem acima.
pause
