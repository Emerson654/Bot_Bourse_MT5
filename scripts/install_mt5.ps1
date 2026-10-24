# Installation silencieuse du terminal MetaTrader 5 sur le runner Windows.
$ErrorActionPreference = "Stop"
$exe = "C:\Program Files\MetaTrader 5\terminal64.exe"
Invoke-WebRequest "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe" -OutFile "$env:RUNNER_TEMP\mt5setup.exe"
Start-Process "$env:RUNNER_TEMP\mt5setup.exe" -ArgumentList "/auto"
# L'installateur télécharge le terminal en arrière-plan : on attend le fichier final.
for ($i = 0; $i -lt 60 -and -not (Test-Path $exe); $i++) { Start-Sleep 5 }
if (-not (Test-Path $exe)) { throw "terminal64.exe introuvable après 5 min" }
Start-Sleep 20  # fin de copie des fichiers annexes
Get-Process | Where-Object { $_.Name -like "terminal64*" -or $_.Name -like "mt5setup*" } | Stop-Process -Force -ErrorAction SilentlyContinue
"MT5_PATH=$exe" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
Write-Host "MT5 installé"
