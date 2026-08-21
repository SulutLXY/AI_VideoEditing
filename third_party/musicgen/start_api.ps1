# MusicGen API 启动脚本
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir\venv\Scripts\Activate.ps1"
Set-Location $scriptDir
Write-Host "启动 MusicGen API 服务..." -ForegroundColor Green
Write-Host "地址: http://127.0.0.1:9881" -ForegroundColor Cyan
python api_server.py
