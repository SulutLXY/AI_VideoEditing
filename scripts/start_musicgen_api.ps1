# MusicGen API 启动脚本（包装器）
# 用法: 右键 -> 使用 PowerShell 运行
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$mgDir = Join-Path $projectRoot "third_party\musicgen"

if (-not (Test-Path $mgDir)) {
    Write-Host "未找到 MusicGen 环境，请先运行:" -ForegroundColor Red
    Write-Host "  python scripts/install_musicgen.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "按任意键退出..."
    $null = [System.Console]::ReadKey()
    exit 1
}

$apiPs1 = Join-Path $mgDir "start_api.ps1"
if (-not (Test-Path $apiPs1)) {
    Write-Host "未找到 MusicGen API 启动脚本，请先运行 install_musicgen.py" -ForegroundColor Red
    Write-Host ""
    Write-Host "按任意键退出..."
    $null = [System.Console]::ReadKey()
    exit 1
}

Write-Host "正在启动 MusicGen API 服务..." -ForegroundColor Green
& powershell -ExecutionPolicy Bypass -File $apiPs1
