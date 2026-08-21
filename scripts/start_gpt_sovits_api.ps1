# GPT-SoVITS API 启动脚本（包装器）
# 用法: 右键 -> 使用 PowerShell 运行
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$gptDir = Join-Path $projectRoot "third_party\GPT-SoVITS"

if (-not (Test-Path $gptDir)) {
    Write-Host "未找到 GPT-SoVITS 环境，请先运行:" -ForegroundColor Red
    Write-Host "  python scripts/install_gpt_sovits.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "按任意键退出..."
    $null = [System.Console]::ReadKey()
    exit 1
}

$apiPs1 = Join-Path $gptDir "start_api.ps1"
if (-not (Test-Path $apiPs1)) {
    # 如果安装脚本没有创建 API 启动脚本，则动态创建一个
    $apiContent = @"
# GPT-SoVITS API 启动脚本
`$scriptDir = Split-Path -Parent `$MyInvocation.MyCommand.Path
. "`$scriptDir\venv\Scripts\Activate.ps1"
Set-Location `$scriptDir
Write-Host "启动 GPT-SoVITS API 服务..." -ForegroundColor Green
Write-Host "地址: http://127.0.0.1:9880" -ForegroundColor Cyan
python api.py
"@
    Set-Content -Path $apiPs1 -Value $apiContent -Encoding UTF8
    Write-Host "已创建 API 启动脚本: $apiPs1" -ForegroundColor Green
}

Write-Host "正在启动 GPT-SoVITS API 服务..." -ForegroundColor Green
& powershell -ExecutionPolicy Bypass -File $apiPs1
