@echo off
chcp 65001 >nul

:: LLM-AutoCut Windows 调试启动器
:: 双击此文件会在当前目录生成 start_debug.log，方便排查问题

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

cd /d "%~dp0"

echo 正在启动调试模式，日志会写入 start_debug.log
python launcher.py %* > start_debug.log 2>&1
if %errorlevel% neq 0 (
    echo 启动失败，退出码：%errorlevel%
    echo 请查看 start_debug.log
    start start_debug.log
)
pause
