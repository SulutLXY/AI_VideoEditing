@echo off
chcp 65001 >nul

:: LLM-AutoCut Windows 一键启动器
:: 双击此文件即可启动 Web UI

title LLM-AutoCut 启动器

:: 设置 Python 使用 UTF-8 模式，避免中文输出乱码或编码错误
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ========================================
echo  LLM-AutoCut 智能剪辑工作台
echo  正在启动，请稍候...
echo ========================================
echo.

:: 切换到脚本所在目录，避免路径问题
cd /d "%~dp0"

:: 检查 launcher.py 是否存在
if not exist "launcher.py" (
    echo.
    echo 错误：找不到 launcher.py，请确保 start.bat 与 launcher.py 在同一目录
    echo.
    pause
    exit /b 1
)

:: 优先使用项目内置的 Python 3.11 虚拟环境（含 CUDA PyTorch + 本地模型依赖）
set PY_CMD=
if exist ".venv311\Scripts\python.exe" (
    set PY_CMD=.venv311\Scripts\python.exe
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_CMD=python
    ) else (
        python3 --version >nul 2>&1
        if %errorlevel% equ 0 (
            set PY_CMD=python3
        )
    )
)

if "%PY_CMD%"=="" (
    echo.
    echo 错误：未检测到 Python，请先安装 Python 3.9+
    echo 下载地址：https://www.python.org/downloads/
    echo 安装时请务必勾选 "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo 使用 Python: %PY_CMD%
echo.

:: 启动 launcher.py，出错时继续保留窗口
%PY_CMD% launcher.py %*
if %errorlevel% neq 0 (
    echo.
    echo 启动器异常退出，退出码：%errorlevel%
    echo 请检查上方错误信息，或双击 start_debug.bat 查看详细日志
    echo.
    pause
    exit /b %errorlevel%
)

pause
