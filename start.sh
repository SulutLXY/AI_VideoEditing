#!/usr/bin/env bash
# LLM-AutoCut Linux/macOS 一键启动器
# 运行方式: ./start.sh

set -e

cd "$(dirname "$0")"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

echo "========================================"
echo " LLM-AutoCut 智能剪辑工作台"
echo " 正在启动，请稍候..."
echo "========================================"
echo

# 优先使用项目内置的 Python 3.11 虚拟环境
if [ -f ".venv311/bin/python" ]; then
    PY_CMD=".venv311/bin/python"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    echo "错误：未检测到 python3，请先安装 Python 3.9+"
    exit 1
fi

$PY_CMD --version

# 启动 launcher
$PY_CMD launcher.py "$@"
