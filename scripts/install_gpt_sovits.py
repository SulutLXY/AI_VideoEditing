#!/usr/bin/env python3
"""
GPT-SoVITS 声音克隆环境安装器
用法: python scripts/install_gpt_sovits.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
THIRD_PARTY = PROJECT_ROOT / "third_party"
GPTSOVITS_DIR = THIRD_PARTY / "GPT-SoVITS"

print("=" * 50)
print("  GPT-SoVITS 声音克隆环境安装器")
print("=" * 50)
print()

# ------------------------------------------------------------------
# 1. 检查前提条件
# ------------------------------------------------------------------
print("[1/5] 检查前提条件...")

# Python 版本
try:
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    py_version = result.stdout.strip() or result.stderr.strip()
    print(f"  Python: {py_version}")
except:
    print("错误: 未找到 Python")
    sys.exit(1)

# Git
try:
    result = subprocess.run(["git", "--version"], capture_output=True, text=True)
    print(f"  Git: {result.stdout.strip()}")
except:
    print("错误: 未找到 Git")
    sys.exit(1)

# CUDA（可选）
nvcuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
if nvcuda.exists():
    print("  检测到 NVIDIA GPU/CUDA")
else:
    print("  未检测到 CUDA，将使用 CPU 模式")

# ------------------------------------------------------------------
# 2. 克隆 GPT-SoVITS
# ------------------------------------------------------------------
print()
print("[2/5] 下载 GPT-SoVITS 代码...")

THIRD_PARTY.mkdir(parents=True, exist_ok=True)

if (GPTSOVITS_DIR / ".git").exists():
    print("  已存在，更新代码...")
    subprocess.run(["git", "pull"], cwd=GPTSOVITS_DIR)
else:
    print("  从 Gitee 克隆...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", "https://gitee.com/RVC-Boss/GPT-SoVITS.git", str(GPTSOVITS_DIR)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Gitee 失败，尝试 GitHub...")
        result = subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/RVC-Boss/GPT-SoVITS.git", str(GPTSOVITS_DIR)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  克隆失败: {result.stderr}")
            sys.exit(1)

print("  代码就绪")

# ------------------------------------------------------------------
# 3. 创建虚拟环境
# ------------------------------------------------------------------
print()
print("[3/5] 创建虚拟环境...")

venv_dir = GPTSOVITS_DIR / "venv"
if not venv_dir.exists():
    subprocess.run(["python", "-m", "venv", str(venv_dir)], check=True)
    print(f"  虚拟环境已创建: {venv_dir}")
else:
    print("  虚拟环境已存在")

# pip 路径
pip = venv_dir / "Scripts" / "pip.exe"
python_venv = venv_dir / "Scripts" / "python.exe"

# ------------------------------------------------------------------
# 4. 安装依赖
# ------------------------------------------------------------------
print()
print("[4/5] 安装依赖...")

subprocess.run([str(pip), "install", "--upgrade", "pip"], check=False)

# PyTorch CPU
print("  安装 PyTorch CPU...")
subprocess.run([str(pip), "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"], check=False)

# 基础依赖
print("  安装基础依赖...")
for pkg in ["numpy", "scipy", "librosa", "soundfile", "transformers", "ffmpeg-python", "gradio", "modelscope"]:
    subprocess.run([str(pip), "install", pkg], check=False)

# 额外依赖（失败则跳过）
print("  安装额外依赖（可选）...")
for pkg in ["LangSegment", "pyopenjtalk"]:
    subprocess.run([str(pip), "install", pkg], check=False)

print("  依赖安装完成")

# ------------------------------------------------------------------
# 5. 下载预训练模型
# ------------------------------------------------------------------
print()
print("[5/5] 下载预训练模型（约 2-3GB，请等待）...")

try:
    from modelscope import snapshot_download
    print("  从 ModelScope 下载...")
    downloaded = snapshot_download("RVC-Boss/GPT-SoVITS")
    print(f"  下载完成: {downloaded}")
    
    # 复制到项目目录
    target = GPTSOVITS_DIR / "pretrained_models"
    if Path(downloaded).exists() and not target.exists():
        shutil.copytree(downloaded, target)
        print(f"  已复制到: {target}")
except Exception as e:
    print(f"  模型下载失败（非致命）: {e}")
    print("  将在首次使用时自动下载")

# ------------------------------------------------------------------
# 6. 创建启动脚本
# ------------------------------------------------------------------
print()
print("创建启动脚本...")

# API 启动脚本
api_ps1 = """# GPT-SoVITS WebUI 启动脚本
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir\\venv\\Scripts\\Activate.ps1"
Set-Location $scriptDir
Write-Host "启动 GPT-SoVITS WebUI..." -ForegroundColor Green
Write-Host "请在浏览器中打开显示的地址" -ForegroundColor Cyan
python webui.py
"""

with open(GPTSOVITS_DIR / "start_webui.ps1", "w", encoding="utf-8") as f:
    f.write(api_ps1)

print("  start_webui.ps1 已创建")

# ------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------
print()
print("=" * 50)
print("  GPT-SoVITS 安装完成！")
print("=" * 50)
print()
print("使用方式:")
print()
print("1. 启动 WebUI（图形界面，用于训练和推理）:")
print(f"   cd {GPTSOVITS_DIR}")
print(r"   .\start_webui.ps1")
print()
print("2. 训练克隆音色步骤:")
print("   a. 在 WebUI 上传 5-30 秒样本音频")
print("   b. 点击 '开始语音切割' -> '开始语音文本校对'")
print("   c. 在 '1A-训练集格式化' 中一键三连")
print("   d. 在 '1B-微调训练' 中训练 SoVITS + GPT")
print()
print("3. 主项目调用:")
print("   python main.py tts --text '你好' --voice <音色名>")
print()
print("注意:")
print("- 训练需要约 10-30 分钟")
print("- 样本音频建议 5-30 秒，音质清晰")
print()
