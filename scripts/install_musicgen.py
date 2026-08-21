#!/usr/bin/env python3
"""
MusicGen AI音乐生成环境安装器
用法: python scripts/install_musicgen.py
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
THIRD_PARTY = PROJECT_ROOT / "third_party"
MUSICGEN_DIR = THIRD_PARTY / "musicgen"

print("=" * 50)
print("  MusicGen AI音乐生成环境安装器")
print("=" * 50)
print()

# ------------------------------------------------------------------
# 1. 检查前提条件
# ------------------------------------------------------------------
print("[1/4] 检查前提条件...")

try:
    result = subprocess.run(["python", "--version"], capture_output=True, text=True)
    print(f"  Python: {result.stdout.strip() or result.stderr.strip()}")
except:
    print("错误: 未找到 Python")
    sys.exit(1)

# 磁盘空间
import shutil
total, used, free = shutil.disk_usage(PROJECT_ROOT)
free_gb = free // (2**30)
print(f"  磁盘剩余: {free_gb} GB")
if free_gb < 5:
    print("警告: 磁盘空间不足，需要约 3GB")

# ------------------------------------------------------------------
# 2. 创建虚拟环境
# ------------------------------------------------------------------
print()
print("[2/4] 创建虚拟环境...")

THIRD_PARTY.mkdir(parents=True, exist_ok=True)
MUSICGEN_DIR.mkdir(parents=True, exist_ok=True)

venv_dir = MUSICGEN_DIR / "venv"
if not venv_dir.exists():
    subprocess.run(["python", "-m", "venv", str(venv_dir)], check=True)
    print(f"  虚拟环境已创建")
else:
    print("  虚拟环境已存在")

pip = venv_dir / "Scripts" / "pip.exe"
python_venv = venv_dir / "Scripts" / "python.exe"

# ------------------------------------------------------------------
# 3. 安装依赖
# ------------------------------------------------------------------
print()
print("[3/4] 安装 MusicGen 依赖...")

subprocess.run([str(pip), "install", "--upgrade", "pip"], check=False)

# PyTorch CPU
print("  安装 PyTorch CPU...")
subprocess.run([str(pip), "install", "torch==2.1.0", "torchvision==0.16.0", "torchaudio==2.1.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"], check=False)

# audiocraft
print("  安装 audiocraft...")
try:
    subprocess.run([str(pip), "install", "audiocraft"], check=True)
    print("  audiocraft 安装成功")
except:
    print("  pip 安装失败，尝试从源码...")
    os.chdir(MUSICGEN_DIR)
    if not (MUSICGEN_DIR / "audiocraft").exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/facebookresearch/audiocraft.git"], check=False)
    os.chdir(MUSICGEN_DIR / "audiocraft")
    subprocess.run([str(pip), "install", "-e", "."], check=False)

# 额外依赖
print("  安装额外依赖...")
for pkg in ["transformers", "einops", "flask"]:
    subprocess.run([str(pip), "install", pkg], check=False)

print("  依赖安装完成")

# ------------------------------------------------------------------
# 4. 预下载模型 + 创建脚本
# ------------------------------------------------------------------
print()
print("[4/4] 下载模型并创建脚本...")

# 预下载 small 模型
try:
    print("  预下载 MusicGen-small 模型 (~2.8GB)...")
    sys.path.insert(0, str(venv_dir / "Lib" / "site-packages"))
    from audiocraft.models import MusicGen
    model = MusicGen.get_pretrained("small")
    print("  模型就绪")
except Exception as e:
    print(f"  预下载跳过（将在首次使用时下载）: {e}")

# 创建 API 服务端
api_server = '''from flask import Flask, request, jsonify, send_file
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

app = Flask(__name__)

MODEL_NAME = os.environ.get("MUSICGEN_MODEL", "small")
print(f"Loading MusicGen: facebook/musicgen-{MODEL_NAME}")
model = MusicGen.get_pretrained(MODEL_NAME)
print("Model loaded!")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    duration = min(int(data.get("duration", 30)), 300)
    
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    
    print(f"Generating: {prompt} ({duration}s)")
    model.set_generation_params(duration=duration)
    wav = model.generate([prompt])
    
    output_path = os.path.join(os.path.dirname(__file__), "temp_output.wav")
    audio_write(
        output_path.replace(".wav", ""),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True
    )
    
    return send_file(output_path, mimetype="audio/wav",
                     as_attachment=True,
                     download_name=f"musicgen_{hash(prompt) % 100000}.wav")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9881, debug=False)
'''

with open(MUSICGEN_DIR / "api_server.py", "w", encoding="utf-8") as f:
    f.write(api_server)

# 启动脚本
start_api = '''# MusicGen API 启动脚本
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir\\venv\\Scripts\\Activate.ps1"
Set-Location $scriptDir
Write-Host "启动 MusicGen API 服务..." -ForegroundColor Green
Write-Host "地址: http://127.0.0.1:9881" -ForegroundColor Cyan
python api_server.py
'''

with open(MUSICGEN_DIR / "start_api.ps1", "w", encoding="utf-8") as f:
    f.write(start_api)

# 快速生成脚本
generate_ps1 = '''# MusicGen 快速生成脚本
param(
    [Parameter(Mandatory=$true)]
    [string]$Prompt,
    [int]$Duration = 30,
    [string]$Output = "generated_music.wav",
    [string]$Model = "small"
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptDir\\venv\\Scripts\\Activate.ps1"
Set-Location $scriptDir

Write-Host "生成AI音乐..." -ForegroundColor Green
Write-Host "描述: $Prompt" -ForegroundColor Yellow

python -c "
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('$Model')
model.set_generation_params(duration=$Duration)

print('Generating...')
wav = model.generate(['$Prompt'])

audio_write(
    '$Output'.replace('.wav', ''),
    wav[0].cpu(),
    model.sample_rate,
    strategy='loudness',
    loudness_compressor=True
)
print('Done: $Output')
"
'''

with open(MUSICGEN_DIR / "generate.ps1", "w", encoding="utf-8") as f:
    f.write(generate_ps1)

print("  脚本已创建")

# ------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------
print()
print("=" * 50)
print("  MusicGen 安装完成！")
print("=" * 50)
print()
print("使用方式:")
print()
print("1. 启动 API 服务:")
print(f"   cd {MUSICGEN_DIR}")
print(r"   .\start_api.ps1")
print()
print("2. 快速生成音乐:")
print(r"   .\generate.ps1 -Prompt '轻快钢琴曲' -Duration 30")
print()
print("3. 主项目使用:")
print("   python main.py dub --script script.txt --bgm-mode generate")
print()
print("注意:")
print("- 首次生成需加载模型，约 30-60 秒")
print("- CPU 模式较慢，建议后续装 CUDA PyTorch")
print()
