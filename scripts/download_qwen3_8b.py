#!/usr/bin/env python3
"""
下载 Qwen3-8B GGUF 模型到 models/local/

默认量化：Q4_K_M（适合 8GB 显存，约 7GB 可用显存）
可选量化：Q5_K_M / Q6_K / Q8_0

用法：
    python scripts/download_qwen3_8b.py
    python scripts/download_qwen3_8b.py --quant Q5_K_M
"""
import os
import sys
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict

# 把项目根目录加入路径
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.utils import logger, ensure_dir


# Qwen3-8B GGUF 发布源（按顺序尝试）
# 国内用户优先走 hf-mirror 或 modelscope
REPOS = [
    {"name": "hf-mirror-lmstudio", "repo": "lmstudio-community/Qwen3-8B-GGUF", "url_template": "https://hf-mirror.com/{repo}/resolve/main/{filename}"},
    {"name": "hf-mirror-qwen", "repo": "Qwen/Qwen3-8B-GGUF", "url_template": "https://hf-mirror.com/{repo}/resolve/main/{filename}"},
    {"name": "modelscope", "repo": "qwen/Qwen3-8B-GGUF", "url_template": "https://modelscope.cn/models/{repo}/resolve/master/{filename}"},
    {"name": "huggingface-lmstudio", "repo": "lmstudio-community/Qwen3-8B-GGUF", "url_template": "https://huggingface.co/{repo}/resolve/main/{filename}"},
    {"name": "huggingface-qwen", "repo": "Qwen/Qwen3-8B-GGUF", "url_template": "https://huggingface.co/{repo}/resolve/main/{filename}"},
]

QUANT_FILE_MAP = {
    "Q4_K_M": "Qwen3-8B-Q4_K_M.gguf",
    "Q5_K_M": "Qwen3-8B-Q5_K_M.gguf",
    "Q6_K": "Qwen3-8B-Q6_K.gguf",
    "Q8_0": "Qwen3-8B-Q8_0.gguf",
}

# 各量化级别约占用显存（加载 KV cache 前，含少量余量）
QUANT_VRAM_MB = {
    "Q4_K_M": 5500,
    "Q5_K_M": 6800,
    "Q6_K": 7800,
    "Q8_0": 9500,
}


def detect_available_vram_mb() -> float:
    """检测可用显存（MB）。优先 nvidia-smi，取最大可用 GPU；其次 torch.cuda"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            free_mbs = [float(l) for l in lines]
            if free_mbs:
                best = max(free_mbs)
                logger.info(f"nvidia-smi 检测到可用显存: {free_mbs} MB，取最大: {best:.0f} MB")
                return best
    except Exception as e:
        logger.debug(f"nvidia-smi 不可用: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            logger.info(f"torch.cuda 检测到可用显存: {free:.0f} MB")
            return free
    except Exception as e:
        logger.debug(f"torch.cuda 检测失败: {e}")

    logger.warning("未检测到 GPU 显存，按无独显处理")
    return 0.0


def choose_quant(quant_arg: Optional[str], available_vram_mb: float) -> str:
    """根据显存选择合适量化级别"""
    if quant_arg:
        quant = quant_arg.upper()
        if quant not in QUANT_FILE_MAP:
            raise ValueError(f"不支持的量化级别: {quant}，可选: {list(QUANT_FILE_MAP.keys())}")
        required = QUANT_VRAM_MB.get(quant, 0)
        if available_vram_mb > 0 and required > available_vram_mb:
            logger.warning(
                f"显存 {available_vram_mb:.0f}MB 可能不足 {quant} 所需 {required}MB，"
                f"请确认或换用更小的量化级别"
            )
        return quant

    # 自动选择：默认 Q4_K_M；若显存>8GB 可考虑 Q5_K_M
    if available_vram_mb >= QUANT_VRAM_MB["Q5_K_M"] + 500:
        logger.info(f"显存充裕 ({available_vram_mb:.0f}MB)，自动选择 Q5_K_M")
        return "Q5_K_M"

    logger.info(f"显存 {available_vram_mb:.0f}MB，自动选择 Q4_K_M")
    return "Q4_K_M"


def download_file(url: str, dest: Path) -> bool:
    """使用 urllib 下载文件，显示简单进度"""
    logger.info(f"开始下载: {url}")
    logger.info(f"保存到: {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("content-length", 0))
            block_size = 8192 * 16
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (block_size * 64) == 0:
                        pct = downloaded / total * 100
                        logger.info(f"下载进度: {pct:.1f}% ({downloaded / (1024**2):.0f}MB / {total / (1024**2):.0f}MB)")
        logger.info(f"下载完成: {dest} ({dest.stat().st_size / (1024**2):.1f}MB)")
        return True
    except urllib.error.HTTPError as e:
        logger.error(f"下载失败 HTTP {e.code}: {e.reason} ({url})")
        return False
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def try_download(repo_info: Dict[str, str], quant: str, dest: Path) -> bool:
    """尝试从指定镜像下载指定量化文件"""
    filename = QUANT_FILE_MAP[quant]
    url = repo_info["url_template"].format(repo=repo_info["repo"], filename=filename)
    logger.info(f"尝试镜像: {repo_info['name']} ({repo_info['repo']})")
    return download_file(url, dest)


def test_load_with_llama_cpp(model_path: Path) -> bool:
    """如果安装了 llama-cpp-python，尝试加载模型验证"""
    try:
        from llama_cpp import Llama
        logger.info("尝试用 llama-cpp-python 加载模型验证...")
        _ = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            verbose=False,
            n_gpu_layers=-1,
        )
        logger.info("llama-cpp-python 加载验证通过")
        return True
    except ImportError:
        logger.info("llama-cpp-python 未安装，跳过加载验证")
        return False
    except Exception as e:
        logger.warning(f"llama-cpp-python 加载验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 Qwen3-8B GGUF 模型")
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help=f"量化级别，可选: {', '.join(QUANT_FILE_MAP.keys())}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="模型保存目录，默认 models/local/"
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="下载后不做加载验证"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "models" / "local"
    ensure_dir(str(output_dir))

    available_vram = detect_available_vram_mb()
    quant = choose_quant(args.quant, available_vram)
    filename = QUANT_FILE_MAP[quant]
    dest = output_dir / filename

    logger.info("=" * 60)
    logger.info("Qwen3-8B GGUF 下载")
    logger.info("=" * 60)
    logger.info(f"量化级别: {quant}")
    logger.info(f"目标文件: {dest}")

    if dest.exists():
        logger.info(f"目标文件已存在，跳过下载: {dest}")
    else:
        ok = False
        for repo_info in REPOS:
            ok = try_download(repo_info, quant, dest)
            if ok:
                logger.info(f"镜像 {repo_info['name']} 下载成功")
                break
        if not ok:
            logger.error("下载失败，请检查网络或手动下载后放到 models/local/")
            sys.exit(1)

    # 记录配置
    info_path = output_dir / "qwen3_8b_info.json"
    import json
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen3-8B",
            "quant": quant,
            "file": str(dest),
            "available_vram_mb": round(available_vram, 1),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"模型信息已保存: {info_path}")

    if not args.no_test:
        test_load_with_llama_cpp(dest)

    logger.info("=" * 60)
    logger.info(f"完成。模型路径: {dest}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
