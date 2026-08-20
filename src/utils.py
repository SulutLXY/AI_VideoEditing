"""
LLM-AutoCut 通用工具函数

数据模型已迁移至 src/models.py，这里只保留：
- 日志初始化
- 文件/目录/路径工具
- FFmpeg / ffprobe 封装
- 时间码转换
- 配置加载与素材状态解析
- 剧本大纲解析
- JSON 文件读写

为兼容旧代码，从 models 重新导出核心类。
"""
import os
import re
import json
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 重新导出数据模型，保持旧代码导入方式兼容
from src.models import (
    MaterialState,
    Provenance,
    Relationship,
    Relationships,
    ScriptBeat,
    CVMetadata,
    Segment,
    Boundary,
    Shot,
)

# 配置日志
logger = logging.getLogger(__name__)


def init_logging(output_dir: Optional[str] = None):
    """初始化日志，确保输出目录存在后调用"""
    handlers = [logging.StreamHandler()]
    if output_dir:
        log_path = os.path.join(output_dir, "logs", "pipeline.log")
        ensure_dir(os.path.dirname(log_path))
        handlers.append(logging.FileHandler(log_path, encoding="utf-8", mode="a"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def run_ffmpeg(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """运行 FFmpeg 命令"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args
    logger.debug(f"FFmpeg cmd: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def run_ffprobe(video_path: str) -> Dict:
    """获取视频元数据"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,bit_rate",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,codec_name",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"ffprobe 无法读取视频: {video_path}\n"
            f"返回码: {result.returncode}\n"
            f"stderr: {result.stderr.strip() or '(空)'}"
        )
    return json.loads(result.stdout)


def sec_to_tc(seconds: float, fps: float = 24.0) -> str:
    """秒数转换为时间码 HH:MM:SS:FF"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    frames = int((seconds % 1) * fps)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


def tc_to_sec(tc: str, fps: float = 24.0) -> float:
    """时间码转换为秒数，支持 HH:MM:SS:FF 和 HH:MM:SS.mmm"""
    tc = tc.strip()
    if "." in tc:
        parts = tc.split(":")
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    else:
        parts = tc.split(":")
        if len(parts) == 4:
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        elif len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            f = 0
        else:
            raise ValueError(f"时间码格式错误: {tc}")
        return h * 3600 + m * 60 + s + f / fps


def parse_duration_string(duration_str: str) -> float:
    """把各种时长字符串解析为秒数

    支持格式:
    - "5m" / "5min" / "5分钟"
    - "300s" / "300"
    - "00:05:00" / "0:05:00"
    - "0:05:00:00" (HH:MM:SS:FF)
    """
    if duration_str is None:
        return 0.0
    s = str(duration_str).strip().lower()
    if not s:
        return 0.0

    # 纯数字 -> 秒
    if s.isdigit():
        return float(s)

    # 秒格式: 5s / 5sec / 5秒
    s_match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|sec|秒)$", s)
    if s_match:
        return float(s_match.group(1))

    # 分钟格式: 5m / 5min / 5分钟
    m_match = re.match(r"^(\d+(?:\.\d+)?)\s*(m|min|分钟)$", s)
    if m_match:
        return float(m_match.group(1)) * 60.0

    # 小时格式: 1h / 1hr / 1小时
    h_match = re.match(r"^(\d+(?:\.\d+)?)\s*(h|hr|小时)$", s)
    if h_match:
        return float(h_match.group(1)) * 3600.0

    # 时间码 HH:MM:SS / HH:MM:SS:FF
    if ":" in s:
        try:
            return tc_to_sec(s)
        except Exception:
            pass

    try:
        return float(s)
    except ValueError:
        logger.warning(f"无法解析时长字符串: {duration_str}")
        return 0.0


def get_video_files(directory: str, extensions: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".mxf")) -> List[str]:
    """获取目录下所有视频文件（去重，兼容大小写不敏感文件系统）"""
    seen = set()
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f"*{ext}"))
        files.extend(Path(directory).glob(f"*{ext.upper()}"))
    unique = []
    for f in files:
        key = str(Path(f).resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(str(f))
    return sorted(unique)


def get_image_files(directory: str, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    """获取目录下所有图片文件"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f"*{ext}"))
        files.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted([str(f) for f in files])


def md5_file(filepath: str) -> str:
    """计算文件 MD5"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 从环境变量读取 API Key
    for section in ["vlm", "llm"]:
        if not config.get("models", {}).get(section, {}).get("api_key"):
            env_key = os.getenv(f"{section.upper()}_API_KEY") or os.getenv("OPENAI_API_KEY")
            if env_key:
                config["models"][section]["api_key"] = env_key

    # 补全默认值
    if "split_scoring" not in config:
        config["split_scoring"] = {
            "weights": {
                "camera_change": 0.35,
                "subject_change": 0.25,
                "emotion_break": 0.15,
                "plot_shift": 0.10,
                "action_break": 0.10,
                "dialogue_break": 0.05,
            },
            "thresholds": {"high": 0.55, "medium": 0.40},
            "long_take_protection": {
                "enabled": True,
                "min_duration": 30.0,
                "threshold_boost": 0.15,
            },
        }

    if "materials" not in config:
        # 兼容旧版配置：把 paths.raw_materials 视为 RAW 目录
        config["materials"] = [
            {
                "path": config.get("paths", {}).get("raw_materials", "./materials/"),
                "state": "RAW",
                "split": True,
                "analyze": True,
            }
        ]

    return config


def resolve_material_state(
    video_path: str,
    materials_config: List[Dict],
    overrides: Optional[List[Dict]] = None,
) -> Tuple[str, Dict]:
    """根据配置解析单个素材的状态与覆盖规则"""
    video_path = os.path.abspath(video_path)

    # 优先处理全局 overrides
    if overrides:
        for override in overrides:
            pattern = override.get("match", "")
            if pattern and re.search(pattern, os.path.basename(video_path)):
                return override.get("state", "RAW"), override

    # 再处理 materials 中每个 item 自带的 overrides
    for item in materials_config:
        if "overrides" in item:
            for override in item["overrides"]:
                pattern = override.get("match", "")
                if pattern and re.search(pattern, os.path.basename(video_path)):
                    return override.get("state", "RAW"), override

    # 按目录前缀匹配
    best_len = 0
    matched = None
    for item in materials_config:
        if "path" not in item:
            continue
        item_path = os.path.abspath(item["path"])
        if video_path.startswith(item_path) and len(item_path) > best_len:
            best_len = len(item_path)
            matched = item

    if matched:
        return matched.get("state", "RAW"), matched

    return "RAW", {"path": os.path.dirname(video_path), "state": "RAW", "split": True, "analyze": True}


def parse_script_outline(script_path: str) -> List[ScriptBeat]:
    """
    解析剧本大纲 Markdown 文件

    期望格式:
    # 片名

    ## 第一幕：幕标题
    ### 场1-情节点A
    - 地点：xxx
    - 时间：xxx
    - 内容：xxx
    - 情绪：xxx
    - 关键动作：xxx, xxx
    - 关键台词："xxx"
    """
    beats = []
    current_act = ""

    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 匹配幕标题: ## 第一幕：标题
        if line.startswith("## "):
            current_act = line.replace("## ", "").strip()
            i += 1
            continue

        # 匹配场-情节点: ### 场1-情节点A
        if line.startswith("### "):
            beat_id = line.replace("### ", "").strip()

            # 解析后续属性
            props = {}
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("#"):
                prop_line = lines[j].strip()
                if prop_line.startswith("- "):
                    # 解析 "- 属性名：值" 或 "- 属性名: 值"
                    match = re.match(r"-\s*(\w+)[：:]\s*(.+)", prop_line)
                    if match:
                        key, value = match.group(1), match.group(2).strip()
                        props[key] = value
                j += 1

            beat = ScriptBeat(
                act=current_act,
                scene=beat_id.split("-")[0] if "-" in beat_id else beat_id,
                beat_id=beat_id,
                location=props.get("地点", props.get("location", "")),
                time=props.get("时间", props.get("time", "")),
                content=props.get("内容", props.get("content", "")),
                emotion=props.get("情绪", props.get("emotion", "")),
                key_actions=[a.strip() for a in props.get("关键动作", "").split("，") if a.strip()],
                key_dialogue=props.get("关键台词", props.get("dialogue", "")),
            )
            beats.append(beat)
            i = j
            continue

        i += 1

    logger.info(f"解析剧本大纲: {len(beats)} 个情节点")
    return beats


def save_json(data, path: str):
    """保存 JSON 文件"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {path}")


def load_json(path: str) -> Dict:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
