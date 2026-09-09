# -*- coding: utf-8 -*-
"""
LLM-AutoCut Web UI —— 流程式交互界面

布局：
- 左侧步骤导航（①准备 ②剧本分析 ③镜头筛选 ④剪辑导出），项目是一个整体，步骤只是视角切换
- 顶栏小按钮：配置（弹窗）、API Key（弹窗）、日志（浮窗，点开全屏展开）
- 内容区顶部：细进度条（运行时显示）；每个步骤面板顶部有独立的运行按钮
- 右侧资源弹窗：Phase 0/1 产物（只收录 Phase 1 分析过的资源）

运行模型：
- 每个步骤独立运行（subprocess 调 main.py --phase N）
- 任一运行期间所有运行按钮置灰，完成后恢复并实时刷新各步骤数据
- 运行日志写入内存环形缓冲，浮窗显示摘要，弹窗显示全文

启动：
    python webui.py [--config config.yaml] [--port 7860] [--share]
"""
import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import yaml

from src.utils import load_config, logger, ensure_dir, fallback_vlm_to_local
from src.phase2_anchor_editor import export_anchor_corrections, apply_anchor_corrections


# ----------------------------------------------------------------------
# 常量与会话状态
# ----------------------------------------------------------------------

DEFAULT_CONFIG = """project:
  name: "短片项目_雨夜告白"
  target_duration: "5m"
  style: "紧张中带温情"
  genre: "剧情短片"

paths:
  raw_materials: "./materials/"
  reference_images: "./refs/"
  output: "./output/"
  script_outline: "./script.md"
  temp: "./temp/"

materials:
  - path: "./materials/raw/"
    state: "RAW"
    split: true
    analyze: true

split_scoring:
  weights:
    camera_change: 0.35
    subject_change: 0.25
    emotion_break: 0.15
    plot_shift: 0.10
    action_break: 0.10
    dialogue_break: 0.05
  thresholds:
    high: 0.55
    medium: 0.40
  long_take_protection:
    enabled: true
    min_duration: 30.0
    threshold_boost: 0.15

quality_scoring:
  weights:
    visual_quality: 0.25
    stability: 0.20
    script_confidence: 0.25
    dialogue: 0.10
    duration: 0.10
    metadata_complete: 0.10

phase2:
  materials_dir: ""
  allow_cv_inventory: true

script_preprocessing:
  enabled: true
  provider: deepseek
  max_chunk_chars: 4000
  output_shot_requirements: true

models:
  local:
    enabled: true
    device: "cuda"
    cache_dir: "./models/local"
    vision:
      model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
      load_in_4bit: true
      max_new_tokens: 512
      keyframe_count: 3
      frame_interval: 0.5
      max_frames: 16
      use_script_context: true
      use_reference_images: true
      script_max_chars: 1500
      max_ref_images: 8
  vlm:
    provider: doubao
    model: doubao-seed-2-0-pro-260215
    api_key: ""
    base_url: https://ark.cn-beijing.volces.com/api/v3
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150
  video_vlm:
    provider: vlm_fallback
    model: doubao-seed-2-0-pro-260215
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150
  llm:
    provider: deepseek
    model: deepseek-chat
    api_key: ""
    base_url: https://api.deepseek.com
    max_tokens: 8192
    temperature: 0.5
  asr:
    provider: whisper
    model: large-v3
    device: cpu
    language: zh

processing:
  scene_threshold: 0.3
  min_shot_duration: 1.0
  keyframe_strategy: adaptive
  keyframe_interval: 2
  keyframe_per_shot: 3
  enable_l1_hash: true
  enable_l2_visual: false
  enable_l3_semantic: false
  phash_threshold: 10
  duplicate_similarity: 0.92
  slow_motion_min_fps: 60
  speed_ramping: true
  export_edl: true
  export_fcpxml: true
  export_csv: true
  export_json: true

character_refs:
  auto_detect: true
"""

DEFAULT_SCRIPT = """# 剧本大纲

## 未分幕

### 场1-情节点A
- 地点：
- 时间：
- 内容：
- 情绪：
- 关键动作：
- 关键台词：""
"""

# Phase 1 资源库默认路径（当前项目已生成的分析产物，后续再调整此逻辑）
RESOURCE_ANALYSIS_PATH = "output/phase1_analysis.json"
RESOURCE_SPLIT_CLIPS_DIR = "output/phase1_split_clips"
RESOURCE_ROUGH_CLIPS_DIR = "output/phase0_rough_clips"


class SessionState:
    """会话状态：路径、API Key、运行锁、日志缓冲"""

    def __init__(self):
        self.work_dir = str(WORKSPACE_DIR)
        self.output_dir = os.path.join(self.work_dir, "output")
        self.vlm_key = ""
        self.llm_key = ""
        # 运行状态（模块级单用户够用）
        self.running = False
        self.run_step = ""           # 当前运行阶段标签，如 "2"
        self.run_label = ""          # 进度条文字
        self.run_done = False        # 完成标志，由 Timer 消费
        self.run_ok = False
        self.log = deque(maxlen=800)
        self.suggest_view = ""       # 运行完成后建议切换的视图（如 "review"）
        self.force_edit_view = False
        # 各步骤状态栏文字（tick 定时刷新读取，点击事件写入）
        self.status_msgs = {"prep": "", "script": "", "match": "", "export": ""}

    def reset_run(self):
        self.running = False
        self.run_step = ""
        self.run_label = ""
        self.run_done = False
        self.run_ok = False
        self.suggest_view = ""
        self.force_edit_view = False


SESSION = SessionState()


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------

def _workspace_path(*parts: str) -> str:
    return os.path.join(SESSION.work_dir, *parts)


def _output_path(name: str) -> str:
    return os.path.join(SESSION.output_dir, name)


def _read_text(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _copy_files_to_dir(files, directory):
    """把上传的文件复制到指定目录，返回成功复制的文件名列表"""
    ensure_dir(directory)
    copied = []
    for file in files or []:
        if isinstance(file, str) and os.path.exists(file):
            dest = os.path.join(directory, os.path.basename(file))
            shutil.copy2(file, dest)
            copied.append(os.path.basename(file))
    return copied


def _clear_dir_if_has_upload(files, directory):
    """如果本次有上传，则清空对应子目录，避免历史文件重复/混淆"""
    if not files:
        return
    has_file = any(isinstance(f, str) and os.path.exists(f) for f in files)
    if has_file and os.path.exists(directory):
        for old in os.listdir(directory):
            old_path = os.path.join(directory, old)
            if os.path.isfile(old_path):
                os.remove(old_path)


def _split_comma(text: Any) -> List[str]:
    if pd.isna(text) or text is None:
        return []
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.1f} GB"


def _human_time(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return "-"


# ----------------------------------------------------------------------
# 剧本导入与预处理（沿用已验证逻辑）
# ----------------------------------------------------------------------

def on_script_upload(file_path):
    """上传剧本文档后读取内容并回填编辑器"""
    from src.services.script_service import read_script_file, ScriptReadError

    if file_path is None:
        return gr.update(), "未选择文件"
    if isinstance(file_path, list):
        file_path = file_path[0]
    if not isinstance(file_path, str) or not os.path.exists(file_path):
        return gr.update(), "文件路径无效"

    try:
        text = read_script_file(file_path)
        return text, f"已导入: {os.path.basename(file_path)}"
    except ScriptReadError as e:
        return gr.update(), f"导入失败: {e}"
    except Exception as e:
        return gr.update(), f"导入失败: {e}\n{traceback.format_exc()}"


def preprocess_script(script_text: str, uploaded_file_path, config_text: str = ""):
    """调用 LLM 把原始剧本解析为结构化台本（导入剧本按钮）。

    优先解析用户上传的文件；若未上传文件，则解析编辑器中的文本。
    """
    from src.services.script_service import (
        ScriptPreprocessor, is_structured_outline, read_script_file,
        ScriptReadError, ScriptParseError,
    )
    from src.services.llm_service import LLMService
    from src.utils import parse_script_outline

    work_dir = SESSION.work_dir
    ensure_dir(work_dir)
    script_path = os.path.join(work_dir, "script.md")

    # 编辑器即事实源：优先解析编辑器内容；仅当编辑器为空时才回退上传文件
    source_label = "编辑器内容"
    text_to_parse = script_text
    if not text_to_parse or not text_to_parse.strip():
        if uploaded_file_path:
            if isinstance(uploaded_file_path, list):
                uploaded_file_path = uploaded_file_path[0] if uploaded_file_path else None
            if isinstance(uploaded_file_path, str) and os.path.exists(uploaded_file_path):
                try:
                    text_to_parse = read_script_file(uploaded_file_path)
                    source_label = os.path.basename(uploaded_file_path)
                except ScriptReadError as e:
                    return script_text, f"读取上传文件失败: {e}"
            else:
                return script_text, "上传文件路径无效"

    if not text_to_parse or not text_to_parse.strip():
        return script_text, "剧本内容为空，无法解析"

    # 配置：优先用工作区 config.yaml，其次弹窗内容，再次内置默认
    if os.path.exists(_workspace_path("config.yaml")):
        config_text = _read_text(_workspace_path("config.yaml"))
    if not config_text.strip():
        config_text = DEFAULT_CONFIG
    try:
        config = yaml.safe_load(config_text) or {}
    except Exception as e:
        return script_text, f"配置解析失败: {e}"

    # 注入会话中的 API Key
    if SESSION.vlm_key:
        config.setdefault("models", {}).setdefault("vlm", {})["api_key"] = SESSION.vlm_key
    if SESSION.llm_key:
        config.setdefault("models", {}).setdefault("llm", {})["api_key"] = SESSION.llm_key

    if is_structured_outline(text_to_parse):
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return text_to_parse, f"当前剧本（{source_label}）已是结构化大纲，已保存，可直接运行剧本分析"

    llm_config = config.get("models", {}).get("llm", {})
    if not llm_config.get("api_key"):
        return script_text, "错误：未设置 LLM API Key，请点击右上角 🔑 填写后再解析。"

    try:
        llm_service = LLMService(config)
        preprocessor_config = config.get("script_preprocessing", {})
        preprocessor = ScriptPreprocessor(llm_service, preprocessor_config)
        parsed = preprocessor.preprocess(text_to_parse)

        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=work_dir)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(parsed)
            beats = parse_script_outline(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not beats:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(text_to_parse)
            return (
                text_to_parse,
                f"剧本解析失败（来源：{source_label}）：模型返回了结构化文本，但未能提取出任何情节点。"
            )

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(parsed)
        return parsed, f"剧本解析完成（来源：{source_label}），已生成结构化台本，可检查后运行剧本分析。"
    except ScriptParseError as e:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return (
            script_text,
            f"剧本解析失败（来源：{source_label}）：{e}\n可手动编辑成标准大纲后再试。",
        )
    except Exception as e:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return script_text, f"剧本解析失败: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------
# 工作区准备（① 准备步骤，无 AI）
# ----------------------------------------------------------------------

def prepare_workspace_files(raw_files: list, materials_dir: str = "") -> str:
    """准备项目内工作目录：写配置、归置素材，不运行任何 AI 流程。"""
    try:
        ensure_dir(SESSION.work_dir)
        ensure_dir(SESSION.output_dir)

        # 配置：保持工作区已有 config.yaml；没有则写默认模板
        config_path = _workspace_path("config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(DEFAULT_CONFIG)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            config = yaml.safe_load(DEFAULT_CONFIG)

        # 素材目录
        raw_dir = _workspace_path("materials", "raw")
        _clear_dir_if_has_upload(raw_files, raw_dir)
        uploaded = _copy_files_to_dir(raw_files, raw_dir)

        config["paths"]["raw_materials"] = raw_dir
        config["paths"]["output"] = SESSION.output_dir
        config["paths"]["temp"] = _workspace_path("temp")
        config["paths"]["script_outline"] = _workspace_path("script.md")
        config["paths"]["reference_images"] = _workspace_path("refs")
        for item in config.get("materials", []):
            if item.get("state") == "RAW":
                item["path"] = raw_dir

        if materials_dir and materials_dir.strip():
            materials_dir_abs = os.path.abspath(materials_dir.strip())
            config.setdefault("phase2", {})["materials_dir"] = materials_dir_abs
            config["paths"]["raw_materials"] = materials_dir_abs

        # 注入会话 API Key
        if SESSION.vlm_key:
            config.setdefault("models", {}).setdefault("vlm", {})["api_key"] = SESSION.vlm_key
        if SESSION.llm_key:
            config.setdefault("models", {}).setdefault("llm", {})["api_key"] = SESSION.llm_key

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        script_exists = os.path.exists(_workspace_path("script.md"))
        return (
            f"工作区已就绪: {SESSION.work_dir}\n"
            f"已上传素材: {', '.join(uploaded) if uploaded else '无'}\n"
            f"剧本: {'已存在 ' + _workspace_path('script.md') if script_exists else '尚未创建（到「剧本分析」步骤导入）'}"
        )
    except Exception as e:
        return f"准备工作区失败: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------
# 步骤运行引擎（subprocess 隔离 main.py，日志进环形缓冲）
# ----------------------------------------------------------------------

PHASE_LABELS = {"2": "剧本分析", "3": "镜头筛选", "4": "剪辑导出"}


def _validate_keys(phase: str, config: dict) -> Optional[str]:
    """运行前校验 API Key，缺啥返回错误文案，OK 返回 None"""
    vlm_key = config.get("models", {}).get("vlm", {}).get("api_key")
    llm_key = config.get("models", {}).get("llm", {}).get("api_key")
    vlm_env = any(os.getenv(k) for k in ["OPENAI_API_KEY", "ARK_API_KEY", "VLM_API_KEY"])
    llm_env = any(os.getenv(k) for k in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"])

    local_cfg = config.get("models", {}).get("local", {})
    vlm_provider = config.get("models", {}).get("vlm", {}).get("provider", "openai")
    local_vlm = bool(local_cfg.get("enabled", False)) and vlm_provider == "local"

    if phase == "2" and not llm_key and not llm_env:
        return "错误：未设置 LLM API Key，请点击右上角 🔑 填写。"
    if phase == "3" and not llm_key and not llm_env:
        return "错误：未设置 LLM API Key，请点击右上角 🔑 填写。"
    if phase in ("0", "1") and not local_vlm and not vlm_key and not vlm_env:
        return "错误：未设置 VLM API Key，请点击右上角 🔑 填写。"
    return None


def _subprocess_python() -> str:
    """优先使用项目 .venv311 的 python（含 torch 等完整依赖），否则回退当前解释器"""
    venv_python = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
    return str(venv_python) if venv_python.exists() else sys.executable


def _run_subprocess(phase: str) -> None:
    """在工作区内运行 main.py --phase N，日志写入 SESSION.log"""
    config_path = _workspace_path("config.yaml")
    cmd = [
        _subprocess_python(), "main.py",
        "--config", config_path,
        "--phase", phase,
        "--clean",
    ]
    SESSION.log.append(f"[运行] {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            SESSION.log.append(line)
            # 提取进度文字：取最近的含阶段/百分比/计数的行
            if any(k in line for k in ["Phase", "第", "/", "镜头", "片段", "节点", "beat"]):
                if len(line) <= 80:
                    SESSION.run_label = line.strip()
        return_code = process.wait()
        SESSION.run_ok = (return_code == 0)
        if SESSION.run_ok:
            SESSION.log.append(f"[完成] {PHASE_LABELS.get(phase, phase)} 运行成功")
        else:
            SESSION.log.append(f"[失败] 退出码 {return_code}")
    except Exception as e:
        SESSION.run_ok = False
        SESSION.log.append(f"[异常] {e}\n{traceback.format_exc()}")
    finally:
        SESSION.running = False
        SESSION.run_done = True
        if SESSION.run_ok:
            SESSION.suggest_view = {"2": "review"}.get(phase, "")


def start_step_run(phase: str) -> str:
    """启动步骤运行（非阻塞，后台线程执行）。返回状态文案。"""
    if SESSION.running:
        return "已有任务在运行，请等待完成"

    config_path = _workspace_path("config.yaml")
    if not os.path.exists(config_path):
        return "错误：工作区配置不存在，请先在「① 准备」步骤点击运行"
    try:
        config = load_config(config_path)
    except Exception as e:
        return f"配置读取失败: {e}"

    # 运行前自动回退本地 VLM 配置（与旧行为一致）
    if fallback_vlm_to_local(config):
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    err = _validate_keys(phase, config)
    if err:
        return err

    # 前置产物检查
    if phase == "3" and not os.path.exists(_output_path("script_beats_analysis.json")):
        return "缺少剧本分析结果，请先完成「② 剧本分析」"
    if phase == "4" and not (
        os.path.exists(_output_path("phase2_selected_shots.json"))
        or os.path.exists(_output_path("phase1_analysis.json"))
    ):
        return "缺少素材/筛选结果，请先完成「③ 镜头筛选」"

    SESSION.running = True
    SESSION.run_step = phase
    SESSION.run_label = f"{PHASE_LABELS[phase]} 启动中…"
    threading.Thread(target=_run_subprocess, args=(phase,), daemon=True).start()
    return f"{PHASE_LABELS[phase]} 已开始运行"


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------

def load_beats_data() -> Tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    """加载 Phase 2 剧本节点与分镜点。

    返回 (beat表[可编辑列], sub_beat表, 状态摘要, 是否存在分析结果)
    """
    empty = pd.DataFrame()
    path = _output_path("script_beats_analysis.json")
    if not os.path.exists(path):
        return empty, empty, "尚未运行剧本分析", False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        beat_rows, sub_rows = [], []
        for b in data.get("beats", []):
            bid = b.get("beat_id", "")
            if b.get("locked"):
                bid = "🔒 " + bid
            beat_rows.append({
                "情节点": bid,
                "地点": b.get("location", ""),
                "时间": b.get("time", ""),
                "内容": b.get("content", ""),
                "情绪": b.get("emotion", ""),
                "关键动作": ", ".join(b.get("key_actions", [])),
                "关键台词": b.get("key_dialogue", ""),
                "目标时长s": b.get("estimated_duration", 0),
                "节奏": b.get("pace", ""),
                "优先级": b.get("priority", ""),
            })
            for sb in b.get("sub_beats", []):
                sub_rows.append({
                    "分镜点": sb.get("sub_beat_id", ""),
                    "所属情节点": sb.get("parent_beat_id", ""),
                    "内容": sb.get("content", ""),
                    "关键动作": ", ".join(sb.get("key_actions", [])),
                    "关键台词": sb.get("key_dialogue", ""),
                    "时长s": sb.get("estimated_duration", 0),
                    "情绪": sb.get("emotion", ""),
                    "节奏": sb.get("pace", ""),
                })
        n_beat, n_sub = len(beat_rows), len(sub_rows)
        total = sum(r["目标时长s"] for r in beat_rows if isinstance(r["目标时长s"], (int, float)))
        summary = f"剧本节点 {n_beat} 个 · 分镜点 {n_sub} 个 · 目标总时长 {total:.0f}s"
        return pd.DataFrame(beat_rows), pd.DataFrame(sub_rows), summary, True
    except Exception as e:
        return empty, empty, f"加载剧本分析失败: {e}", False


def load_beats_readonly_with_shots() -> pd.DataFrame:
    """左侧剧本节点表（只读）+ 镜头栏：Phase 3 运行后显示每个节点匹配到的镜头编号与顺序。"""
    path = _output_path("script_beats_analysis.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # shot -> beat 归属与顺序（phase2_selected_shots.json 由 Phase 3 写出）
    shot_map: Dict[str, List[str]] = {}
    selected_path = _output_path("phase2_selected_shots.json")
    if os.path.exists(selected_path):
        try:
            with open(selected_path, "r", encoding="utf-8") as f:
                sel = json.load(f)
            indexed = []
            for s in sel.get("shots", []):
                anchor = s.get("script_anchor") or {}
                beat = anchor.get("beat", "")
                if not beat or beat == "UNMATCHED":
                    continue
                # 只统计核心/保留类镜头；备选不计入"匹配镜头"
                if s.get("status") not in ("核心", "保留", "强制保留", "待复核"):
                    continue
                indexed.append((beat, s.get("shot_id", ""), s.get("source_file", "")))
            for beat, shot_id, source in indexed:
                shot_map.setdefault(beat, []).append(f"{shot_id}")
        except Exception:
            pass

    rows = []
    for b in data.get("beats", []):
        bid = b.get("beat_id", "")
        if b.get("locked"):
            bid = "🔒 " + bid
        shots = shot_map.get(b.get("beat_id", ""), [])
        rows.append({
            "情节点": bid,
            "内容": b.get("content", ""),
            "目标时长s": round(b.get("estimated_duration", 0) or 0, 1),
            "对白": (b.get("key_dialogue") or "")[:30],
            "匹配镜头": " → ".join(shots) if shots else "—",
            "镜头数": len(shots),
        })
    return pd.DataFrame(rows)


def load_split_clip_configs() -> List[Dict[str, Any]]:
    """读取 phase1_split_clips/ 下每个镜头的 Sxxx_config.json，返回 config 列表。"""
    split_dir = _output_path(os.path.basename(RESOURCE_SPLIT_CLIPS_DIR))
    configs: List[Dict[str, Any]] = []
    if not os.path.isdir(split_dir):
        return configs
    for name in sorted(os.listdir(split_dir)):
        if not name.endswith("_config.json"):
            continue
        try:
            with open(os.path.join(split_dir, name), "r", encoding="utf-8") as f:
                configs.append(json.load(f))
        except Exception as e:
            logger.warning(f"读取 {name} 失败: {e}")
    return configs


def load_materials_data() -> pd.DataFrame:
    """右侧素材表：Phase 1 最终素材表（数据源 phase1_split_clips/Sxxx_config.json）。"""
    rows = []
    for c in load_split_clip_configs():
        rows.append({
            "镜头编号": c.get("shot_id", ""),
            "源文件": c.get("source_file", ""),
            "时长s": round(c.get("duration_sec", 0) or 0, 1),
            "景别": c.get("shot_type", ""),
            "运镜": c.get("camera_movement", ""),
            "内容摘要": c.get("content_summary", ""),
            "动作": c.get("action", ""),
            "情绪": c.get("emotion", ""),
        })
    return pd.DataFrame(rows)


def load_anchor_assignment() -> pd.DataFrame:
    """镜头→剧情归属分配表（可编辑「归属情节点」列），用于 Phase 3 手动调整。"""
    selected_path = _output_path("phase2_selected_shots.json")
    if not os.path.exists(selected_path):
        return pd.DataFrame()
    try:
        with open(selected_path, "r", encoding="utf-8") as f:
            sel = json.load(f)
        rows = []
        for s in sel.get("shots", []):
            anchor = s.get("script_anchor") or {}
            beat = anchor.get("beat", "")
            rows.append({
                "镜头编号": s.get("shot_id", ""),
                "源文件": s.get("source_file", ""),
                "归属情节点": "" if beat in ("", "UNMATCHED") else beat,
                "置信度": anchor.get("confidence", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"加载归属分配失败: {e}")
        return pd.DataFrame()


def save_anchor_assignment(df: pd.DataFrame) -> str:
    """把手动调整写回 phase2_selected_shots.json（复用锚定校正机制）。"""
    if df is None or df.empty:
        return "分配表为空，未保存"
    selected_path = _output_path("phase2_selected_shots.json")
    if not os.path.exists(selected_path):
        return "尚未运行镜头筛选，无法保存分配"
    try:
        # 导出当前锚定校正 CSV，用用户改的「归属情节点」覆盖 corrected_beat 后应用
        csv_path = _output_path("phase2_anchor_corrections.csv")
        export_anchor_corrections(SESSION.output_dir, csv_path)
        corr = pd.read_csv(csv_path, encoding="utf-8-sig")
        assign = {str(r.get("镜头编号", "")).strip(): str(r.get("归属情节点", "")).strip()
                  for _, r in df.iterrows()}
        corr["corrected_beat"] = corr.apply(
            lambda r: assign.get(str(r.get("shot_id", "")).strip(), r.get("corrected_beat", "")),
            axis=1,
        )
        corr.to_csv(csv_path, index=False, encoding="utf-8-sig")
        apply_anchor_corrections(SESSION.output_dir, csv_path)
        changed = sum(1 for v in assign.values() if v)
        return f"已保存 {changed} 条镜头归属分配，左侧节点镜头栏已同步"
    except Exception as e:
        return f"保存分配失败: {e}\n{traceback.format_exc()}"


def save_beats(df_beats: pd.DataFrame, df_sub: pd.DataFrame) -> str:
    """保存左侧审核节点（可编辑）→ script_beats_analysis.json + workspace/script.md"""
    if df_beats is None or df_beats.empty:
        return "剧本节点表为空，无法保存"
    path = _output_path("script_beats_analysis.json")
    if not os.path.exists(path):
        return "尚未运行剧本分析，无法保存"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sub_map: Dict[str, List[Dict[str, Any]]] = {}
        old_subs = {sb.get("sub_beat_id", ""):
                    sb for b in data.get("beats", []) for sb in b.get("sub_beats", [])}
        has_sub_edit = df_sub is not None and not df_sub.empty
        if has_sub_edit:
            for _, sr in df_sub.iterrows():
                sb_id = str(sr.get("分镜点", "")).strip()
                parent = str(sr.get("所属情节点", "")).strip()
                if not sb_id or not parent:
                    continue
                old_sb = old_subs.get(sb_id, {})
                sub_map.setdefault(parent, []).append({
                    "sub_beat_id": sb_id,
                    "parent_beat_id": parent,
                    "act": sr.get("幕", "") or old_sb.get("act", ""),
                    "scene": sr.get("场", "") or old_sb.get("scene", ""),
                    "content": sr.get("内容", ""),
                    "key_actions": _split_comma(sr.get("关键动作", "")),
                    "key_dialogue": sr.get("关键台词", ""),
                    "estimated_duration": float(sr.get("时长s", 0) or 0),
                    "emotion": sr.get("情绪", ""),
                    "pace": sr.get("节奏", ""),
                })

        updated = []
        for _, br in df_beats.iterrows():
            beat_id = str(br.get("情节点", "")).strip().replace("🔒", "").strip()
            old = next((o for o in data.get("beats", []) if o.get("beat_id") == beat_id), {})
            beat = {
                "act": br.get("幕", "") or old.get("act", ""),
                "scene": br.get("场", "") or old.get("scene", ""),
                "beat_id": beat_id,
                "location": br.get("地点", ""),
                "time": br.get("时间", ""),
                "content": br.get("内容", ""),
                "emotion": br.get("情绪", ""),
                "key_actions": _split_comma(br.get("关键动作", "")),
                "key_dialogue": br.get("关键台词", ""),
                "estimated_duration": float(br.get("目标时长s", 0) or 0),
                "pace": br.get("节奏", ""),
            }
            beat["sub_beats"] = sub_map.get(beat_id, []) if has_sub_edit else old.get("sub_beats", [])
            for keep in ["dialogue_entries", "gender_state", "gender_transition",
                         "emotion_intensity", "priority", "required_shots_count", "locked"]:
                if keep in old:
                    beat[keep] = old[keep]
            updated.append(beat)

        data["beats"] = updated
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        script_path = _workspace_path("script.md")
        try:
            _save_beats_to_script_md(updated, script_path)
        except Exception as e:
            logger.warning(f"更新 script.md 失败: {e}")

        return f"已保存 {len(updated)} 个节点（时长/内容按你的修改原样生效，未走 AI 分配）"
    except Exception as e:
        return f"保存失败: {e}\n{traceback.format_exc()}"


def _save_beats_to_script_md(beats: List[Dict[str, Any]], script_path: str):
    """把 beat/sub-beat 数据写回标准 Markdown 大纲"""
    lines = ["# 剧本大纲", ""]
    acts: Dict[str, List[Dict[str, Any]]] = {}
    for b in beats:
        acts.setdefault(b.get("act") or "未分幕", []).append(b)
    for act, act_beats in acts.items():
        lines.append(f"## {act}")
        lines.append("")
        for b in act_beats:
            beat_id = b.get("beat_id") or f"{b.get('scene', '场')}-情节点"
            lines.append(f"### {beat_id}")
            lines.append(f"- 地点：{b.get('location', '')}")
            lines.append(f"- 时间：{b.get('time', '')}")
            lines.append(f"- 内容：{b.get('content', '')}")
            lines.append(f"- 情绪：{b.get('emotion', '')}")
            actions = b.get("key_actions", [])
            if isinstance(actions, str):
                actions = [a.strip() for a in actions.split(",") if a.strip()]
            lines.append(f"- 关键动作：{'，'.join(actions)}")
            lines.append(f"- 关键台词：\"{b.get('key_dialogue', '')}\"")
            if b.get("gender_state"):
                lines.append(f"- 性别状态：{b.get('gender_state')}")
            if b.get("gender_transition"):
                lines.append(f"- 状态切换：{b.get('gender_transition')}")
            if b.get("locked"):
                lines.append("- 标记：锁定")
            lines.append(f"- 建议时长：{b.get('estimated_duration', '')}")
            lines.append(f"- 节奏：{b.get('pace', '')}")
            lines.append(f"- 优先级：{b.get('priority', '')}")
            for sb in b.get("sub_beats", []):
                lines.append(f"#### {sb.get('sub_beat_id', '')}")
                lines.append(f"  - 内容：{sb.get('content', '')}")
                lines.append(f"  - 情绪：{sb.get('emotion', '')}")
                lines.append(f"  - 关键台词：\"{sb.get('key_dialogue', '')}\"")
                lines.append(f"  - 建议时长：{sb.get('estimated_duration', '')}")
            lines.append("")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_config_dict() -> Dict[str, Any]:
    """读取 workspace/config.yaml 为 dict，注入会话 API Key。"""
    text = _read_text(_workspace_path("config.yaml")) or DEFAULT_CONFIG
    config = yaml.safe_load(text) or {}
    if SESSION.vlm_key:
        config.setdefault("models", {}).setdefault("vlm", {})["api_key"] = SESSION.vlm_key
    if SESSION.llm_key:
        config.setdefault("models", {}).setdefault("llm", {})["api_key"] = SESSION.llm_key
    return config


def read_target_duration() -> float:
    """从 workspace/config.yaml 读 project.target_duration，返回秒。"""
    try:
        from src.utils import parse_duration_string
        config = _load_config_dict()
        return float(parse_duration_string(config.get("project", {}).get("target_duration", 0)) or 0)
    except Exception:
        return 0.0


def _write_target_duration(total_sec: float) -> None:
    """把新总时长写回 workspace/config.yaml 的 project.target_duration（保留注释）。"""
    path = _workspace_path("config.yaml")
    text = _read_text(path)
    new_line = f'  target_duration: "{int(total_sec)}s"'
    import re
    if re.search(r"^\s*target_duration\s*:", text, flags=re.MULTILINE):
        text = re.sub(r"^\s*target_duration\s*:.*$", new_line, text, flags=re.MULTILINE)
    else:
        # project: 段不存在时补一个
        if re.search(r"^project\s*:", text, flags=re.MULTILINE):
            text = re.sub(r"^(project\s*:\s*)$", r"\1\n" + new_line, text, flags=re.MULTILINE)
        else:
            text = f"project:\n{new_line}\n\n" + text
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def apply_target_duration(total_sec) -> str:
    """应用用户输入的目标总时长：写回 config，并调用 LLM 按剧情重新分配各节点时长。"""
    from types import SimpleNamespace
    from src.services.llm_service import LLMService

    try:
        total = float(total_sec or 0)
    except (TypeError, ValueError):
        return "目标总时长无效，请输入数字（秒）"
    if total < 5:
        return "目标总时长至少 5 秒"

    path = _output_path("script_beats_analysis.json")
    if not os.path.exists(path):
        return "尚未运行剧本分析，无法分配时长"

    config = _load_config_dict()
    llm_cfg = config.get("models", {}).get("llm", {})
    if not llm_cfg.get("api_key"):
        return "错误：未设置 LLM API Key，请点击右上角 🔑 填写后再应用时长。"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        beats_raw = data.get("beats", [])
        if not beats_raw:
            return "剧本节点为空，无法分配时长"

        # 构造 analyze_script_beats 需要的轻量 beat 对象（鸭子类型）
        beats = [SimpleNamespace(
            beat_id=b.get("beat_id", ""),
            act=b.get("act", ""), scene=b.get("scene", ""),
            location=b.get("location", ""), time=b.get("time", ""),
            content=b.get("content", ""), emotion=b.get("emotion", ""),
            key_actions=b.get("key_actions", []) or [],
            key_dialogue=b.get("key_dialogue", ""),
        ) for b in beats_raw]

        llm_service = LLMService(config)
        analysis = llm_service.analyze_script_beats(beats, total)
        if not analysis:
            return "LLM 时长分配失败（无返回），请查看日志"

        # 写回 json：只覆盖 LLM 返回的字段，保留 dialogue_entries 等其他字段
        for b in beats_raw:
            info = analysis.get(b.get("beat_id", ""))
            if not info:
                continue
            b["estimated_duration"] = round(float(info.get("estimated_duration", 0) or 0), 1)
            b["pace"] = info.get("pace", b.get("pace", ""))
            b["emotion_intensity"] = info.get("emotion_intensity", b.get("emotion_intensity", 0))
            b["priority"] = info.get("priority", b.get("priority", 3))
            b["required_shots_count"] = info.get("required_shots_count", b.get("required_shots_count", 1))
            if info.get("key_actions"):
                b["key_actions"] = info["key_actions"]

        data["beats"] = beats_raw
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            _save_beats_to_script_md(beats_raw, _workspace_path("script.md"))
        except Exception as e:
            logger.warning(f"更新 script.md 失败: {e}")
        _write_target_duration(total)

        got = sum(b.get("estimated_duration", 0) for b in beats_raw)
        return (f"已按新目标 {total:.0f}s 由 LLM 重新分配时长，"
                f"各节点合计 {got:.1f}s，配置已同步")
    except Exception as e:
        return f"应用目标时长失败: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------
# Phase 4 产物与资源库
# ----------------------------------------------------------------------

PRODUCT_FILES = [
    ("成片视频", "final_with_dubbing.mp4"),
    ("混音音频", "mixed_audio.wav"),
    ("剪辑决策", "phase3_edit_decision.json"),
    ("EDL 时间线", "timeline.edl"),
    ("FCPXML 时间线", "timeline.fcpxml"),
    ("CSV 时间线", "timeline_final.csv"),
    ("配音信息", "dub_info.json"),
    ("最终时间线", "timeline.json"),
]


def load_products() -> pd.DataFrame:
    rows = []
    for label, name in PRODUCT_FILES:
        p = _output_path(name)
        exists = os.path.exists(p)
        rows.append({
            "产物": label,
            "文件名": name,
            "状态": "已生成" if exists else "未生成",
            "大小": _fmt_size(os.path.getsize(p)) if exists else "-",
            "生成时间": _human_time(os.path.getmtime(p)) if exists else "-",
        })
    return pd.DataFrame(rows)


def existing_product_names() -> List[str]:
    return [name for _, name in PRODUCT_FILES if os.path.exists(_output_path(name))]


def load_used_materials() -> List[Tuple[str, str]]:
    """成片实际引用到的源素材 [(文件名, 路径)]，用于单独下载"""
    path = _output_path("timeline.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        seen: Dict[str, str] = {}
        for item in data.get("timeline", []):
            src = item.get("source_path") or ""
            name = item.get("source_file") or os.path.basename(src)
            if src and os.path.exists(src):
                seen.setdefault(name, src)
        return sorted(seen.items())
    except Exception:
        return []


def load_resource_library() -> Tuple[str, pd.DataFrame]:
    """右侧资源弹窗内容：Phase 0/1 产物（只收录 Phase 1 分析过的资源）。"""
    lines = []
    df = pd.DataFrame()

    p0_dir = _output_path(os.path.basename(RESOURCE_ROUGH_CLIPS_DIR))
    if os.path.isdir(p0_dir):
        n = len([x for x in os.listdir(p0_dir) if os.path.isfile(os.path.join(p0_dir, x))])
        lines.append(f"Phase 0 粗剪片段：{n} 个（{RESOURCE_ROUGH_CLIPS_DIR}/）")
    split_dir = _output_path(os.path.basename(RESOURCE_SPLIT_CLIPS_DIR))

    configs = load_split_clip_configs()
    if configs:
        lines.append(f"Phase 1 素材分析：{len(configs)} 个镜头（{RESOURCE_SPLIT_CLIPS_DIR}/ 下 Sxxx_config.json）")
        n_clips = len([x for x in os.listdir(split_dir)
                       if x.endswith((".mp4", ".mov"))]) if os.path.isdir(split_dir) else 0
        lines.append(f"切分片段目录：{RESOURCE_SPLIT_CLIPS_DIR}/（{n_clips} 个视频文件）")
        rows = []
        for c in configs:
            rows.append({
                "镜头编号": c.get("shot_id", ""),
                "片段文件": c.get("clip_path", ""),
                "源文件": c.get("source_file", ""),
                "时长s": round(c.get("duration_sec", 0) or 0, 1),
                "景别": c.get("shot_type", ""),
                "内容摘要": c.get("content_summary", ""),
                "动作": c.get("action", ""),
                "情绪": c.get("emotion", ""),
            })
        df = pd.DataFrame(rows)
    else:
        lines.append(f"未找到 Phase 1 素材分析结果（{RESOURCE_SPLIT_CLIPS_DIR}/），请先运行素材分析")

    return "\n\n".join(lines), df


def save_resource_edits(df: pd.DataFrame) -> str:
    """把资源库弹窗中用户修改的「内容摘要」「动作」写回：
    - phase1_split_clips/Sxxx_config.json（资源库/素材表的数据源）
    - phase1_analysis.json（shot.action + cv_metadata.shot_config，Phase 2/3 匹配提示词读取）
    - phase2_selected_shots.json（若存在，同样结构）
    """
    if df is None or df.empty:
        return "资源表为空，未保存"
    split_dir = _output_path(os.path.basename(RESOURCE_SPLIT_CLIPS_DIR))
    edits = {
        str(r.get("镜头编号", "")).strip(): (
            str(r.get("内容摘要", "") or "").strip(),
            str(r.get("动作", "") or "").strip(),
        )
        for _, r in df.iterrows()
        if str(r.get("镜头编号", "")).strip()
    }
    if not edits:
        return "未找到可保存的镜头编号"
    try:
        # 1) Sxxx_config.json
        n_cfg = 0
        for sid, (summary, action) in edits.items():
            cfg_path = os.path.join(split_dir, f"{sid}_config.json")
            if not os.path.exists(cfg_path):
                continue
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["content_summary"] = summary
            cfg["action"] = action
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            n_cfg += 1

        # 2) 分析库 JSON（phase1_analysis.json / phase2_selected_shots.json）
        def _apply_to_shots(shots: List[Dict[str, Any]]) -> int:
            n = 0
            for s in shots:
                sid = str(s.get("shot_id", "")).strip()
                if sid not in edits:
                    continue
                summary, action = edits[sid]
                s["action"] = action
                sc = (s.get("cv_metadata") or {}).get("shot_config")
                if isinstance(sc, dict):
                    sc["content_summary"] = summary
                    sc["action"] = action
                n += 1
            return n

        n_ana = 0
        for name in ("phase1_analysis.json", "phase2_selected_shots.json"):
            path = _output_path(name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            n_ana += _apply_to_shots(data.get("shots", []))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return f"已保存 {n_cfg} 个镜头描述（配置文件 {n_cfg} 条，分析库 {n_ana} 条），③ 素材表已同步"
    except Exception as e:
        return f"保存失败: {e}\n{traceback.format_exc()}"


# ----------------------------------------------------------------------
# 角色参考图管理（workspace/refs + refs.json 清单）
# 清单条目: {"file": "相对文件名", "name": "角色名", "description": "说明"}
# 说明文字会注入本地 VLM 的 prompt，帮助角色对照分析。
# ----------------------------------------------------------------------

REFS_MANIFEST_NAME = "refs.json"
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _refs_dir() -> str:
    path = _workspace_path("refs")
    os.makedirs(path, exist_ok=True)
    return path


def _refs_manifest_path() -> str:
    return os.path.join(_refs_dir(), REFS_MANIFEST_NAME)


def load_refs_entries() -> List[Dict[str, Any]]:
    """读取参考图清单；无清单时回退扫描 refs/ 目录（子目录=角色名 或 角色名_角度.jpg）。"""
    entries: List[Dict[str, Any]] = []
    manifest = _refs_manifest_path()
    if os.path.exists(manifest):
        try:
            with open(manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            for e in data.get("images", []):
                file_name = str(e.get("file", "")).strip()
                if not file_name:
                    continue
                if not os.path.exists(os.path.join(_refs_dir(), file_name)):
                    continue
                entries.append({
                    "file": file_name,
                    "name": str(e.get("name", "")).strip(),
                    "description": str(e.get("description", "")).strip(),
                })
            return entries
        except Exception as e:
            logger.warning(f"读取参考图清单失败，回退目录扫描: {e}")

    base = _refs_dir()
    try:
        for entry in sorted(os.listdir(base)):
            full = os.path.join(base, entry)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    if f.lower().endswith(_IMG_EXTS):
                        entries.append({"file": os.path.join(entry, f), "name": entry, "description": ""})
            elif entry.lower().endswith(_IMG_EXTS) and entry != REFS_MANIFEST_NAME:
                stem = os.path.splitext(entry)[0]
                entries.append({"file": entry, "name": stem.split("_")[0], "description": ""})
    except Exception as e:
        logger.warning(f"扫描参考图目录失败: {e}")
    return entries


def save_refs_entries(entries: List[Dict[str, Any]]):
    with open(_refs_manifest_path(), "w", encoding="utf-8") as f:
        json.dump({"version": 1, "images": entries}, f, ensure_ascii=False, indent=2)


def load_refs_ui() -> Tuple[List, pd.DataFrame, str]:
    """参考图弹窗内容：画廊值、可编辑表格、说明文字"""
    entries = load_refs_entries()
    gallery, rows = [], []
    for i, e in enumerate(entries):
        full = os.path.join(_refs_dir(), e["file"])
        caption = e["name"] or os.path.basename(e["file"])
        if e["description"]:
            caption += "｜" + (e["description"][:36] + ("…" if len(e["description"]) > 36 else ""))
        gallery.append((full, f"#{i + 1} {caption}"))
        rows.append({"序号": i + 1, "文件": e["file"], "角色名": e["name"], "说明": e["description"]})
    md = (
        f"已有 {len(entries)} 张参考图。点击画廊中的图可选中（用于删除）；"
        "在下方表格修改「角色名 / 说明」后点保存。说明会注入 Phase 1 VLM 分析 prompt。"
    )
    return gallery, pd.DataFrame(rows), md


def add_ref_image(file_path: str, name: str, description: str) -> str:
    """新增参考图：拷贝文件到 workspace/refs 并写入清单"""
    if not file_path or not os.path.exists(file_path):
        return "请先选择图片文件"
    name = (name or "").strip()
    if not name:
        return "请填写角色名"
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _IMG_EXTS:
        return f"不支持的图片格式: {ext}"
    # 先读清单再拷贝文件，避免无清单时目录扫描把刚拷贝的文件当成旧条目
    entries = load_refs_entries()
    base = _refs_dir()
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", name)
    file_name, n = f"{stem}{ext}", 1
    while os.path.exists(os.path.join(base, file_name)):
        n += 1
        file_name = f"{stem}_{n}{ext}"
    try:
        shutil.copy2(file_path, os.path.join(base, file_name))
    except Exception as e:
        return f"图片拷贝失败: {e}"
    entries.append({"file": file_name, "name": name, "description": (description or "").strip()})
    save_refs_entries(entries)
    return f"已新增参考图 #{len(entries)}：{name}"


def save_refs_edits(df: pd.DataFrame) -> str:
    """把弹窗表格里的「角色名 / 说明」按行序写回清单（不支持增删行，增删用专门按钮）"""
    entries = load_refs_entries()
    if df is None or df.empty:
        return "参考图表为空，未保存"
    if len(df) != len(entries):
        return f"行数不一致（表 {len(df)} 行 / 清单 {len(entries)} 条），请在表格中撤销增删行后重试"
    for (_, row), e in zip(df.iterrows(), entries):
        e["name"] = str(row.get("角色名", "") or "").strip()
        e["description"] = str(row.get("说明", "") or "").strip()
    save_refs_entries(entries)
    return f"已保存 {len(entries)} 条参考图信息"


def delete_ref_image(index: int) -> str:
    """按序号删除参考图（清单条目 + 物理文件）"""
    entries = load_refs_entries()
    if not (0 <= index < len(entries)):
        return f"无效的序号：{index + 1}（共 {len(entries)} 张）"
    e = entries.pop(index)
    try:
        full = os.path.join(_refs_dir(), e["file"])
        if os.path.exists(full):
            os.remove(full)
    except Exception as ex:
        return f"清单已更新，但删除文件失败: {ex}"
    save_refs_entries(entries)
    return f"已删除 #{index + 1}：{e['name'] or e['file']}（剩余 {len(entries)} 张）"


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

CUSTOM_CSS = """
.prog-wrap { height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden; }
.prog-inner { height: 100%; background: #f97316; border-radius: 3px; width: 0%; transition: width .4s; }
.prog-inner.running { width: 30%; animation: prog-slide 1.1s ease-in-out infinite; }
@keyframes prog-slide { 0% { margin-left: -30%; } 100% { margin-left: 100%; } }
.res-fab { position: fixed; right: 10px; top: 42%; z-index: 90; writing-mode: vertical-lr; letter-spacing: 2px; }
.log-fab { position: fixed; right: 10px; bottom: 14px; z-index: 90; }
.step-status { font-size: 12px; color: #6b7280; margin-top: -8px; }
.modal-wrap { position: fixed !important; inset: 0; background: rgba(0,0,0,.45); z-index: 100;
    display: flex; align-items: center; justify-content: center; padding: 24px; }
.modal-wrap.hide { display: none !important; }
.modal-wrap > .gr-group { background: var(--body-background-fill, #fff); border-radius: 12px;
    padding: 20px; width: 720px; max-width: 92vw; max-height: 84vh; overflow: auto;
    box-shadow: 0 12px 40px rgba(0,0,0,.25); }
.modal-close { margin-left: auto; }
.gr-dataframe td { white-space: normal !important; word-break: break-word; }
#beats-review-table { min-height: 560px; }
#beats-review-table td:nth-child(8), #beats-review-table th:nth-child(8) { white-space: nowrap !important; }
"""

from contextlib import contextmanager


@contextmanager
def modal(title: str):
    """模拟弹窗：Gradio 6.25 无 gr.Modal，用 fixed 定位的 Column + Group 实现。
    标题栏右侧自带「✕」关闭按钮，返回 (容器, 关闭按钮)。"""
    with gr.Column(visible=False, elem_classes="modal-wrap") as col:
        with gr.Group():
            with gr.Row():
                gr.Markdown(f"### {title}")
                close = gr.Button("✕", size="sm", scale=0, min_width=36,
                                  elem_classes="modal-close")
            yield col, close


def _progress_html() -> str:
    if SESSION.running:
        label = SESSION.run_label or f"{PHASE_LABELS.get(SESSION.run_step, '')} 运行中…"
        return (
            '<div class="prog-wrap"><div class="prog-inner running"></div></div>'
            f'<div style="font-size:12px;color:#9a3412;margin-top:2px">▶ {label}</div>'
        )
    return '<div class="prog-wrap"><div class="prog-inner" style="width:100%;background:#22c55e"></div></div>'


def _log_summary() -> str:
    lines = [l for l in SESSION.log if l.strip()]
    last = lines[-1] if lines else "暂无日志"
    if len(last) > 60:
        last = last[:60] + "…"
    badge = "🔴" if (SESSION.run_done and not SESSION.run_ok) else ("🟢" if SESSION.run_done else "🟠")
    return f"{badge} {last}"


def build_ui():
    with gr.Blocks(title="LLM-AutoCut") as demo:

        # ---------------- 顶栏 ----------------
        with gr.Row():
            gr.Markdown("## 🎬 LLM-AutoCut 智能剪辑工作台", scale=4)
            btn_config = gr.Button("📄 配置", scale=0, min_width=80)
            btn_refs = gr.Button("🖼 参考图", scale=0, min_width=90)
            btn_apikey = gr.Button("🔑 API Key", scale=0, min_width=90)
            btn_log = gr.Button("📋 日志", scale=0, min_width=80, elem_classes="log-fab")

        # 细进度条（内容区顶部，运行时可见动画，完成变绿）
        progress_html = gr.HTML(_progress_html())

        with gr.Row():
            # ---------------- 左侧步骤导航 ----------------
            with gr.Sidebar(open=True, width="200px"):
                nav = gr.Radio(
                    choices=["① 准备", "② 剧本分析", "③ 镜头筛选", "④ 剪辑导出"],
                    value="① 准备",
                    label="步骤",
                )
                st1 = gr.Markdown("—", elem_classes="step-status")
                st2 = gr.Markdown("—", elem_classes="step-status")
                st3 = gr.Markdown("—", elem_classes="step-status")
                st4 = gr.Markdown("—", elem_classes="step-status")

            # ---------------- 右侧资源弹窗按钮 ----------------
            btn_resource = gr.Button("📁\n资\n源", scale=0, min_width=44, elem_classes="res-fab")

            # ---------------- 步骤面板 ----------------
            with gr.Column(scale=8):

                # ===== ① 准备 =====
                with gr.Column(visible=True) as panel1:
                    with gr.Row():
                        run1 = gr.Button("▶ 准备工作区", variant="primary", scale=0)
                        prep_status = gr.Textbox(label="状态", interactive=False, scale=3)
                    with gr.Row():
                        raw_upload = gr.File(
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                            label="上传 RAW 素材",
                        )
                    materials_dir = gr.Textbox(
                        label="素材库文件夹路径（可选）",
                        placeholder="D:\\素材库\\项目名",
                        lines=1,
                    )
                    prep_summary = gr.Markdown("")

                # ===== ② 剧本分析 =====
                with gr.Column(visible=False) as panel2:
                    with gr.Row():
                        run2 = gr.Button("▶ 运行剧本分析（Phase 2）", variant="primary", scale=0)
                        save2 = gr.Button("保存修改", scale=0)
                        change2 = gr.Button("更改剧本", scale=0)
                        script_status = gr.Textbox(label="状态", interactive=False, scale=3)
                    with gr.Row():
                        target_dur = gr.Number(
                            label="目标总时长（秒）· 用户输入，AI 按剧情配平",
                            minimum=5, step=5, scale=0, min_width=160,
                        )
                        apply_dur = gr.Button("应用时长（AI 重新分配）", scale=0)
                        dur_status = gr.Textbox(label="时长状态", interactive=False, scale=3)

                    # 编辑态：无分析结果 / 用户点「更改剧本」
                    with gr.Column(visible=True) as script_edit_view:
                        with gr.Row():
                            script_upload = gr.File(
                                file_count="single",
                                file_types=[".md", ".txt", ".docx", ".pdf"],
                                label="导入剧本（自动解析为台本）",
                            )
                            parse_btn = gr.Button("解析为台本", variant="secondary", scale=0)
                        script_editor = gr.Code(
                            value=DEFAULT_SCRIPT, language="markdown",
                            label="剧本编写（可直接编辑）", lines=22,
                        )
                        with gr.Row():
                            split_anchor = gr.Textbox(
                                label="一键分镜头：填入定位文字（节点内容中的几个字）",
                                placeholder="例如：黑猫跃起",
                                scale=3,
                            )
                            split_btn = gr.Button("✂️ 在此拆分镜头节点", variant="secondary", scale=0)
                        split_msg = gr.Markdown("")

                    # 审核态：可编辑节点表（全宽）
                    with gr.Column(visible=False) as script_review_view:
                        beats_md = gr.Markdown("")
                        beats_df = gr.DataFrame(
                            label="剧本审核节点（可编辑 · 改完点上方「保存修改」直接生效，不走 AI）",
                            interactive=True, wrap=True, elem_id="beats-review-table",
                            max_height=620,
                            column_widths=[110, 130, 130, 350, 130, 150, 350, 95, 60, 65],
                        )

                # ===== ③ 镜头筛选 =====
                with gr.Column(visible=False) as panel3:
                    with gr.Row():
                        run3 = gr.Button("▶ 运行镜头筛选（Phase 3）", variant="primary", scale=0)
                        save3 = gr.Button("保存镜头分配", scale=0)
                        match_status = gr.Textbox(label="状态", interactive=False, scale=3)
                    with gr.Row():
                        with gr.Column(scale=1):
                            beats_shots_df = gr.DataFrame(
                                label="剧本节点表（左侧 · 含镜头栏）",
                                interactive=False, wrap=True,
                                column_widths=[110, 340, 80, 180, 260, 60],
                            )
                        with gr.Column(scale=1):
                            materials_df = gr.DataFrame(
                                label="素材表（Phase 1 分析结果）",
                                interactive=False, wrap=True,
                                column_widths=[70, 160, 60, 60, 80, 230, 180, 60],
                            )
                            assign_df = gr.DataFrame(
                                label="镜头归属分配（改「归属情节点」列后保存）",
                                interactive=True, wrap=True,
                                column_widths=[70, 160, 110, 70],
                            )

                # ===== ④ 剪辑导出 =====
                with gr.Column(visible=False) as panel4:
                    with gr.Row():
                        run4 = gr.Button("▶ 运行剪辑导出（Phase 4）", variant="primary", scale=0)
                        export_status = gr.Textbox(label="状态", interactive=False, scale=3)
                    with gr.Row():
                        with gr.Column(scale=1):
                            beats_ro4_df = gr.DataFrame(
                                label="剧本节点表", interactive=False, wrap=True,
                                column_widths=[110, 340, 80, 180, 260, 60],
                            )
                        with gr.Column(scale=2):
                            final_video = gr.Video(label="成片预览")
                            export_summary = gr.Textbox(label="决策概览", interactive=False, lines=2)
                    with gr.Row():
                        with gr.Column(scale=1):
                            products_df = gr.DataFrame(
                                label="导出产物（可单独下载）", interactive=False, wrap=True,
                                column_widths=[260],
                            )
                            product_pick = gr.Dropdown(label="选择产物下载", choices=[], interactive=True)
                            product_file = gr.File(label="产物文件")
                        with gr.Column(scale=1):
                            used_md = gr.Markdown("**成片用到的素材**")
                            used_pick = gr.Dropdown(label="选择素材下载", choices=[], interactive=True)
                            used_file = gr.File(label="素材文件")

        # ---------------- 弹窗 ----------------
        with modal("配置（config.yaml）") as (config_modal, config_close):
            config_editor = gr.Code(language="yaml", label="config.yaml", lines=26)
            with gr.Row():
                config_save = gr.Button("保存配置", variant="primary")
                config_msg = gr.Markdown("")

        with modal("API Key（仅保存在当前会话，运行时注入配置）") as (apikey_modal, key_close):
            vlm_key_input = gr.Textbox(label="VLM Key（豆包 ARK / OpenAI 等）", type="password")
            llm_key_input = gr.Textbox(label="LLM Key（DeepSeek 等）", type="password")
            with gr.Row():
                key_save = gr.Button("保存", variant="primary")
                key_msg = gr.Markdown("")

        with modal("运行日志") as (log_modal, log_close):
            log_code = gr.Code(label="日志（自动滚动到底部）", language="markdown", lines=30, interactive=False)

        with modal("素材资源库（Phase 0 / Phase 1 产物）") as (resource_modal, resource_close):
            resource_md = gr.Markdown("")
            resource_df = gr.DataFrame(
                interactive=True, wrap=True,
                column_widths=[70, 200, 160, 60, 60, 230, 180, 60],
            )
            with gr.Row():
                resource_save = gr.Button("💾 保存修改（内容摘要 / 动作）", variant="primary", scale=0)
                resource_msg = gr.Markdown("", scale=3)

        with modal("角色参考图管理（注入 Phase 1 VLM 分析）") as (refs_modal, refs_close):
            refs_md = gr.Markdown("")
            refs_gallery = gr.Gallery(
                label="现有参考图（点击选中）", columns=6, rows=2,
                object_fit="contain", height="auto",
            )
            refs_sel_state = gr.State(-1)
            refs_df = gr.DataFrame(
                interactive=True, wrap=True,
                label="角色名 / 说明（改完点保存）",
                column_widths=[50, 180, 120, 360],
            )
            with gr.Row():
                refs_save = gr.Button("💾 保存修改（角色名 / 说明）", variant="primary", scale=0)
                refs_del = gr.Button("🗑 删除选中图", variant="stop", scale=0)
                refs_msg = gr.Markdown("", scale=3)
            with gr.Row():
                refs_upload = gr.File(
                    file_count="single",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    label="上传新参考图",
                )
                refs_name = gr.Textbox(label="角色名", placeholder="例如：云琛-女装", scale=0, min_width=140)
                refs_desc = gr.Textbox(
                    label="说明（会注入分析 prompt）",
                    placeholder="例如：男主换装后的形象，青衫长发，夜戏",
                    scale=3,
                )
                refs_add = gr.Button("➕ 新增", variant="primary", scale=0)

        with modal("确认更改剧本？") as (confirm_modal, confirm_close):
            gr.Markdown("已有剧本分析结果会被保留，重新分析后覆盖。确认进入剧本编辑？")
            with gr.Row():
                confirm_yes = gr.Button("确认更改", variant="stop")
                confirm_no = gr.Button("取消")

        timer = gr.Timer(1.0, active=True)

        # ==================================================================
        # 事件绑定
        # ==================================================================

        run_btns = [run1, run2, run3, run4]

        def _btns_state(interactive: bool):
            return [gr.update(interactive=interactive) for _ in run_btns]

        def _script_views():
            """决定 ② 剧本页显示编辑态还是审核态"""
            _, _, _, has = load_beats_data()
            if SESSION.force_edit_view:
                return gr.update(visible=True), gr.update(visible=False)
            if has:
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=True), gr.update(visible=False)

        def _refresh_values():
            """加载全部步骤数据，返回 dict"""
            beats_df_v, sub_df_v, beats_summary, has_analysis = load_beats_data()
            beats_ro4_v = load_beats_readonly_with_shots() if has_analysis else pd.DataFrame()
            beats_shots_v = beats_ro4_v
            materials_v = load_materials_data()
            assign_v = load_anchor_assignment()
            products_v = load_products()
            used = load_used_materials()
            video_path = _output_path("final_with_dubbing.mp4")

            n_generated = sum(1 for _, n in PRODUCT_FILES if os.path.exists(_output_path(n)))
            export_summary_v = ""
            tpath = _output_path("timeline.json")
            if os.path.exists(tpath):
                try:
                    with open(tpath, "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                    export_summary_v = f"成片镜头 {tdata.get('total_clips', len(tdata.get('timeline', [])))} 个 · 产物 {n_generated}/{len(PRODUCT_FILES)}"
                except Exception:
                    pass
            if os.path.exists(video_path):
                dur = 0.0
                try:
                    r = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                        capture_output=True, text=True, check=False)
                    dur = float(r.stdout.strip() or 0)
                except Exception:
                    pass
                if dur:
                    export_summary_v += f" · 成片时长 {dur:.1f}s"

            # 侧栏状态
            script_ok = os.path.exists(_workspace_path("script.md"))
            matched = os.path.exists(_output_path("phase2_selected_shots.json"))
            st1_v = "✓ 工作区就绪" if script_ok else "待准备：上传素材并运行"
            st2_v = f"✓ {beats_summary}" if has_analysis else "未分析"
            if matched:
                try:
                    with open(_output_path("phase2_selected_shots.json"), "r", encoding="utf-8") as f:
                        n_shots = len(json.load(f).get("shots", []))
                    st3_v = f"✓ 已筛选（{n_shots} 镜头）"
                except Exception:
                    st3_v = "✓ 已筛选"
            else:
                st3_v = "未运行"
            st4_v = "✓ 成片已生成" if os.path.exists(video_path) else "未导出"

            edit_vis, review_vis = (gr.update(visible=True), gr.update(visible=False)) \
                if SESSION.force_edit_view or not has_analysis else \
                (gr.update(visible=False), gr.update(visible=True))

            return {
                "progress": _progress_html(),
                "logfab": f"📋 日志\n{_log_summary()[:40]}",
                "log_code": "\n".join(SESSION.log),
                "prep_status": SESSION.status_msgs["prep"],
                "script_status": SESSION.status_msgs["script"],
                "match_status": SESSION.status_msgs["match"],
                "export_status": SESSION.status_msgs["export"],
                "st1": st1_v, "st2": st2_v, "st3": st3_v, "st4": st4_v,
                "beats_df": beats_df_v,
                "beats_md": beats_summary,
                "edit_vis": edit_vis, "review_vis": review_vis,
                "beats_shots": beats_shots_v,
                "beats_ro4": beats_ro4_v,
                "materials": materials_v, "assign": assign_v,
                "video": gr.update(value=video_path if os.path.exists(video_path) else None),
                "export_summary": export_summary_v,
                "products": products_v,
                "product_choices": gr.update(choices=existing_product_names()),
                "used_choices": gr.update(choices=[n for n, _ in used]),
                "used_md": "**成片用到的素材**" + ("" if used else "\n\n（尚未生成剪辑时间线）"),
                "run_btns": _btns_state(not SESSION.running),
            }

        TICK_KEYS = [
            # 总是刷新区（进度/日志/各步骤状态，运行中也要可见）
            "progress", "logfab", "log_code",
            "prep_status", "script_status", "match_status", "export_status",
            # 数据区（仅运行完成时刷新，避免表格/视频闪烁）
            "st1", "st2", "st3", "st4",
            "beats_df", "beats_md",
            "edit_vis", "review_vis",
            "beats_shots", "beats_ro4", "materials", "assign",
            "video", "export_summary", "products", "product_choices", "used_choices", "used_md",
            "b1", "b2", "b3", "b4",
        ]

        TICK_COMPONENTS = [
            progress_html, btn_log, log_code,
            prep_status, script_status, match_status, export_status,
            st1, st2, st3, st4,
            beats_df, beats_md,
            script_edit_view, script_review_view,
            beats_shots_df, beats_ro4_df, materials_df, assign_df,
            final_video, export_summary, products_df, product_pick, used_pick, used_md,
            run1, run2, run3, run4,
        ]

        # 总是刷新区的组件个数（progress/logfab/log_code + 4 个状态）
        TICK_ALWAYS = 7
        # 运行完成时，把结果写进步骤状态栏：TICK_KEYS 中的下标
        TICK_STEP_STATUS_IDX = {"2": 4, "3": 5, "4": 6}

        def on_tick():
            full = SESSION.run_done
            step = ""
            result_msg = ""
            if full:
                SESSION.run_done = False
                step = SESSION.run_step
                result_msg = "运行完成 ✓" if SESSION.run_ok else "运行失败，请展开日志查看"
                if step == "2" and SESSION.run_ok:
                    SESSION.force_edit_view = False
                SESSION.run_step = ""
                SESSION.log.append(f"[状态] {result_msg}")
            vals = _refresh_values()
            out = [vals[k] for k in TICK_KEYS[:-4]]  # 末尾 4 项是按钮，由 run_btns 提供
            if full:
                idx = TICK_STEP_STATUS_IDX.get(step)
                if idx is not None:
                    out[idx] = result_msg
                    status_key = {"2": "script", "3": "match", "4": "export"}.get(step)
                    if status_key:
                        SESSION.status_msgs[status_key] = result_msg
            out += vals["run_btns"]
            # 非完成时刻，数据类组件不刷新，避免表格/视频闪烁
            if not full:
                for i in range(TICK_ALWAYS, len(out) - 4):
                    out[i] = gr.update()
            return out

        timer.tick(fn=on_tick, outputs=TICK_COMPONENTS)

        # ---------------- 步骤切换 ----------------
        def on_nav(step):
            vis = [gr.update(visible=(step == s)) for s in
                   ["① 准备", "② 剧本分析", "③ 镜头筛选", "④ 剪辑导出"]]
            vals = _refresh_values()
            return vis + [
                vals["st1"], vals["st2"], vals["st3"], vals["st4"],
                vals["beats_df"],
                vals["beats_md"], vals["edit_vis"], vals["review_vis"],
                vals["beats_shots"], vals["beats_ro4"], vals["materials"], vals["assign"],
                vals["video"], vals["export_summary"], vals["products"],
                vals["product_choices"], vals["used_choices"], vals["used_md"],
            ]

        nav.change(
            fn=on_nav,
            inputs=[nav],
            outputs=[
                panel1, panel2, panel3, panel4,
                st1, st2, st3, st4,
                beats_df,
                beats_md, script_edit_view, script_review_view,
                beats_shots_df, beats_ro4_df, materials_df, assign_df,
                final_video, export_summary, products_df,
                product_pick, used_pick, used_md,
            ],
        )

        # ---------------- ① 准备 ----------------
        def on_prepare(raw_files, materials_dir):
            if SESSION.running:
                return "已有任务在运行", ""
            msg = prepare_workspace_files(raw_files or [], materials_dir or "")
            SESSION.status_msgs["prep"] = msg
            return msg, msg.replace("\n", "\n\n")

        run1.click(
            fn=on_prepare,
            inputs=[raw_upload, materials_dir],
            outputs=[prep_status, prep_summary],
        )

        # ---------------- ② 剧本 ----------------
        def on_run2():
            msg = start_step_run("2")
            SESSION.status_msgs["script"] = msg
            if msg.endswith("已开始运行"):
                SESSION.force_edit_view = False
                return msg, *_btns_state(False)
            return msg, *_btns_state(True)

        run2.click(fn=on_run2, outputs=[script_status, run1, run2, run3, run4])

        def on_save2(d):
            """直接保存用户手改的节点表（时长/内容原样生效，不走 AI 分配）"""
            msg = save_beats(d, None)
            SESSION.status_msgs["script"] = msg
            vals = _refresh_values()
            return msg, vals["beats_df"], vals["beats_md"]

        save2.click(
            fn=on_save2,
            inputs=[beats_df],
            outputs=[script_status, beats_df, beats_md],
        )

        def on_apply_dur(v):
            msg = apply_target_duration(v)
            vals = _refresh_values()
            return msg, vals["beats_df"], vals["beats_md"]

        apply_dur.click(
            fn=on_apply_dur,
            inputs=[target_dur],
            outputs=[dur_status, beats_df, beats_md],
        )

        script_upload.change(
            fn=on_script_upload,
            inputs=[script_upload],
            outputs=[script_editor, script_status],
        )

        parse_btn.click(
            fn=preprocess_script,
            inputs=[script_editor, script_upload],
            outputs=[script_editor, script_status],
        )

        def on_split_beat(script_text, anchor):
            """一键分镜头：在包含定位文字的行后插入一个锁定的空分节点，交给 AI 补全"""
            anchor = (anchor or "").strip()
            if not script_text or not script_text.strip():
                return script_text, "台本为空，无法拆分"
            if not anchor:
                return script_text, "请先填入定位文字（要拆分位置对应的几个字）"
            lines = script_text.split("\n")
            hit = next((idx for idx, line in enumerate(lines) if anchor in line), -1)
            if hit < 0:
                return script_text, f"未找到「{anchor}」，多打几个字再试"
            n = 1
            while f"### 分节点·{n}" in script_text:
                n += 1
            node = [
                "",
                f"### 分节点·{n}",
                "- 标记：锁定",
                "（此节点内容留空，AI 重新分析时将依据上下文补全；"
                "也可在此行下加「- 内容：...」直接填写）",
                "",
            ]
            new_text = "\n".join(lines[:hit + 1] + node + lines[hit + 1:])
            return new_text, f"已插入「分节点·{n}」（锁定）。确认后点「解析为台本」保存，再运行剧本分析生效"

        split_btn.click(
            fn=on_split_beat,
            inputs=[script_editor, split_anchor],
            outputs=[script_editor, split_msg],
        )

        change2.click(
            fn=lambda: gr.update(visible=True),
            outputs=[confirm_modal],
        )

        def on_confirm_change():
            SESSION.force_edit_view = True
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        confirm_yes.click(
            fn=on_confirm_change,
            outputs=[confirm_modal, script_edit_view, script_review_view],
        )
        confirm_no.click(fn=lambda: gr.update(visible=False), outputs=[confirm_modal])

        # ---------------- ③ 镜头筛选 ----------------
        def on_run3():
            msg = start_step_run("3")
            SESSION.status_msgs["match"] = msg
            return (msg, *_btns_state(not SESSION.running))

        run3.click(fn=on_run3, outputs=[match_status, run1, run2, run3, run4])

        def on_save3(df):
            msg = save_anchor_assignment(df)
            SESSION.status_msgs["match"] = msg
            vals = _refresh_values()
            return msg, vals["beats_shots"], vals["assign"]

        save3.click(fn=on_save3, inputs=[assign_df], outputs=[match_status, beats_shots_df, assign_df])

        # ---------------- ④ 剪辑导出 ----------------
        def on_run4():
            msg = start_step_run("4")
            SESSION.status_msgs["export"] = msg
            return (msg, *_btns_state(not SESSION.running))

        run4.click(fn=on_run4, outputs=[export_status, run1, run2, run3, run4])

        product_pick.change(
            fn=lambda name: _output_path(name) if name else None,
            inputs=[product_pick],
            outputs=[product_file],
        )

        def on_pick_used(name):
            if not name:
                return None
            for n, p in load_used_materials():
                if n == name:
                    return p
            return None

        used_pick.change(fn=on_pick_used, inputs=[used_pick], outputs=[used_file])

        # ---------------- 弹窗 ----------------
        for btn, modal_box in [
            (config_close, config_modal),
            (key_close, apikey_modal),
            (log_close, log_modal),
            (resource_close, resource_modal),
            (refs_close, refs_modal),
            (confirm_close, confirm_modal),
        ]:
            btn.click(fn=lambda m=modal_box: gr.update(visible=False), outputs=[modal_box])

        def open_config():
            text = _read_text(_workspace_path("config.yaml")) or DEFAULT_CONFIG
            return gr.update(visible=True), text

        btn_config.click(fn=open_config, outputs=[config_modal, config_editor])

        def save_config(text):
            try:
                yaml.safe_load(text)
            except Exception as e:
                return f"配置格式错误: {e}"
            try:
                with open(_workspace_path("config.yaml"), "w", encoding="utf-8") as f:
                    f.write(text)
                return "已保存到 workspace/config.yaml"
            except Exception as e:
                return f"保存失败: {e}"

        config_save.click(fn=save_config, inputs=[config_editor], outputs=[config_msg])

        btn_apikey.click(
            fn=lambda: (gr.update(visible=True), SESSION.vlm_key, SESSION.llm_key),
            outputs=[apikey_modal, vlm_key_input, llm_key_input],
        )

        def save_keys(v, l):
            SESSION.vlm_key = (v or "").strip()
            SESSION.llm_key = (l or "").strip()
            ok = []
            if SESSION.vlm_key:
                ok.append("VLM ✓")
            if SESSION.llm_key:
                ok.append("LLM ✓")
            return "已保存: " + ("、".join(ok) if ok else "（为空，将使用配置/环境变量中的 Key）")

        key_save.click(fn=save_keys, inputs=[vlm_key_input, llm_key_input], outputs=[key_msg])

        btn_log.click(
            fn=lambda: (gr.update(visible=True), "\n".join(SESSION.log)),
            outputs=[log_modal, log_code],
        )

        def open_resource():
            md, df = load_resource_library()
            return gr.update(visible=True), md, df

        btn_resource.click(fn=open_resource, outputs=[resource_modal, resource_md, resource_df])

        def on_save_resource(df):
            msg = save_resource_edits(df)
            vals = _refresh_values()
            return msg, vals["materials"]

        resource_save.click(
            fn=on_save_resource,
            inputs=[resource_df],
            outputs=[resource_msg, materials_df],
        )

        # ---------------- 参考图弹窗 ----------------
        def open_refs():
            gallery, df, md = load_refs_ui()
            return gr.update(visible=True), gallery, df, md, -1, ""

        btn_refs.click(
            fn=open_refs,
            outputs=[refs_modal, refs_gallery, refs_df, refs_md, refs_sel_state, refs_msg],
        )

        def on_refs_select(evt: gr.SelectData):
            idx = evt.index if evt.index is not None else -1
            return idx, (f"已选中第 {idx + 1} 张，可点「删除选中图」移除" if idx >= 0 else "")

        refs_gallery.select(fn=on_refs_select, outputs=[refs_sel_state, refs_msg])

        def on_refs_add(file, name, desc):
            msg = add_ref_image(file.name if file is not None else "", name, desc)
            gallery, df, _ = load_refs_ui()
            return msg, gallery, df, None, "", ""

        refs_add.click(
            fn=on_refs_add,
            inputs=[refs_upload, refs_name, refs_desc],
            outputs=[refs_msg, refs_gallery, refs_df, refs_upload, refs_name, refs_desc],
        )

        def on_refs_save(df):
            msg = save_refs_edits(df)
            gallery, df_v, _ = load_refs_ui()
            return msg, gallery, df_v

        refs_save.click(fn=on_refs_save, inputs=[refs_df], outputs=[refs_msg, refs_gallery, refs_df])

        def on_refs_delete(idx):
            msg = delete_ref_image(int(idx) if idx is not None else -1)
            gallery, df, _ = load_refs_ui()
            return msg, gallery, df, -1

        refs_del.click(
            fn=on_refs_delete,
            inputs=[refs_sel_state],
            outputs=[refs_msg, refs_gallery, refs_df, refs_sel_state],
        )

        # ---------------- 初始加载 ----------------
        def on_load():
            vals = _refresh_values()
            script_editor_v = _read_text(_workspace_path("script.md")) or DEFAULT_SCRIPT
            return [
                vals["st1"], vals["st2"], vals["st3"], vals["st4"],
                vals["beats_df"],
                vals["beats_md"], vals["edit_vis"], vals["review_vis"],
                vals["beats_shots"], vals["beats_ro4"], vals["materials"], vals["assign"],
                vals["video"], vals["export_summary"], vals["products"],
                vals["product_choices"], vals["used_choices"], vals["used_md"],
                progress_html, script_editor_v,
                read_target_duration() or None,
            ]

        demo.load(
            fn=on_load,
            outputs=[
                st1, st2, st3, st4,
                beats_df,
                beats_md, script_edit_view, script_review_view,
                beats_shots_df, beats_ro4_df, materials_df, assign_df,
                final_video, export_summary, products_df,
                product_pick, used_pick, used_md,
                progress_html, script_editor,
                target_dur,
            ],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LLM-AutoCut Web UI")
    parser.add_argument("--config", default=None, help="预加载配置文件路径")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0", server_port=args.port, share=args.share,
        css=CUSTOM_CSS, theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
