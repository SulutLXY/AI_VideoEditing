# -*- coding: utf-8 -*-
"""
LLM-AutoCut Web UI

基于 Gradio 的快速原型界面，用于：
- 编辑配置、上传素材、编写剧本大纲
- 分阶段运行 pipeline
- 查看日志和输出结果
- 下载生成的 EDL / FCPXML / CSV / JSON

启动方式:
    python webui.py

可选参数:
    python webui.py --config config/config.yaml --port 7860
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import yaml

from src.utils import load_config, parse_script_outline, logger, ensure_dir, fallback_vlm_to_local
from src.phase2_anchor_editor import export_anchor_corrections, apply_anchor_corrections

import pandas as pd


# 默认配置模板
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
  - path: "./materials/processed/"
    state: "PROCESSED"
    split: false
    analyze: true
  - path: "./materials/analyzed/"
    state: "ANALYZED"
    split: false
    analyze: false
    meta_format: "autocut_v1"

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
  # Phase 2 独立运行时可直接指定素材库目录做 CV 轻量清点
  materials_dir: ""
  allow_cv_inventory: true

script_preprocessing:
  enabled: true
  provider: "deepseek"
  max_chunk_chars: 4000
  output_shot_requirements: true

models:
  vlm:
    # 豆包示例：provider="doubao", model="doubao-seed-2-0-pro-260215"
    # base_url: "https://ark.cn-beijing.volces.com/api/v3"
    provider: "doubao"
    model: "doubao-seed-2-0-pro-260215"
    api_key: ""
    base_url: "https://ark.cn-beijing.volces.com/api/v3"
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150
  video_vlm:
    provider: "vlm_fallback"
    model: "doubao-seed-2-0-pro-260215"
    max_tokens: 4096
    temperature: 0.3
    frame_sample_rate: 5
    max_frames: 150
  llm:
    # 豆包文本模型示例：model="doubao-pro-32k"
    provider: "deepseek"
    model: "deepseek-chat"
    api_key: ""
    base_url: "https://api.deepseek.com"
    max_tokens: 8192
    temperature: 0.5
  asr:
    provider: "whisper"
    model: "large-v3"
    device: "cpu"
    language: "zh"

processing:
  scene_threshold: 0.3
  min_shot_duration: 1.0
  keyframe_strategy: "adaptive"
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

DEFAULT_SCRIPT = """# 雨夜告白

## 第一幕：等待
### 场1-情节点A
- 地点：咖啡馆内
- 时间：傍晚
- 内容：男主独自等待，表现焦虑，反复看表
- 情绪：焦虑、压抑
- 关键动作：看表、深呼吸、望向门口
- 关键台词：""

### 场1-情节点B
- 地点：咖啡馆门口
- 时间：雨夜
- 内容：女主推门而入，两人对视
- 情绪：紧张 → 期待
- 关键动作：推门、对视、停顿
- 关键台词：""
"""


class SessionState:
    """简单的会话状态管理"""
    def __init__(self):
        # 默认指向项目内 workspace，避免页面刷新后 SESSION 丢失导致面板无法加载已有产物
        self.work_dir = str(WORKSPACE_DIR)
        self.output_dir = os.path.join(self.work_dir, "output")


SESSION = SessionState()


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


def _ensure_structured_script(
    script_text: str,
    script_path: str,
    config: dict,
) -> Tuple[str, str]:
    """确保 script_text 是结构化剧本大纲；如果不是，自动调用 LLM 预处理。

    返回: (处理后的剧本文本, 状态信息)
    """
    from src.services.script_service import (
        is_structured_outline,
        ScriptPreprocessor,
        ScriptParseError,
    )
    from src.services.llm_service import LLMService
    from src.utils import parse_script_outline

    if is_structured_outline(script_text):
        return script_text, "剧本已是结构化大纲，无需预处理"

    # 非结构化剧本，尝试自动预处理
    llm_config = config.get("models", {}).get("llm", {})
    if not llm_config.get("api_key"):
        return (
            script_text,
            "警告：当前剧本不是结构化大纲，且未配置 LLM API Key，无法自动解析。"
            "请先点击「自动解析为台本」，或手动编辑为标准 Markdown 大纲后再运行。",
        )

    try:
        llm_service = LLMService(config)
        preprocessor_config = config.get("script_preprocessing", {})
        preprocessor = ScriptPreprocessor(llm_service, preprocessor_config)
        parsed = preprocessor.preprocess(script_text)

        # 二次校验
        import tempfile

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=os.path.dirname(script_path))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(parsed)
            beats = parse_script_outline(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if not beats:
            return (
                script_text,
                "警告：剧本自动解析失败，模型返回的结构化文本未能提取出任何情节点。"
                "请先点击「自动解析为台本」检查模型输出，或手动编辑为标准 Markdown 大纲后再运行。",
            )

        return (
            parsed,
            f"剧本已自动解析为结构化大纲（共 {len(beats)} 个情节点），可直接运行 Pipeline。",
        )
    except ScriptParseError as e:
        return (
            script_text,
            f"警告：剧本自动解析失败: {e} 请先点击「自动解析为台本」重试，或手动编辑为标准 Markdown 大纲。",
        )
    except Exception as e:
        return (
            script_text,
            f"警告：剧本自动解析时发生异常: {e} 请先点击「自动解析为台本」重试，或手动编辑为标准 Markdown 大纲。",
        )


def prepare_workspace(
    config_text: str,
    script_text: str,
    raw_files: list,
    processed_files: list,
    analyzed_videos: list,
    analyzed_metas: list,
    materials_dir: str = "",
    vlm_key: str = "",
    llm_key: str = "",
    clear_output: bool = True,
) -> str:
    """准备项目内工作目录，写入配置、剧本和素材，可选注入 API Key"""
    try:
        # 解析配置，确保基本结构正确
        config = yaml.safe_load(config_text)
        if not config:
            return "错误：配置为空或无法解析"

        # 如果用户填写了 API Key，注入到配置中（优先于环境变量）
        if vlm_key and vlm_key.strip():
            config.setdefault("models", {}).setdefault("vlm", {})["api_key"] = vlm_key.strip()
        if llm_key and llm_key.strip():
            config.setdefault("models", {}).setdefault("llm", {})["api_key"] = llm_key.strip()

        # 使用项目内 workspace 目录，避免存入系统临时目录
        SESSION.work_dir = str(WORKSPACE_DIR)
        SESSION.output_dir = os.path.join(SESSION.work_dir, "output")
        ensure_dir(SESSION.work_dir)
        ensure_dir(SESSION.output_dir)

        # 仅完整工作流（--all）时默认清空整个输出目录；单阶段运行保留其它阶段结果
        if clear_output and os.path.exists(SESSION.output_dir):
            shutil.rmtree(SESSION.output_dir, ignore_errors=True)
        ensure_dir(SESSION.output_dir)

        # 写入配置（先写一版，用于后续路径更新后再覆盖）
        config_path = os.path.join(SESSION.work_dir, "config.yaml")

        # 写入剧本：先尝试自动解析为结构化大纲
        script_path = os.path.join(SESSION.work_dir, "script.md")
        structured_script, script_status = _ensure_structured_script(script_text, script_path, config)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(structured_script)

        # 创建三种素材子目录
        raw_dir = os.path.join(SESSION.work_dir, "materials", "raw")
        processed_dir = os.path.join(SESSION.work_dir, "materials", "processed")
        analyzed_dir = os.path.join(SESSION.work_dir, "materials", "analyzed")

        # 仅对本次有上传的目录做清空，保留未上传状态的旧素材
        _clear_dir_if_has_upload(raw_files, raw_dir)
        _clear_dir_if_has_upload(processed_files, processed_dir)
        _clear_dir_if_has_upload(analyzed_videos, analyzed_dir)
        # analyzed_metas 也和视频一起放在 analyzed_dir，随视频目录清空即可

        uploaded = []
        uploaded.extend([f"[RAW] {n}" for n in _copy_files_to_dir(raw_files, raw_dir)])
        uploaded.extend([f"[PROCESSED] {n}" for n in _copy_files_to_dir(processed_files, processed_dir)])
        uploaded.extend([f"[ANALYZED] {n}" for n in _copy_files_to_dir(analyzed_videos, analyzed_dir)])
        uploaded.extend([f"[ANALYZED_META] {n}" for n in _copy_files_to_dir(analyzed_metas, analyzed_dir)])

        # 更新配置中的路径为绝对路径
        config["paths"]["raw_materials"] = raw_dir
        config["paths"]["output"] = SESSION.output_dir
        config["paths"]["temp"] = os.path.join(SESSION.work_dir, "temp")
        config["paths"]["script_outline"] = script_path
        config["paths"]["reference_images"] = os.path.join(SESSION.work_dir, "refs")
        # 确保 materials 配置指向工作目录
        for item in config.get("materials", []):
            if item.get("state") == "RAW":
                item["path"] = raw_dir
            elif item.get("state") == "PROCESSED":
                item["path"] = processed_dir
            elif item.get("state") == "ANALYZED":
                item["path"] = analyzed_dir

        # 若用户指定了素材库目录，写入 Phase 2 配置并覆盖 raw_materials 路径
        if materials_dir and materials_dir.strip():
            materials_dir_abs = os.path.abspath(materials_dir.strip())
            config.setdefault("phase2", {})["materials_dir"] = materials_dir_abs
            config["paths"]["raw_materials"] = materials_dir_abs

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        return (
            f"工作目录已准备: {SESSION.work_dir}\n"
            f"配置: {config_path}\n"
            f"剧本: {script_path}\n"
            f"剧本状态: {script_status}\n"
            f"已上传素材: {uploaded or '无'}\n"
            f"输出目录: {SESSION.output_dir}"
        )
    except Exception as e:
        return f"准备工作区失败: {e}\n{traceback.format_exc()}"


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


def preprocess_script(
    config_text: str,
    script_text: str,
    uploaded_file_path,
    vlm_key: str = "",
    llm_key: str = "",
):
    """调用 LLM 把原始剧本解析为结构化台本。

    优先解析用户上传的文件；若未上传文件，则解析右侧编辑器中的文本。
    """
    from src.services.script_service import ScriptPreprocessor, is_structured_outline, read_script_file, ScriptReadError, ScriptParseError
    from src.services.llm_service import LLMService
    from src.utils import parse_script_outline

    # 确保工作区存在，用于保存 script.md
    work_dir = str(WORKSPACE_DIR)
    ensure_dir(work_dir)
    script_path = os.path.join(work_dir, "script.md")

    # 优先读取上传的文件
    source_label = "编辑器内容"
    text_to_parse = script_text
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

    try:
        config = yaml.safe_load(config_text) or {}
    except Exception as e:
        return script_text, f"配置解析失败: {e}"

    # 注入用户填写的 API Key
    if vlm_key and vlm_key.strip():
        config.setdefault("models", {}).setdefault("vlm", {})["api_key"] = vlm_key.strip()
    if llm_key and llm_key.strip():
        config.setdefault("models", {}).setdefault("llm", {})["api_key"] = llm_key.strip()

    # 如果已经是结构化大纲，直接保存
    if is_structured_outline(text_to_parse):
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return text_to_parse, f"当前剧本（{source_label}）已是结构化大纲，无需解析"

    # 检查 LLM Key
    llm_config = config.get("models", {}).get("llm", {})
    if not llm_config.get("api_key"):
        return script_text, "错误：未设置 LLM API Key，无法解析剧本。请在配置中填写 models.llm.api_key。"

    try:
        llm_service = LLMService(config)
        preprocessor_config = config.get("script_preprocessing", {})
        preprocessor = ScriptPreprocessor(llm_service, preprocessor_config)
        parsed = preprocessor.preprocess(text_to_parse)

        # 二次校验：先把解析结果写到临时文件，再用 parse_script_outline 验证是否真的能提取出情节点
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
                "请检查模型输出或手动编辑成标准大纲后再试。",
            )

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(parsed)

        return parsed, f"剧本解析完成（来源：{source_label}），已生成结构化台本。建议检查并微调后再运行 Pipeline。"
    except ScriptParseError as e:
        # 解析失败时不把占位符写入 script.md，保留原始文本让用户手动调整
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return (
            script_text,
            f"剧本解析失败（来源：{source_label}）：{e}\n"
            "右侧仍保留原始文本，你可以手动编辑成标准大纲，或尝试更换 LLM 模型后再点解析。",
        )
    except Exception as e:
        # 未知异常时仍保留原始文本，避免 script.md 变成未定义状态
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(text_to_parse)
        return script_text, f"剧本解析失败: {e}\n{traceback.format_exc()}"


def run_pipeline(
    phase: str,
    config_text: str,
    script_text: str,
    raw_files: list,
    processed_files: list,
    analyzed_videos: list,
    analyzed_metas: list,
    materials_dir: str = "",
    vlm_key: str = "",
    llm_key: str = "",
) -> str:
    """运行 pipeline，使用 subprocess 隔离 main.py，避免 SystemExit 导致 Gradio 崩溃"""
    # 先准备工作区，并注入用户填写的 API Key
    prep_msg = prepare_workspace(
        config_text,
        script_text,
        raw_files,
        processed_files,
        analyzed_videos,
        analyzed_metas,
        materials_dir,
        vlm_key,
        llm_key,
        clear_output=(phase == "all"),
    )
    if prep_msg.startswith("错误"):
        return prep_msg

    config_path = os.path.join(SESSION.work_dir, "config.yaml")

    # 校验 API Key（按阶段解耦：Phase 1 需 VLM，Phase 2/3 需 LLM，Phase 4 不需要）
    try:
        config = load_config(config_path)

        # 未配置 VLM API Key 时，自动回退到本地 Qwen2.5-VL
        if fallback_vlm_to_local(config):
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, sort_keys=False)

        vlm_key = config.get("models", {}).get("vlm", {}).get("api_key")
        llm_key = config.get("models", {}).get("llm", {}).get("api_key")

        vlm_env_keys = ["OPENAI_API_KEY", "ARK_API_KEY", "VLM_API_KEY"]
        llm_env_keys = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY"]

        vlm_env_key = next((k for k in vlm_env_keys if os.getenv(k)), None)
        llm_env_key = next((k for k in llm_env_keys if os.getenv(k)), None)

        need_vlm = phase in ("all", "1")
        need_llm = phase in ("all", "2", "3")
        # Phase 0 纯 CV 粗剪不需要任何 API Key

        # 如果启用了本地模型，VLM 使用本地 Qwen2.5-VL，不需要在线 API Key
        local_cfg = config.get("models", {}).get("local", {})
        vlm_provider = config.get("models", {}).get("vlm", {}).get("provider", "openai")
        if bool(local_cfg.get("enabled", False)) and vlm_provider == "local":
            need_vlm = False

        if need_vlm and not vlm_key and not vlm_env_key:
            return (
                f"{prep_msg}\n\n错误：未设置 VLM API Key。\n"
                f"请在 config.yaml 中填写 models.vlm.api_key，或在系统环境变量中设置：\n"
                f"  - OPENAI_API_KEY（OpenAI 官方）\n"
                f"  - ARK_API_KEY（豆包/火山方舟）\n"
                f"  - VLM_API_KEY（通用）"
            )
        if need_llm and not llm_key and not llm_env_key:
            return (
                f"{prep_msg}\n\n错误：未设置 LLM API Key。\n"
                f"请在 config.yaml 中填写 models.llm.api_key，或在系统环境变量中设置：\n"
                f"  - DEEPSEEK_API_KEY（DeepSeek）\n"
                f"  - OPENAI_API_KEY（OpenAI 官方/兼容）\n"
                f"  - LLM_API_KEY（通用）"
            )
    except Exception as e:
        return f"{prep_msg}\n\n配置校验失败: {e}"

    if phase == "all":
        phase_arg = ["--all"]
        clean_arg = ["--clean"]
    else:
        phase_arg = ["--phase", str(int(phase))]
        # 单阶段运行时让 main.py 只清理本阶段产物，保留其它阶段结果
        clean_arg = ["--clean"]
    cmd = [sys.executable, "main.py", "--config", config_path] + phase_arg + clean_arg

    # Phase 2 独立运行时传入素材库目录
    if phase == "2" and materials_dir and materials_dir.strip():
        cmd += ["--materials-dir", os.path.abspath(materials_dir.strip())]

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

        log_lines = []
        for line in process.stdout:
            log_lines.append(line.rstrip())
            # 限制内存占用，保留最近 500 行
            if len(log_lines) > 500:
                log_lines.pop(0)

        return_code = process.wait()

        if return_code != 0:
            return f"{prep_msg}\n\nPipeline 退出码 {return_code}，日志如下:\n\n" + "\n".join(log_lines[-200:])

        return f"{prep_msg}\n\n运行完成！\n\n" + "\n".join(log_lines[-200:])
    except Exception as e:
        return f"{prep_msg}\n\n启动 Pipeline 失败: {e}\n{traceback.format_exc()}"


def list_outputs() -> str:
    """列出输出目录中的文件"""
    if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
        return "暂无输出文件"

    lines = []
    for root, dirs, files in os.walk(SESSION.output_dir):
        level = root.replace(SESSION.output_dir, "").count(os.sep)
        indent = "  " * level
        lines.append(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for file in sorted(files):
            if not file.endswith(".log"):
                lines.append(f"{subindent}{file}")
    return "\n".join(lines) if lines else "暂无输出文件"


def preview_json(filename: str) -> str:
    """预览输出 JSON 文件内容"""
    if not SESSION.output_dir or not filename:
        return ""
    path = os.path.join(SESSION.output_dir, filename)
    if not os.path.exists(path):
        return f"文件不存在: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) if filename.endswith((".yaml", ".yml")) else f.read()
        if filename.endswith(".json"):
            import json
            return json.dumps(json.loads(data), ensure_ascii=False, indent=2)
        return str(data)[:5000]
    except Exception as e:
        return f"读取失败: {e}"


def download_file(filename: str):
    """返回文件路径供下载"""
    if not SESSION.output_dir or not filename:
        return None
    path = os.path.join(SESSION.output_dir, filename)
    if os.path.exists(path):
        return path
    return None


def open_directory(kind: str) -> str:
    """在系统文件管理器中打开目录（kind=output/materials）"""
    if kind == "output":
        path = SESSION.output_dir
    elif kind == "materials":
        path = os.path.join(SESSION.work_dir, "materials") if SESSION.work_dir else None
    else:
        return "未知目录类型"

    if not path or not os.path.exists(path):
        return f"目录不存在或尚未创建工作区: {path or kind}"

    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return f"已打开目录: {path}"
    except Exception as e:
        return f"打开目录失败: {e}"


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面"""
    with gr.Blocks(title="LLM-AutoCut 智能剪辑工作台") as demo:
        gr.Markdown("# LLM-AutoCut 智能剪辑工作台")
        gr.Markdown("基于多模态理解的 AI 辅助影视后期剪辑系统")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### API Key 设置")
                vlm_key_input = gr.Textbox(
                    label="VLM API Key（视觉模型，用于 Phase 1 分析视频）",
                    placeholder="粘贴豆包/火山方舟或 OpenAI 的 API Key",
                    type="password",
                    lines=1,
                )
                llm_key_input = gr.Textbox(
                    label="LLM API Key（文本模型，用于 Phase 2/3 决策）",
                    placeholder="粘贴 DeepSeek / OpenAI / 豆包文本模型的 API Key",
                    type="password",
                    lines=1,
                )
                gr.Markdown("""
                <small>
                提示：Key 仅保存在当前会话中，运行前自动注入 config.yaml，不会写入项目文件。<br>
                也可在「配置」标签页手动编辑 api_key，或设置系统环境变量 <code>ARK_API_KEY</code> / <code>DEEPSEEK_API_KEY</code>。
                </small>
                """)
                phase = gr.Radio(
                    choices=[
                        ("完整流程", "all"),
                        ("Phase 0: 粗剪", "0"),
                        ("Phase 1: 素材分析", "1"),
                        ("Phase 2: 剧本分析", "2"),
                        ("Phase 3: 镜头筛选", "3"),
                        ("Phase 4: 剪辑导出", "4"),
                    ],
                    value="all",
                    label="运行阶段",
                )
                run_btn = gr.Button("运行 Pipeline", variant="primary")

            with gr.Column(scale=2):
                log_output = gr.Textbox(label="运行日志", lines=24, max_lines=40, interactive=False)

        with gr.Tabs():
            with gr.TabItem("配置"):
                config_editor = gr.Code(
                    value=DEFAULT_CONFIG,
                    language="yaml",
                    label="config.yaml",
                    lines=30,
                )
            with gr.TabItem("剧本"):
                with gr.Row():
                    with gr.Column(scale=1):
                        script_upload = gr.File(
                            file_count="single",
                            file_types=[".md", ".txt", ".docx", ".pdf"],
                            label="导入剧本文档",
                        )
                        parse_script_btn = gr.Button("自动解析为台本", variant="secondary")
                        save_script_btn = gr.Button("保存剧本", variant="primary")
                        script_status = gr.Textbox(label="剧本处理状态", interactive=False)
                    with gr.Column(scale=2):
                        script_editor = gr.Code(
                            value=DEFAULT_SCRIPT,
                            language="markdown",
                            label="script.md（可在此编辑）",
                            lines=30,
                        )
            with gr.TabItem("素材"):
                with gr.Row():
                    with gr.Column():
                        materials_dir_input = gr.Textbox(
                            label="素材库文件夹路径（Phase 2 独立运行时使用）",
                            placeholder="粘贴素材库文件夹的绝对路径，如 D:\\素材库\\雨夜告白",
                            lines=1,
                        )
                with gr.Row():
                    with gr.Column():
                        raw_upload = gr.File(
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"],
                            label="RAW 素材（分析 + 切分）",
                        )
                    with gr.Column():
                        processed_upload = gr.File(
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"],
                            label="PROCESSED 素材（只分析，不切分）",
                        )
                with gr.Row():
                    with gr.Column():
                        analyzed_upload = gr.File(
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"],
                            label="ANALYZED 素材视频（只转译配置）",
                        )
                    with gr.Column():
                        analyzed_meta_upload = gr.File(
                            file_count="multiple",
                            file_types=[".json", ".csv"],
                            label="ANALYZED 素材配置文件（JSON / CSV）",
                        )
                gr.Markdown("""
                <small>
                提示：三种素材会自动放入对应子目录。<br>
                RAW：原始片场素材；PROCESSED：人工已处理好的镜头；ANALYZED：已有分析结果需转译。<br>
                ANALYZED 的配置文件命名需与视频同名或带 <code>_config</code> / <code>_meta</code> 后缀，如 <code>scene.mp4 + scene_config.json</code>。<br>
                <strong>Phase 2 独立运行</strong>时，可只填写「素材库文件夹路径」，系统会做 CV 轻量清点并匹配剧本。
                </small>
                """)
            with gr.TabItem("结果"):
                with gr.Row():
                    refresh_btn = gr.Button("刷新输出文件列表")
                    open_output_btn = gr.Button("打开输出目录")
                    open_materials_btn = gr.Button("打开素材目录")
                with gr.Row():
                    preview_select = gr.Dropdown(
                        choices=[],
                        label="选择文件预览",
                        interactive=True,
                    )
                output_list = gr.Textbox(label="输出文件", lines=10, interactive=False)
                preview_box = gr.Code(label="文件预览", language="json", lines=30, interactive=False)
                download_btn = gr.Button("下载选中文件")
                download_file_obj = gr.File(label="下载")

            with gr.TabItem("Phase 2 审核"):
                gr.Markdown("### Phase 2 剧本分析与分镜点审核\n\n在此查看由剧本拆解出的情节点（beat）和分镜点（sub-beat），可直接编辑后保存。")
                with gr.Row():
                    refresh_phase2_btn = gr.Button("刷新审核表", variant="secondary")
                    save_phase2_btn = gr.Button("保存修改", variant="primary")
                phase2_status = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        beats_df = gr.DataFrame(label="剧本节点表 (script_beats_analysis.json)", interactive=True)
                        beats_review_box = gr.Code(label="剧本节点审核 Markdown (phase2_beats_review.md)", language="markdown", lines=10, interactive=False)
                    with gr.Column(scale=1):
                        sub_beat_df = gr.DataFrame(label="分镜点表 (sub-beat)", interactive=True)
                beat_ref_box = gr.Textbox(label="可用 beat_id / sub_beat_id 参考", lines=4, interactive=False)

            with gr.TabItem("Phase 3 审核"):
                gr.Markdown("### Phase 3 镜头筛选与排序\n\n在此查看每个镜头的匹配得分、选择状态和去重结果，可手动修正镜头-剧本锚定。")
                with gr.Row():
                    refresh_phase3_btn = gr.Button("刷新 Phase 3 筛选表", variant="secondary")
                    export_anchor_btn = gr.Button("导出锚定校正表", variant="secondary")
                    apply_anchor_btn = gr.Button("应用锚定校正", variant="primary")
                phase3_status = gr.Textbox(label="状态", interactive=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        dedup_df = gr.DataFrame(label="去重/选择表 (phase2_deduplication.csv)", interactive=False)
                    with gr.Column(scale=1):
                        anchor_df = gr.DataFrame(label="锚定校正表（可编辑列：corrected_beat / corrected_confidence / corrected_function / corrected_reasoning / notes）", interactive=True)

            with gr.TabItem("Phase 4 导出"):
                gr.Markdown("### Phase 4 剪辑决策与导出\n\n在此查看剪辑决策时间线、最终成片和工程文件。")
                with gr.Row():
                    refresh_phase4_btn = gr.Button("刷新导出产物", variant="secondary")
                    open_output_btn2 = gr.Button("打开输出目录", variant="secondary")
                phase4_status = gr.Textbox(label="状态", interactive=False)
                phase4_summary = gr.Textbox(label="决策概览", lines=2, interactive=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        phase4_timeline_df = gr.DataFrame(label="剪辑决策时间线 (phase3_edit_decision.json)", interactive=False)
                    with gr.Column(scale=1):
                        phase4_final_timeline_df = gr.DataFrame(label="最终成片时间线 (timeline.json)", interactive=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        phase4_video_preview = gr.Video(label="成片预览 (final_with_dubbing.mp4)")
                    with gr.Column(scale=1):
                        phase4_audio_preview = gr.Audio(label="混音预览 (mixed_audio.wav)")
                phase4_files_df = gr.DataFrame(label="导出产物列表", interactive=False)
                phase4_download_file = gr.File(label="下载选中产物")

        # 事件绑定
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                phase,
                config_editor,
                script_editor,
                raw_upload,
                processed_upload,
                analyzed_upload,
                analyzed_meta_upload,
                materials_dir_input,
                vlm_key_input,
                llm_key_input,
            ],
            outputs=log_output,
        )

        script_upload.change(
            fn=on_script_upload,
            inputs=[script_upload],
            outputs=[script_editor, script_status],
        )

        parse_script_btn.click(
            fn=preprocess_script,
            inputs=[config_editor, script_editor, script_upload, vlm_key_input, llm_key_input],
            outputs=[script_editor, script_status],
        )

        def save_script(script_text: str):
            """把剧本编辑器中的内容保存到 workspace/script.md"""
            if not SESSION.work_dir:
                SESSION.work_dir = str(WORKSPACE_DIR)
            script_path = os.path.join(SESSION.work_dir, "script.md")
            try:
                ensure_dir(SESSION.work_dir)
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
                return f"剧本已保存: {script_path}"
            except Exception as e:
                return f"保存剧本失败: {e}\n{traceback.format_exc()}"

        save_script_btn.click(
            fn=save_script,
            inputs=[script_editor],
            outputs=[script_status],
        )

        def refresh_outputs():
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return "暂无输出文件", gr.update(choices=[])
            choices = []
            for root, _, files in os.walk(SESSION.output_dir):
                for file in sorted(files):
                    if not file.endswith(".log"):
                        choices.append(os.path.relpath(os.path.join(root, file), SESSION.output_dir))
            return list_outputs(), gr.update(choices=choices, value=choices[0] if choices else None)

        refresh_btn.click(
            fn=refresh_outputs,
            inputs=[],
            outputs=[output_list, preview_select],
        )

        preview_select.change(
            fn=preview_json,
            inputs=[preview_select],
            outputs=preview_box,
        )

        download_btn.click(
            fn=download_file,
            inputs=[preview_select],
            outputs=download_file_obj,
        )

        open_output_btn.click(
            fn=open_directory,
            inputs=[gr.Textbox(value="output", visible=False)],
            outputs=log_output,
        )
        open_materials_btn.click(
            fn=open_directory,
            inputs=[gr.Textbox(value="materials", visible=False)],
            outputs=log_output,
        )

        # ------------------------------------------------------------------
        # Phase 2 审核面板函数
        # ------------------------------------------------------------------
        def _load_csv_if_exists(csv_path: str) -> pd.DataFrame:
            if not csv_path or not os.path.exists(csv_path):
                return pd.DataFrame()
            try:
                return pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                try:
                    return pd.read_csv(csv_path, encoding="utf-8")
                except Exception:
                    return pd.DataFrame()

        def _load_beats_and_sub_beats():
            """加载剧本节点和分镜点，返回 (beats_df, sub_beat_df, ref_text, status)"""
            analysis_path = os.path.join(SESSION.output_dir, "script_beats_analysis.json")
            if not os.path.exists(analysis_path):
                return pd.DataFrame(), pd.DataFrame(), "", "尚未运行 Phase 2，无输出文件"
            try:
                with open(analysis_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                beat_rows = []
                sub_rows = []
                ref_lines = []
                for b in data.get("beats", []):
                    beat_rows.append({
                        "beat_id": b.get("beat_id", ""),
                        "act": b.get("act", ""),
                        "scene": b.get("scene", ""),
                        "content": b.get("content", ""),
                        "key_actions": ", ".join(b.get("key_actions", [])),
                        "key_dialogue": b.get("key_dialogue", ""),
                        "estimated_duration": b.get("estimated_duration", 0),
                        "pace": b.get("pace", ""),
                        "priority": b.get("priority", ""),
                        "emotion_intensity": b.get("emotion_intensity", ""),
                    })
                    ref_lines.append(f"{b.get('beat_id', '')}: {b.get('content', '')[:80]}")

                    for sb in b.get("sub_beats", []):
                        sub_rows.append({
                            "sub_beat_id": sb.get("sub_beat_id", ""),
                            "parent_beat_id": sb.get("parent_beat_id", ""),
                            "content": sb.get("content", ""),
                            "key_actions": ", ".join(sb.get("key_actions", [])),
                            "key_dialogue": sb.get("key_dialogue", ""),
                            "estimated_duration": sb.get("estimated_duration", 0),
                            "emotion": sb.get("emotion", ""),
                            "pace": sb.get("pace", ""),
                            "gender_state": sb.get("gender_state", ""),
                            "gender_transition": sb.get("gender_transition", ""),
                        })
                        ref_lines.append(f"  {sb.get('sub_beat_id', '')}: {sb.get('content', '')[:80]}")

                status = f"已加载 Phase 2 审核表。剧本节点: {len(beat_rows)} 个, 分镜点: {len(sub_rows)} 个。"
                return pd.DataFrame(beat_rows), pd.DataFrame(sub_rows), "\n".join(ref_lines), status
            except Exception as e:
                return pd.DataFrame(), pd.DataFrame(), "", f"加载剧本节点失败: {e}"

        def refresh_phase2_review():
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return pd.DataFrame(), pd.DataFrame(), "", "", "尚未运行 Phase 2，无输出文件"

            beats_df, sub_beat_df, beat_ref, status = _load_beats_and_sub_beats()

            md_path = os.path.join(SESSION.output_dir, "phase2_beats_review.md")
            md_text = ""
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()

            return beats_df, sub_beat_df, md_text, beat_ref, status

        def _split_comma(text: Any) -> List[str]:
            if pd.isna(text) or text is None:
                return []
            return [s.strip() for s in str(text).split(",") if s.strip()]

        def save_phase2_beats(df_beats, df_sub_beats):
            """把用户在网页端对 beat / sub-beat 的修改保存回 script_beats_analysis.json 和 workspace/script.md"""
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return "尚未运行 Phase 2，无法保存"
            if df_beats is None or df_beats.empty:
                return "剧本节点表为空，无法保存"

            analysis_path = os.path.join(SESSION.output_dir, "script_beats_analysis.json")
            if not os.path.exists(analysis_path):
                return f"找不到 {analysis_path}"

            try:
                with open(analysis_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 用 DataFrame 更新 beats
                beat_rows = df_beats.to_dict("records")
                sub_rows = df_sub_beats.to_dict("records") if df_sub_beats is not None and not df_sub_beats.empty else []

                sub_beat_map = {}
                for sr in sub_rows:
                    sb_id = sr.get("sub_beat_id", "")
                    parent = sr.get("parent_beat_id", "")
                    if not sb_id or not parent:
                        continue
                    sub_beat_map.setdefault(parent, []).append({
                        "sub_beat_id": sb_id,
                        "parent_beat_id": parent,
                        "act": sr.get("act", ""),
                        "scene": sr.get("scene", ""),
                        "content": sr.get("content", ""),
                        "key_actions": _split_comma(sr.get("key_actions", "")),
                        "key_dialogue": sr.get("key_dialogue", ""),
                        "estimated_duration": float(sr.get("estimated_duration", 0) or 0),
                        "emotion": sr.get("emotion", ""),
                        "pace": sr.get("pace", ""),
                        "gender_state": sr.get("gender_state", ""),
                        "gender_transition": sr.get("gender_transition", ""),
                    })

                updated_beats = []
                for br in beat_rows:
                    beat_id = br.get("beat_id", "")
                    beat = {
                        "act": br.get("act", ""),
                        "scene": br.get("scene", ""),
                        "beat_id": beat_id,
                        "location": br.get("location", ""),
                        "time": br.get("time", ""),
                        "content": br.get("content", ""),
                        "emotion": br.get("emotion", ""),
                        "key_actions": _split_comma(br.get("key_actions", "")),
                        "key_dialogue": br.get("key_dialogue", ""),
                        "gender_state": br.get("gender_state", ""),
                        "gender_transition": br.get("gender_transition", ""),
                        "estimated_duration": float(br.get("estimated_duration", 0) or 0),
                        "pace": br.get("pace", ""),
                        "emotion_intensity": float(br.get("emotion_intensity", 0) or 0),
                        "priority": int(br.get("priority", 3) or 3),
                        "required_shots_count": int(br.get("required_shots_count", 1) or 1),
                    }
                    beat["sub_beats"] = sub_beat_map.get(beat_id, [])
                    # 保留原有 dialogue_entries，避免被覆盖
                    for old_beat in data.get("beats", []):
                        if old_beat.get("beat_id") == beat_id and "dialogue_entries" in old_beat:
                            beat["dialogue_entries"] = old_beat["dialogue_entries"]
                            break
                    updated_beats.append(beat)

                data["beats"] = updated_beats
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                # 同步更新 workspace/script.md
                script_path = os.path.join(SESSION.work_dir, "script.md")
                if SESSION.work_dir and os.path.exists(SESSION.work_dir):
                    try:
                        _save_beats_to_script_md(updated_beats, script_path)
                    except Exception as e:
                        logger.warning(f"更新 script.md 失败: {e}")

                return (
                    f"已保存 {len(updated_beats)} 个剧本节点和 {len(sub_rows)} 个分镜点。\n"
                    f"script_beats_analysis.json: {analysis_path}\n"
                    f"script.md: {script_path if SESSION.work_dir else '未找到工作区'}"
                )
            except Exception as e:
                return f"保存失败: {e}\n{traceback.format_exc()}"

        def _save_beats_to_script_md(beats: List[Dict[str, Any]], script_path: str):
            """把 beat/sub-beat 数据写回标准 Markdown 大纲"""
            lines = ["# 剧本大纲", ""]

            # 按 act 分组
            acts: Dict[str, List[Dict[str, Any]]] = {}
            for b in beats:
                act = b.get("act") or "未分幕"
                acts.setdefault(act, []).append(b)

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
                    lines.append(f"- 建议时长：{b.get('estimated_duration', '')}")
                    lines.append(f"- 节奏：{b.get('pace', '')}")
                    lines.append(f"- 优先级：{b.get('priority', '')}")
                    lines.append(f"- 情绪强度：{b.get('emotion_intensity', '')}")

                    # sub-beat
                    for sb in b.get("sub_beats", []):
                        lines.append(f"#### {sb.get('sub_beat_id', '')}")
                        lines.append(f"  - 内容：{sb.get('content', '')}")
                        lines.append(f"  - 情绪：{sb.get('emotion', '')}")
                        sb_actions = sb.get("key_actions", [])
                        if isinstance(sb_actions, str):
                            sb_actions = [a.strip() for a in sb_actions.split(",") if a.strip()]
                        lines.append(f"  - 关键动作：{'，'.join(sb_actions)}")
                        lines.append(f"  - 关键台词：\"{sb.get('key_dialogue', '')}\"")
                        lines.append(f"  - 建议时长：{sb.get('estimated_duration', '')}")
                        if sb.get("gender_state"):
                            lines.append(f"  - 性别状态：{sb.get('gender_state')}")
                        if sb.get("gender_transition"):
                            lines.append(f"  - 状态切换：{sb.get('gender_transition')}")
                    lines.append("")

            with open(script_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines).strip() + "\n")

        refresh_phase2_btn.click(
            fn=refresh_phase2_review,
            inputs=[],
            outputs=[beats_df, sub_beat_df, beats_review_box, beat_ref_box, phase2_status],
        )
        save_phase2_btn.click(
            fn=save_phase2_beats,
            inputs=[beats_df, sub_beat_df],
            outputs=[phase2_status],
        )

        # ------------------------------------------------------------------
        # Phase 3 审核面板函数
        # ------------------------------------------------------------------
        def refresh_phase3_review():
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return pd.DataFrame(), pd.DataFrame(), "尚未运行 Phase 3，无输出文件"

            dedup_path = os.path.join(SESSION.output_dir, "phase2_deduplication.csv")
            dedup_df = _load_csv_if_exists(dedup_path)

            csv_path = os.path.join(SESSION.output_dir, "phase2_anchor_corrections.csv")
            if not os.path.exists(csv_path):
                try:
                    export_anchor_corrections(SESSION.output_dir, csv_path)
                except Exception as e:
                    return dedup_df, pd.DataFrame(), f"导出锚定校正表失败: {e}"
            anchor_df = _load_csv_if_exists(csv_path)

            status = (
                f"已加载 Phase 3 筛选表。"
                f"去重/选择行数: {len(dedup_df)}, 锚定校正行数: {len(anchor_df)}。"
                f"可在下方「锚定校正表」中修改 corrected_* 列，然后点击「应用锚定校正」。"
            )
            return dedup_df, anchor_df, status

        def export_anchor_corrections_ui():
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return pd.DataFrame(), "尚未运行 Phase 3，无法导出"
            csv_path = os.path.join(SESSION.output_dir, "phase2_anchor_corrections.csv")
            try:
                export_anchor_corrections(SESSION.output_dir, csv_path)
                anchor_df = _load_csv_if_exists(csv_path)
                return anchor_df, f"已导出锚定校正表: {csv_path}"
            except Exception as e:
                return pd.DataFrame(), f"导出失败: {e}"

        def apply_anchor_corrections_ui(df):
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return "尚未运行 Phase 3，无法应用"
            if df is None or df.empty:
                return "表格为空，未做任何修改"
            csv_path = os.path.join(SESSION.output_dir, "phase2_anchor_corrections.csv")
            try:
                # 把 DataFrame 写回 CSV，保留原列顺序
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                apply_anchor_corrections(SESSION.output_dir, csv_path)
                return f"锚定校正已应用并保存至 phase2_selected_shots.json。建议重新运行 Phase 4 查看效果。"
            except Exception as e:
                return f"应用失败: {e}\n{traceback.format_exc()}"

        refresh_phase3_btn.click(
            fn=refresh_phase3_review,
            inputs=[],
            outputs=[dedup_df, anchor_df, phase3_status],
        )
        export_anchor_btn.click(
            fn=export_anchor_corrections_ui,
            inputs=[],
            outputs=[anchor_df, phase3_status],
        )
        apply_anchor_btn.click(
            fn=apply_anchor_corrections_ui,
            inputs=[anchor_df],
            outputs=[phase3_status],
        )

        # ------------------------------------------------------------------
        # Phase 4 导出面板函数
        # ------------------------------------------------------------------
        def _format_file_size(size_bytes: int) -> str:
            if size_bytes < 1024:
                return f"{size_bytes} B"
            if size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            if size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

        def _human_time(timestamp: float) -> str:
            try:
                return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
            except Exception:
                return ""

        def _load_edit_decision_df():
            """加载 phase3_edit_decision.json 为剪辑决策时间线表格。"""
            path = os.path.join(SESSION.output_dir, "phase3_edit_decision.json")
            if not os.path.exists(path):
                return pd.DataFrame()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                decisions = data.get("timeline", [])
                if not decisions:
                    return pd.DataFrame()
                columns = [
                    "sequence", "shot_id", "source_file", "tc_in", "tc_out",
                    "speed", "technique", "transition", "audio", "purpose",
                    "beat_id", "sub_beat_id", "duration_sec", "notes",
                ]
                rows = []
                for d in decisions:
                    rows.append({col: d.get(col, "") for col in columns})
                return pd.DataFrame(rows)
            except Exception as e:
                logger.warning(f"加载 phase3_edit_decision.json 失败: {e}")
                return pd.DataFrame()

        def _load_final_timeline_df():
            """加载 timeline.json 为最终成片时间线表格。"""
            path = os.path.join(SESSION.output_dir, "timeline.json")
            if not os.path.exists(path):
                return pd.DataFrame()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                timeline = data.get("timeline", [])
                if not timeline:
                    return pd.DataFrame()
                columns = [
                    "sequence", "shot_id", "source_file", "source_in", "source_out",
                    "timeline_in", "timeline_out", "speed", "technique",
                    "transition", "audio", "purpose", "notes",
                ]
                rows = []
                for d in timeline:
                    rows.append({col: d.get(col, "") for col in columns})
                return pd.DataFrame(rows)
            except Exception as e:
                logger.warning(f"加载 timeline.json 失败: {e}")
                return pd.DataFrame()

        def _build_phase4_summary(data: dict) -> str:
            total = data.get("total_clips", len(data.get("timeline", [])))
            target = data.get("target_duration", "-")
            export_time = data.get("export_time", "-")
            if export_time and export_time != "-":
                try:
                    export_time = export_time[:19].replace("T", " ")
                except Exception:
                    pass
            total_duration = 0.0
            for d in data.get("timeline", []):
                try:
                    timeline_in = d.get("timeline_in", "00:00:00:00")
                    timeline_out = d.get("timeline_out", "00:00:00:00")
                    total_duration += _tc_to_sec(timeline_out) - _tc_to_sec(timeline_in)
                except Exception:
                    pass
            return (
                f"共 {total} 个镜头 | 目标时长 {target} | 成片总时长 {_sec_to_tc(total_duration)} | "
                f"导出时间 {export_time}"
            )

        def _tc_to_sec(tc: str) -> float:
            """把 HH:MM:SS:FF 或 HH:MM:SS.SSS 转成秒。"""
            try:
                parts = str(tc).replace(";", ":").split(":")
                if len(parts) == 4:
                    h, m, s, f = parts
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / 24.0
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                pass
            return 0.0

        def _sec_to_tc(sec: float) -> str:
            """秒转 HH:MM:SS。"""
            try:
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:
                return "00:00:00"

        def refresh_phase4_exports():
            if not SESSION.output_dir or not os.path.exists(SESSION.output_dir):
                return (
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    None,
                    None,
                    "",
                    "尚未运行 Phase 4，无输出文件",
                )

            # 剪辑决策时间线
            decision_df = _load_edit_decision_df()

            # 最终成片时间线
            final_df = _load_final_timeline_df()

            # 产物列表
            phase4_files = [
                ("成片视频", "final_with_dubbing.mp4"),
                ("混音音频", "mixed_audio.wav"),
                ("EDL 时间线", "timeline.edl"),
                ("FCPXML 时间线", "timeline.fcpxml"),
                ("CSV 时间线", "timeline_final.csv"),
                ("配音信息", "dub_info.json"),
            ]

            rows = []
            video_path = ""
            audio_path = ""
            first_existing = ""
            for label, filename in phase4_files:
                file_path = os.path.join(SESSION.output_dir, filename)
                exists = os.path.exists(file_path)
                size = os.path.getsize(file_path) if exists else 0
                mtime = os.path.getmtime(file_path) if exists else 0
                rows.append({
                    "产物": label,
                    "文件名": filename,
                    "状态": "已生成" if exists else "未生成",
                    "大小": _format_file_size(size) if exists else "-",
                    "生成时间": _human_time(mtime) if exists else "-",
                    "路径": file_path if exists else "",
                })
                if exists and not first_existing:
                    first_existing = file_path
                if filename == "final_with_dubbing.mp4" and exists:
                    video_path = file_path
                if filename == "mixed_audio.wav" and exists:
                    audio_path = file_path

            files_df = pd.DataFrame(rows)

            # 决策概览
            timeline_path = os.path.join(SESSION.output_dir, "timeline.json")
            summary = ""
            if os.path.exists(timeline_path):
                try:
                    with open(timeline_path, "r", encoding="utf-8") as f:
                        timeline_data = json.load(f)
                    summary = _build_phase4_summary(timeline_data)
                except Exception as e:
                    logger.warning(f"生成 Phase 4 概览失败: {e}")

            status = f"已扫描 Phase 4 导出产物，已生成 {sum(1 for r in rows if r['状态'] == '已生成')}/{len(rows)} 项。"

            return (
                decision_df,
                final_df,
                files_df,
                video_path if video_path else None,
                audio_path if audio_path else None,
                summary,
                status,
            )

        refresh_phase4_btn.click(
            fn=refresh_phase4_exports,
            inputs=[],
            outputs=[
                phase4_timeline_df,
                phase4_final_timeline_df,
                phase4_files_df,
                phase4_video_preview,
                phase4_audio_preview,
                phase4_summary,
                phase4_status,
            ],
        )
        open_output_btn2.click(
            fn=open_directory,
            inputs=[gr.Textbox(value="output", visible=False)],
            outputs=[phase4_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LLM-AutoCut Web UI")
    parser.add_argument("--config", default=None, help="预加载的配置文件路径")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公开分享链接")
    args = parser.parse_args()

    # 如果指定了配置文件，读取并替换默认配置
    initial_config = DEFAULT_CONFIG
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            initial_config = f.read()

    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share, max_threads=5)


if __name__ == "__main__":
    main()
