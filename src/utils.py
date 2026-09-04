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


class _Utf8StreamHandler(logging.StreamHandler):
    """强制使用 UTF-8 输出的 StreamHandler，解决 Windows 控制台日志乱码"""

    def __init__(self):
        import sys
        import io
        # 将 sys.stdout 包装为 UTF-8 编码，避免中文在控制台输出乱码
        try:
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        except Exception:
            stream = sys.stdout
        super().__init__(stream)


def init_logging(output_dir: Optional[str] = None):
    """初始化日志，确保输出目录存在后调用"""
    handlers = [_Utf8StreamHandler()]
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


def fallback_vlm_to_local(config: Dict[str, Any]) -> bool:
    """
    当配置使用需要 API Key 的在线 VLM 且未提供 Key 时，自动回退到本地 VLM。
    直接修改传入的 config 字典，返回是否发生了切换。
    """
    vlm_cfg = config.get("models", {}).get("vlm", {})
    provider = vlm_cfg.get("provider", "")
    api_key = (vlm_cfg.get("api_key") or "").strip()

    # 已经是本地模型则不处理
    if provider == "local":
        return False

    # 已填写 API Key 也不切换
    if api_key:
        return False

    # 环境变量中存在通用 VLM Key 也不切换
    env_keys = ["OPENAI_API_KEY", "ARK_API_KEY", "VLM_API_KEY"]
    if any(os.getenv(k) for k in env_keys):
        return False

    local_cfg = config.setdefault("models", {}).setdefault("local", {})
    local_cfg.setdefault("enabled", True)
    local_cfg.setdefault("device", "cuda")
    local_cfg.setdefault("cache_dir", os.path.join("models", "local"))
    vision_cfg = local_cfg.setdefault("vision", {})
    vision_cfg.setdefault("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
    vision_cfg.setdefault("model_path", "")
    vision_cfg.setdefault("load_in_4bit", True)
    vision_cfg.setdefault("max_new_tokens", 512)
    vision_cfg.setdefault("keyframe_count", 3)

    vlm_cfg["provider"] = "local"
    vlm_cfg["model"] = vision_cfg["model_id"]
    logger.warning(
        "未检测到 VLM API Key，已自动切换为本地 VLM (Qwen2.5-VL)。"
        "如需使用在线模型，请在 config.yaml 中填写 models.vlm.api_key。"
    )
    return True


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


def _split_gender_transition_beats(beats: List[ScriptBeat]) -> List[ScriptBeat]:
    """
    把旧式 7-beat 结构自动拆分为 9-beat，确保"变装"状态转换有独立 beat。
    仅对已知结构（场1-情节点A~G）且包含变身/换装关键词的剧本生效。
    """
    beat_ids = [b.beat_id for b in beats]
    # 只处理标准 7-beat 结构
    expected = ["场1-情节点A", "场1-情节点B", "场1-情节点C", "场1-情节点D",
                "场1-情节点E", "场1-情节点F", "场1-情节点G"]
    if beat_ids != expected:
        return beats

    def find_beat(beat_id: str) -> Optional[ScriptBeat]:
        for b in beats:
            if b.beat_id == beat_id:
                return b
        return None

    a = find_beat("场1-情节点A")
    c = find_beat("场1-情节点C")
    if not a or not c:
        return beats

    # 判断 A 是否包含变身女装：content 或 key_actions 里有相关关键词
    a_content = a.content or ""
    a_actions = "、".join(a.key_actions or [])
    a_has_female_transition = (
        "男装褪去" in a_content or "露出女装" in a_content
        or "变身" in a_actions or "女装" in a_actions
    )

    # 判断 C/D 区域是否包含换回男装
    c_content = c.content or ""
    d = find_beat("场1-情节点D")
    d_content = d.content or "" if d else ""
    cd_content = c_content + d_content
    c_actions = "、".join(c.key_actions or [])
    cd_has_male_transition = (
        "穿上男装" in cd_content or "换装" in cd_content
        or "快速换装" in c_actions or "换装" in c_actions
    )

    if not (a_has_female_transition and cd_has_male_transition):
        return beats

    new_beats: List[ScriptBeat] = []

    # A -> A1 追猫 + A2 变身接猫
    new_beats.append(ScriptBeat(
        act=a.act, scene=a.scene, beat_id="场1-情节点A1",
        location=a.location, time=a.time,
        content="云琛身穿男装，手拿烧饼，第一视角正快速奔跑追逐一只黑猫。云琛边跑边喊，黑猫快速窜上屋檐。",
        emotion="紧张/兴奋",
        key_actions=["奔跑追猫", "黑猫上檐"],
        key_dialogue="\"天天又叫又跑的？半个月你都离家出走十二回了，害得妙妙天天哭。咋的，浪子发情啊你？\"",
        gender_state="男装",
        gender_transition="",
    ))
    new_beats.append(ScriptBeat(
        act=a.act, scene=a.scene, beat_id="场1-情节点A2",
        location=a.location, time=a.time,
        content="云琛观察四周无人，极速起身跃到半空中，男装褪去露出女装，身后月光洒落，先一步落在屋檐上把黑猫接在怀中。",
        emotion="惊艳/紧张",
        key_actions=["跃起", "男装→女装", "接猫入怀"],
        key_dialogue="",
        gender_state="女装",
        gender_transition="男装→女装",
    ))

    # B 保持不变
    b = find_beat("场1-情节点B")
    if b:
        b.gender_state = b.gender_state or "女装"
        new_beats.append(b)

    # C -> C1 女装扶胸独白 + C2 听到招聘后扔饼换回男装
    # 关键台词保留在原 C 拆分后的 C1：女装状态下的独白；C2 承接换装动作。
    new_beats.append(ScriptBeat(
        act=c.act, scene=c.scene, beat_id="场1-情节点C1",
        location=c.location, time=c.time,
        content="云琛扶了扶胸部，表情稍有不舒服。云琛女装状态，自言自语吐槽束胸的不适。",
        emotion="尴尬/无奈",
        key_actions=["扶胸", "不舒服", "女装独白"],
        key_dialogue="\"五年了，还真是不舒服，都勒得我快平了。\"",
        gender_state="女装",
        gender_transition="",
    ))
    new_beats.append(ScriptBeat(
        act=c.act, scene=c.scene, beat_id="场1-情节点C2",
        location=c.location, time=c.time,
        content="远处传来小六喊招聘护卫的声音。云琛脸色微变，低声说\"是他\"，迅速朝小六扔烧饼过去，趁烧饼遮挡视线的一瞬迅速穿上男装。",
        emotion="机警/紧张",
        key_actions=["听到声音", "脸色微变", "扔烧饼", "女装→男装"],
        key_dialogue="\"是他。\"",
        gender_state="男装",
        gender_transition="女装→男装",
    ))

    # D/E/F/G 保持不变，D 改为小六视角的过渡 beat
    for bid in ["场1-情节点D", "场1-情节点E", "场1-情节点F", "场1-情节点G"]:
        b = find_beat(bid)
        if not b:
            continue
        if bid == "场1-情节点D":
            b.content = "小六喊声越来越近，云琛已换好男装。小六接住烧饼仰头看屋檐，黑猫从空中落入云琛怀中。"
            b.key_dialogue = ""
            b.key_actions = ["接烧饼", "黑猫落入怀中"]
            b.gender_state = b.gender_state or "男装"
        elif bid == "场1-情节点E":
            b.gender_state = b.gender_state or "男装"
        elif bid == "场1-情节点F":
            b.gender_state = b.gender_state or "男装"
        elif bid == "场1-情节点G":
            b.gender_state = b.gender_state or ""
        new_beats.append(b)

    logger.info(f"自动拆分性别转换 beat: {len(beats)} -> {len(new_beats)} 个情节点")
    return new_beats


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
    - 性别状态：xxx（可选）
    - 状态切换：xxx（可选）
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
                        # 支持同一属性跨多行重复定义（如多个关键台词），用换行拼接
                        if key in props:
                            props[key] = props[key] + "\n" + value
                        else:
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
                key_actions=[a.strip() for a in re.split(r"[，、,]", props.get("关键动作", "")) if a.strip()],
                key_dialogue=props.get("关键台词", props.get("dialogue", "")),
                gender_state=props.get("性别状态", props.get("gender_state", "")),
                gender_transition=props.get("状态切换", props.get("gender_transition", "")),
            )
            beats.append(beat)
            i = j
            continue

        i += 1

    # 自动拆分旧式 7-beat 结构中隐含的状态转换
    beats = _split_gender_transition_beats(beats)

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


def save_beats_review_md(
    beats: List[ScriptBeat],
    output_path: str,
    beat_analysis: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """生成用户可二次编辑的 beat 级审核表 Markdown。

    输出包含每个 beat 的剧情摘要、关键动作/台词、时长/节奏/情绪、对白条目、匹配镜头数。
    用户可直接修改此文件后，通过 --input-json 指定 custom beats 重新跑 Phase 2（未来可扩展导入）。
    """
    ensure_dir(os.path.dirname(output_path))
    beat_analysis = beat_analysis or {}

    lines = [
        "# Phase 2 剧本节点审核表",
        "",
        "> 本表由 Phase 2 剧本节奏分析自动生成，用于人工二次确认/修改剧情节点。",
        "> 修改方式：直接编辑项目根目录下的 `workspace/script.md`，然后重新运行 Phase 2。",
        "> 或者编辑本表后，通过后续导入功能（开发中）覆盖自动分析结果。",
        "",
        "## 概览",
        "",
        f"- 情节点总数: {len(beats)}",
        f"- 含对白情节点: {sum(1 for b in beats if b.dialogue_entries)}",
        f"- 总对白 1x 时长: {round(sum(sum(e.estimated_duration for e in b.dialogue_entries) for b in beats), 2)} 秒",
        "",
        "---",
        "",
    ]

    for idx, beat in enumerate(beats, 1):
        analysis = beat_analysis.get(beat.beat_id, {})
        dialogue_text = "\n".join(
            f"  - **{e.speaker}**（{e.gender_state or '默认'}）: {e.text} "
            f"[语速 {e.pace}, 1x 时长 {e.estimated_duration:.2f}s]"
            for e in beat.dialogue_entries
        ) or "  - 无"

        lines.extend([
            f"## {idx}. {beat.beat_id}",
            "",
            f"- **幕/场**: {beat.act} / {beat.scene}",
            f"- **地点**: {beat.location}",
            f"- **时间**: {beat.time}",
            f"- **情绪**: {beat.emotion}",
            f"- **状态/切换**: {beat.gender_state or '无'} / {beat.gender_transition or '无切换'}",
            f"- **建议时长**: {beat.estimated_duration:.2f}s",
            f"- **节奏**: {beat.pace}",
            f"- **情绪强度**: {beat.emotion_intensity:.1f}",
            f"- **优先级**: {beat.priority}",
            f"- **最少镜头数**: {beat.required_shots_count}",
            f"- **关键动作**: {', '.join(beat.key_actions) or '无'}",
            f"- **关键台词**: {beat.key_dialogue or '无'}",
            f"- **内容摘要**: {beat.content}",
            f"- **分析理由**: {analysis.get('reasoning', '无')}",
            "",
            "### 对白条目",
            "",
            dialogue_text,
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 修改记录",
        "",
        "- 初始生成时间: 见文件生成时间戳",
        "- 修改人: __________",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"已生成 Phase 2 剧本节点审核表: {output_path}")
    return output_path
