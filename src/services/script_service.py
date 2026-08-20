"""
剧本预处理服务

职责：
- 读取多种格式的剧本文档（.md / .txt / .docx / .pdf）
- 判断文本是否已经是项目所需的结构化大纲
- 如果不是，调用 LLM 把原始剧本解析为标准化 script.md 格式
"""
import json
import os
import re
from typing import List, Dict, Any, Optional

from src.services.llm_service import LLMService
from src.utils import logger


class ScriptReadError(Exception):
    """剧本文件读取失败"""
    pass


class ScriptParseError(Exception):
    """剧本内容解析失败"""
    pass


def read_script_file(file_path: str) -> str:
    """读取剧本文档，支持 .md / .txt / .docx / .pdf"""
    if not os.path.exists(file_path):
        raise ScriptReadError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".md", ".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".docx":
        try:
            from docx import Document
        except ImportError as e:
            raise ScriptReadError(
                "读取 .docx 需要安装 python-docx，请运行: pip install python-docx"
            ) from e
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ScriptReadError(
                "读取 .pdf 需要安装 pypdf，请运行: pip install pypdf"
            ) from e
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ScriptReadError(f"不支持的剧本格式: {ext}")


def is_structured_outline(text: str) -> bool:
    """判断文本是否已经是项目标准的结构化大纲"""
    # 必须同时包含幕标题和场-情节点标题
    has_act = re.search(r"^##\s+", text, re.MULTILINE) is not None
    has_beat = re.search(r"^###\s+场\d+", text, re.MULTILINE) is not None
    has_fields = "- 地点：" in text or "- 地点:" in text
    return has_act and has_beat and has_fields


def clean_script_text(text: str) -> str:
    """简单清洗：统一换行、去掉多余空行、页码等"""
    # 统一换行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去掉页码行（纯数字或 "第X页"）
    text = re.sub(r"^\s*第\s*\d+\s*页\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # 合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SCENE_HEADING_PATTERNS = [
    r"^第\s*[一二三四五六七八九十百零\d]+\s*场[：:.\s]",       # 第三场 / 第3场
    r"^第\s*[一二三四五六七八九十百零\d]+\s*章[：:.\s]",       # 第一章 / 第3章
    r"^第\s*[一二三四五六七八九十百零\d]+\s*节[：:.\s]",       # 第一节 / 第3节
    r"^第\s*[一二三四五六七八九十百零\d]+\s*回[：:.\s]",       # 第一回 / 第3回
    r"^第\s*[一二三四五六七八九十百零\d]+\s*集[：:.\s]",       # 第一集 / 第3集
    r"^第\s*[一二三四五六七八九十百零\d]+\s*幕[：:.\s]",       # 第一幕 / 第3幕
    r"^场\s*\d+\s*[：:.\s]",                                   # 场3
    r"^(?:内景|外景|日内|日外|夜内|夜外|晨内|晨外)[\.\\/]",   # 内景/外景
    r"^(?:INT|EXT|INT\./EXT|EXT\./INT)[\.\\/\s]",           # 英文场景头
    r"^Chapter\s+\d+",                                          # Chapter 1
    r"^Scene\s+\d+",                                             # Scene 1
]


def split_into_scenes(text: str) -> List[str]:
    """把剧本按场景标题切分成多个场景块"""
    combined = "|".join(SCENE_HEADING_PATTERNS)
    regex = re.compile(f"({combined})", re.MULTILINE)

    parts = regex.split(text)
    if len(parts) <= 1:
        # 没识别到场景标题，按固定长度切分（避免单 prompt 过长）
        return _chunk_by_length(text)

    scenes = []
    current_heading = ""
    for part in parts:
        if not part.strip():
            continue
        # 用未 strip 的 part 判断是否是场景标题（标题包含尾部空白/分隔符）
        if regex.match(part):
            current_heading = part.strip()
            continue
        scenes.append(f"{current_heading}\n{part.strip()}".strip())
        current_heading = ""

    return [s for s in scenes if s]


def _chunk_by_length(text: str, max_chars: int = 4000) -> List[str]:
    """按段落切分文本，控制每块长度"""
    paragraphs = text.split("\n")
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len > max_chars and current:
            chunks.append("\n".join(current))
            current = [p]
            current_len = p_len
        else:
            current.append(p)
            current_len += p_len
    if current:
        chunks.append("\n".join(current))
    return chunks


class ScriptPreprocessor:
    """把原始剧本解析为项目标准结构化大纲"""

    def __init__(self, llm_service: LLMService, config: Optional[Dict[str, Any]] = None):
        self.llm_service = llm_service
        self.config = config or {}
        self.output_shot_requirements = self.config.get("output_shot_requirements", False)
        self.max_chunk_chars = self.config.get("max_chunk_chars", 4000)

    def preprocess(self, text: str) -> str:
        """入口：返回标准 script.md 文本

        如果无法解析出任何情节点，抛出 ScriptParseError，
        由调用方决定是否保留原始文本并提示用户。
        """
        text = clean_script_text(text)

        if is_structured_outline(text):
            logger.info("检测到已格式化的剧本大纲，直接透传")
            return text

        logger.info("检测到非结构化剧本，开始 LLM 解析...")
        scenes = split_into_scenes(text)
        logger.info(f"剧本已切分为 {len(scenes)} 个场景/段落")

        all_beats = []
        for idx, scene_text in enumerate(scenes, 1):
            logger.info(f"解析场景 {idx}/{len(scenes)} ...")
            try:
                parsed = self._parse_scene(scene_text)
            except Exception as e:
                logger.warning(f"场景 {idx} 解析失败: {e}")
                parsed = []
            all_beats.extend(parsed)

        if not all_beats:
            raise ScriptParseError(
                "未能从剧本中解析出任何情节点。可能原因："
                "1) 文件内容为空或非剧本/叙事格式；"
                "2) LLM 返回格式异常；"
                "3) 当前模型不适合解析该文本。"
            )

        return self._build_markdown(all_beats)

    def _parse_scene(self, scene_text: str) -> List[Dict[str, Any]]:
        """调用 LLM 解析单个场景，返回 beat 列表"""
        prompt = self._build_prompt(scene_text)
        content = self.llm_service.generate(prompt)
        data = self._extract_json(content)
        # 兼容两种输出：{"act":..., "scene":..., "beats":[...]} 或 [{...}]
        if isinstance(data, dict):
            beats = data.get("beats", [data])
        elif isinstance(data, list):
            beats = data
        else:
            beats = []
        # 补全必要字段
        for b in beats:
            b.setdefault("act", "")
            b.setdefault("scene", "")
            b.setdefault("beat_id", "")
            b.setdefault("location", "")
            b.setdefault("time", "")
            b.setdefault("content", "")
            b.setdefault("emotion", "")
            b.setdefault("key_actions", [])
            b.setdefault("key_dialogue", "")
            b.setdefault("suggested_shots", "")
        return beats

    def _build_prompt(self, scene_text: str) -> str:
        shot_req = ""
        if self.output_shot_requirements:
            shot_req = (
                "每个情节点额外输出一个字段 suggested_shots（字符串），"
                "简要说明该情节点可能需要哪些镜头（如：主镜头/反应镜头/插入镜头/空镜/B-roll）。"
            )

        template = self._load_prompt_template()
        # 使用 replace 而非 format，避免模板中的 JSON 花括号被误解析为占位符
        return (
            template
            .replace("{scene_text}", scene_text)
            .replace("{shot_requirements}", shot_req)
        )

    def _load_prompt_template(self) -> str:
        template_path = os.path.join("prompts", "phase0_script_preprocess.txt")
        if os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        return self._default_prompt()

    def _default_prompt(self) -> str:
        return """你是一位资深的影视编剧助理，擅长把原始剧本、分场稿、小说片段或故事梗概转换成结构化的剪辑台本。

## 任务
请阅读下面提供的内容：
- 如果是分场景剧本/分场稿，按场景拆分为情节点；
- 如果是小说/叙事文本/故事梗概，请按叙事节奏和情节转折拆分为情节点；
- 每个情节点是叙事上相对完整的一个小单元。

对拆分出的每个情节点，输出以下字段：
- act: 所属幕（如"第一幕"，如果无法判断可填空字符串）
- scene: 场号（如"场1"，如果无法判断可填空字符串）
- beat_id: 情节点ID（如"场1-情节点A"、"场1-情节点B"）
- location: 地点
- time: 时间（白天/夜晚/傍晚/室内灯光等）
- content: 该情节点的主要内容（一句话概括）
- emotion: 情绪
- key_actions: 关键动作数组
- key_dialogue: 关键台词（如有，保留原文；没有则空字符串）
{shot_requirements}

## 这场戏的内容
{scene_text}

## 输出格式
只输出 JSON，不要其他文字。JSON 结构如下：
{{
  "act": "第一幕",
  "scene": "场1",
  "beats": [
    {{
      "beat_id": "场1-情节点A",
      "location": "咖啡馆",
      "time": "傍晚",
      "content": "男主独自等待，表现焦虑",
      "emotion": "焦虑",
      "key_actions": ["看表", "深呼吸"],
      "key_dialogue": "",
      "suggested_shots": ""
    }}
  ]
}}
"""

    @staticmethod
    def _extract_json(content: str) -> Any:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())

    def _build_markdown(self, beats: List[Dict[str, Any]]) -> str:
        """把解析结果合并为项目标准的 Markdown 大纲"""
        if not beats:
            return "# 剧本大纲\n\n（未能解析出任何情节点）\n"

        # 按 act 分组
        acts: Dict[str, List[Dict[str, Any]]] = {}
        for b in beats:
            act = b.get("act") or "未分幕"
            acts.setdefault(act, []).append(b)

        lines = ["# 剧本大纲", ""]
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
                if self.output_shot_requirements and b.get("suggested_shots"):
                    lines.append(f"- 建议镜头：{b.get('suggested_shots')}")
                lines.append("")

        return "\n".join(lines).strip() + "\n"
