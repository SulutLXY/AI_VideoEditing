"""
LLM-AutoCut 数据模型

所有核心数据结构统一放在这里，避免散落在各模块中。
"""
import enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any


class MaterialState(enum.Enum):
    """素材处理状态"""
    RAW = "RAW"                 # 原始素材：分析 + 切分
    PROCESSED = "PROCESSED"     # 人工处理好的素材：只分析不切分
    ANALYZED = "ANALYZED"       # 已有分析结果：只做格式转换


@dataclass
class Provenance:
    """每个 Shot 的溯源信息"""
    state: str
    generated_by: str = "unknown"
    split_decision: Optional[Dict[str, Any]] = None
    conversion: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> Optional["Provenance"]:
        if not data:
            return None
        return Provenance(
            state=data.get("state", "unknown"),
            generated_by=data.get("generated_by", "unknown"),
            split_decision=data.get("split_decision"),
            conversion=data.get("conversion"),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Relationship:
    """镜头与相邻镜头的关系"""
    shot_id: Optional[str] = None
    relationship_type: str = "未知"  # 情绪延续 / 动作衔接 / 机位跳切 / 场景切换 / 对话衔接 / 时间跳跃
    coherence_score: float = 0.0      # 0.0 - 1.0，越高越连贯

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> Optional["Relationship"]:
        if not data:
            return None
        return Relationship(
            shot_id=data.get("shot_id"),
            relationship_type=data.get("relationship_type", "未知"),
            coherence_score=data.get("coherence_score", 0.0),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Relationships:
    """Shot 的前后关系"""
    prev: Optional[Relationship] = None
    next: Optional[Relationship] = None

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "Relationships":
        if not data:
            return Relationships()
        return Relationships(
            prev=Relationship.from_dict(data.get("prev")),
            next=Relationship.from_dict(data.get("next")),
        )

    def to_dict(self):
        return {
            "prev": self.prev.to_dict() if self.prev else None,
            "next": self.next.to_dict() if self.next else None,
        }


@dataclass
class ScriptBeat:
    """剧本情节点"""
    act: str
    scene: str
    beat_id: str
    location: str
    time: str
    content: str
    emotion: str
    key_actions: List[str] = field(default_factory=list)
    key_dialogue: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class CVMetadata:
    """CV 预扫描得到的物理属性"""
    resolution: Tuple[int, int] = (1920, 1080)
    aspect_ratio: str = "16:9"
    fps: float = 24.0
    duration: float = 0.0
    bitrate: Optional[str] = None
    codec: Optional[str] = None
    visual_quality: Optional[float] = None
    stability: Optional[float] = None
    brightness: Optional[float] = None
    scene_change_candidates: List[float] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Segment:
    """视频内部连续片段的语义描述（用于切分分析）"""
    start: float
    end: float
    description: str = ""
    camera_position: str = "未知"
    camera_movement: str = "未知"
    shot_size: str = "未知"
    framing: str = ""
    lighting: str = ""
    color_tone: str = ""
    characters: List[str] = field(default_factory=list)
    action: str = ""
    emotion: str = ""
    location: str = ""
    time_of_day: str = ""
    dialogue: str = ""
    key_objects: List[str] = field(default_factory=list)
    coherence_score: float = 0.0       # 段落内部连贯性 (0-1)，越高越不该切分
    is_long_take: bool = False          # 是否一镜到底/长镜头
    coherence_with_previous: float = 0.0

    # 新增：影视方向/表演/动作细节/连续性
    direction: str = ""               # 人物朝向 / 运动方向（如"从左向右 / 面向镜头"）
    performance: str = ""             # 表演评估（自然度、情绪强度、是否出戏）
    action_details: str = ""          # 动作细节描述
    continuity_score: float = 0.0     # 镜头内部连续性评分 (0-1)，越高越连贯
    continuity_notes: str = ""      # 连续性说明

    # 风格/氛围/文化
    style: str = ""
    atmosphere: str = ""
    culture: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Boundary:
    """候选切分点"""
    time_sec: float
    score: float = 0.0
    reason: str = ""
    confidence: str = "low"  # high / medium / low

    def to_dict(self):
        return asdict(self)


@dataclass
class Shot:
    """镜头数据结构（v0.2 完整版）"""
    # 基础标识
    shot_id: str
    state: str
    source_file: str
    source_path: str

    # 时间码
    tc_in: str          # HH:MM:SS:FF
    tc_out: str
    duration_sec: float

    # 物理属性
    fps: float = 24.0
    resolution: Tuple[int, int] = (1920, 1080)
    aspect_ratio: str = "16:9"
    bitrate: Optional[str] = None
    codec: Optional[str] = None

    # 视觉内容
    shot_size: str = "未知"
    camera_position: str = "未知"
    camera_movement: str = "未知"
    framing: str = ""
    lighting: str = ""
    color_tone: str = ""

    # 内容语义
    location: str = ""
    time_of_day: str = ""
    characters: List[str] = field(default_factory=list)
    action: str = ""
    emotion: str = ""
    dialogue: str = ""
    key_objects: List[str] = field(default_factory=list)

    # 风格/氛围/文化/标签
    style: str = ""
    atmosphere: str = ""
    culture: str = ""
    tags: List[str] = field(default_factory=list)

    # 新增：影视方向/表演/动作细节/连续性
    direction: str = ""               # 人物朝向 / 运动方向
    performance: str = ""             # 表演评估
    action_details: str = ""          # 动作细节
    continuity_score: float = 0.0     # 镜头内部连续性评分 (0-1)
    continuity_notes: str = ""      # 连续性说明

    # 质量评估
    visual_quality: Optional[float] = None
    stability: Optional[float] = None
    exposure: Optional[str] = None
    focus: Optional[str] = None
    noise: Optional[str] = None

    # 关键帧与 CV 信息
    keyframes: List[str] = field(default_factory=list)
    cv_metadata: Optional[Dict[str, Any]] = None

    # 剧本锚定
    script_anchor: Optional[Dict[str, Any]] = None

    # 前后关系
    relationships: Relationships = field(default_factory=Relationships)

    # 处理标记
    do_not_split: bool = False
    needs_review: bool = False
    is_long_take: bool = False
    coherence_score: float = 0.0
    internal_segments: Optional[List[Dict[str, Any]]] = None
    soft_transitions: List[Dict[str, Any]] = field(default_factory=list)

    # 溯源
    provenance: Optional[Provenance] = None

    # ASR
    asr_text: str = ""

    # Phase 2 去重/选择状态
    status: str = "候选"
    dedup_reason: str = ""
    quality_score: float = 0.0

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Shot":
        """从字典重建 Shot 对象，兼容旧版 vlm_description 字典"""
        relationships = Relationships.from_dict(data.get("relationships"))
        provenance = Provenance.from_dict(data.get("provenance"))

        # 兼容旧版：如果存在 vlm_description 字典，则反解到扁平字段
        vlm = data.get("vlm_description", {})

        def pick(field_name: str, default: Any = None) -> Any:
            """优先取顶层字段，其次取 vlm_description 中的字段"""
            if field_name in data and data[field_name] is not None:
                return data[field_name]
            return vlm.get(field_name, default)

        return Shot(
            shot_id=data.get("shot_id", "S000"),
            state=data.get("state", "RAW"),
            source_file=data.get("source_file", ""),
            source_path=data.get("source_path", ""),
            tc_in=data.get("tc_in", "00:00:00:00"),
            tc_out=data.get("tc_out", "00:00:00:00"),
            duration_sec=data.get("duration_sec", 0.0),
            fps=data.get("fps", 24.0),
            resolution=tuple(data.get("resolution", (1920, 1080))),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
            bitrate=data.get("bitrate"),
            codec=data.get("codec"),
            shot_size=pick("shot_size", ""),
            camera_position=pick("camera_position", ""),
            camera_movement=pick("camera_movement", ""),
            framing=pick("framing", ""),
            lighting=pick("lighting", ""),
            color_tone=pick("color_tone", ""),
            location=pick("location", ""),
            time_of_day=pick("time_of_day", ""),
            characters=pick("characters", []),
            action=pick("action", ""),
            emotion=pick("emotion", ""),
            dialogue=pick("dialogue", ""),
            key_objects=pick("key_objects", []),
            style=pick("style", ""),
            atmosphere=pick("atmosphere", ""),
            culture=pick("culture", ""),
            tags=pick("tags", []),
            direction=pick("direction", ""),
            performance=pick("performance", ""),
            action_details=pick("action_details", ""),
            continuity_score=pick("continuity_score", 0.0),
            continuity_notes=pick("continuity_notes", ""),
            visual_quality=pick("visual_quality"),
            stability=pick("stability"),
            exposure=pick("exposure"),
            focus=pick("focus"),
            noise=pick("noise"),
            keyframes=data.get("keyframes", []),
            cv_metadata=data.get("cv_metadata"),
            script_anchor=data.get("script_anchor"),
            relationships=relationships,
            do_not_split=data.get("do_not_split", False),
            needs_review=data.get("needs_review", False),
            is_long_take=data.get("is_long_take", False),
            coherence_score=data.get("coherence_score", 0.0),
            internal_segments=data.get("internal_segments"),
            provenance=provenance,
            asr_text=data.get("asr_text", ""),
            status=data.get("status", "候选"),
            dedup_reason=data.get("dedup_reason", ""),
            quality_score=data.get("quality_score", 0.0),
        )

    def to_dict(self):
        data = asdict(self)
        # 手动处理嵌套对象
        data["relationships"] = self.relationships.to_dict() if self.relationships else None
        data["provenance"] = self.provenance.to_dict() if self.provenance else None
        data["cv_metadata"] = self.cv_metadata
        return data

    @property
    def vlm_description(self) -> Dict[str, Any]:
        """兼容旧版代码：将扁平字段打包为 vlm_description 字典"""
        return {
            "location": self.location,
            "time_of_day": self.time_of_day,
            "characters": self.characters,
            "action": self.action,
            "emotion": self.emotion,
            "dialogue": self.dialogue,
            "shot_size": self.shot_size,
            "camera_position": self.camera_position,
            "camera_movement": self.camera_movement,
            "visual_quality": self.visual_quality,
            "stability": self.stability,
            "exposure": self.exposure,
            "focus": self.focus,
            "noise": self.noise,
            "framing": self.framing,
            "lighting": self.lighting,
            "color_tone": self.color_tone,
            "key_objects": self.key_objects,
            "style": self.style,
            "atmosphere": self.atmosphere,
            "culture": self.culture,
            "tags": self.tags,
            "direction": self.direction,
            "performance": self.performance,
            "action_details": self.action_details,
            "continuity_score": self.continuity_score,
            "continuity_notes": self.continuity_notes,
        }

    @vlm_description.setter
    def vlm_description(self, value: Dict[str, Any]):
        """兼容旧版代码：从字典反解到扁平字段"""
        if not value:
            return
        self.location = value.get("location", self.location)
        self.time_of_day = value.get("time_of_day", self.time_of_day)
        self.characters = value.get("characters", self.characters)
        self.action = value.get("action", self.action)
        self.emotion = value.get("emotion", self.emotion)
        self.dialogue = value.get("dialogue", self.dialogue)
        self.shot_size = value.get("shot_size", self.shot_size)
        self.camera_position = value.get("camera_position", self.camera_position)
        self.camera_movement = value.get("camera_movement", self.camera_movement)
        self.visual_quality = value.get("visual_quality", self.visual_quality)
        self.stability = value.get("stability", self.stability)
        self.exposure = value.get("exposure", self.exposure)
        self.focus = value.get("focus", self.focus)
        self.noise = value.get("noise", self.noise)
        self.framing = value.get("framing", self.framing)
        self.lighting = value.get("lighting", self.lighting)
        self.color_tone = value.get("color_tone", self.color_tone)
        self.key_objects = value.get("key_objects", self.key_objects)
        self.style = value.get("style", self.style)
        self.atmosphere = value.get("atmosphere", self.atmosphere)
        self.culture = value.get("culture", self.culture)
        self.tags = value.get("tags", self.tags)
        self.direction = value.get("direction", self.direction)
        self.performance = value.get("performance", self.performance)
        self.action_details = value.get("action_details", self.action_details)
        self.continuity_score = value.get("continuity_score", self.continuity_score)
        self.continuity_notes = value.get("continuity_notes", self.continuity_notes)
