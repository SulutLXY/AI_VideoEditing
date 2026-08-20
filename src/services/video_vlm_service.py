"""
Video-Native VLM 服务层

职责：
- 对整段视频做连续语义理解，而非仅看孤立抽帧。
- 输出段落级描述 + 段落内部连贯性评分 + 建议切分点。
- 当前实现以现有抽帧 VLM 作为 fallback，接口设计兼容未来接入真正的视频模型。

输出结构：
- VideoSegment: 连续段落
- VideoAnalysisResult: 整段视频分析结果
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

from src.models import Segment
from src.services.vlm_service import VLMService
from src.utils import logger


@dataclass
class VideoSegment:
    """视频原生理解输出的连续段落"""
    start: float
    end: float
    description: str = ""
    coherence_score: float = 0.0       # 0-1，越高越连贯，越不该切分
    is_long_take: bool = False

    # 视觉内容
    shot_size: str = "未知"
    camera_position: str = "未知"
    camera_movement: str = "未知"
    framing: str = ""
    lighting: str = ""
    color_tone: str = ""
    key_objects: List[str] = field(default_factory=list)

    # 内容语义
    location: str = ""
    time_of_day: str = ""
    characters: List[str] = field(default_factory=list)
    action: str = ""
    emotion: str = ""
    dialogue: str = ""

    # 风格/氛围/文化
    style: str = ""
    atmosphere: str = ""
    culture: str = ""
    tags: List[str] = field(default_factory=list)

    def to_segment(self) -> Segment:
        """转换为标准 Segment 对象"""
        return Segment(
            start=self.start,
            end=self.end,
            description=self.description,
            camera_position=self.camera_position,
            camera_movement=self.camera_movement,
            shot_size=self.shot_size,
            framing=self.framing,
            lighting=self.lighting,
            color_tone=self.color_tone,
            characters=self.characters,
            action=self.action,
            emotion=self.emotion,
            location=self.location,
            time_of_day=self.time_of_day,
            dialogue=self.dialogue,
            coherence_score=self.coherence_score,
            is_long_take=self.is_long_take,
            key_objects=self.key_objects,
        )


@dataclass
class VideoAnalysisResult:
    """整段视频分析结果"""
    segments: List[VideoSegment] = field(default_factory=list)
    suggested_cut_points: List[float] = field(default_factory=list)
    overall_summary: str = ""
    overall_coherence: float = 0.0


class VideoVLMService:
    """视频原生理解服务

    当前默认以抽帧 VLM 作为 fallback，因为主流视频 VLM API  still 以图像序列
    或有限视频片段为输入。未来可替换为直接上传视频文件的 provider。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("models", {}).get("video_vlm", {})
        self.provider = self.config.get("provider", "vlm_fallback")
        self.model_name = self.config.get("model", "gpt-4o-mini")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.3)
        self.frame_sample_rate = self.config.get("frame_sample_rate", 1)
        self.max_frames = self.config.get("max_frames", 60)

        # 当前 fallback：使用现有抽帧 VLM 服务
        self._vlm_fallback = VLMService(config)

    def analyze_video(
        self,
        video_path: str,
        duration: float,
        temp_dir: str,
    ) -> VideoAnalysisResult:
        """对视频做原生理解，返回段落和切分建议"""
        logger.info(f"[VideoVLM] 分析视频: {os.path.basename(video_path)} (provider={self.provider})")

        frames = self._vlm_fallback.sample_frames(video_path, duration, temp_dir)
        if not frames:
            logger.warning("未抽取到帧，返回整体素材")
            return VideoAnalysisResult(
                segments=[VideoSegment(start=0.0, end=duration, description="整体素材")]
            )

        if self.provider == "vlm_fallback":
            return self._analyze_with_frame_sequence(video_path, frames, duration)

        raise NotImplementedError(f"Video VLM provider {self.provider} 尚未实现")

    def _analyze_with_frame_sequence(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> VideoAnalysisResult:
        """用抽帧序列模拟视频原生理解"""
        prompt = self._build_video_native_prompt(frames, duration)
        messages = self._vlm_fallback.build_messages(prompt, frames)

        try:
            content = self._vlm_fallback._call(messages)
            data = self._extract_json(content)
            return self._parse_result(data)
        except Exception as e:
            logger.error(f"Video VLM 分析失败: {e}")
            return VideoAnalysisResult(
                segments=[VideoSegment(start=0.0, end=duration, description="视频分析失败，整体作为一个段落")]
            )

    def _build_video_native_prompt(self, frames: List[Tuple[float, str]], duration: float) -> str:
        frame_times = ", ".join([f"{t:.1f}s" for t, _ in frames])
        return f"""你是一位专业的电影镜头分析师，擅长从视频连续画面中理解镜头内容、时序变化和影视标签。

下面是一组按时间顺序排列的视频帧（时间点分别为：{frame_times}），总时长约为 {duration:.1f} 秒。

请从视频原生理解的角度，分析这段视频中是否存在自然的段落边界。特别注意：
1. 一镜到底、连续长镜头、动作/情绪连贯的段落应视为一个整体，不要拆分。
2. 只有当你明确观察到机位切换、主体变化、场景转换、情绪转折或动作中断时，才建议切分。
3. 对每个段落给出内部连贯性评分（coherence_score，0-1），分数越高表示越不应该切分。
4. 为每个段落输出丰富的影视标签，便于后续素材检索与匹配。

## 输出格式
请输出纯 JSON，包含以下字段：

{{
  "overall_summary": "视频整体内容概述",
  "overall_coherence": 0.85,
  "suggested_cut_points": [12.5, 34.0],
  "segments": [
    {{
      "start": 0.0,
      "end": 12.5,
      "description": "段落内容描述，50字以内",
      "coherence_score": 0.9,
      "is_long_take": false,
      "shot_size": "特写/近景/中景/全景/大全景",
      "camera_position": "机位描述，如柜台正面、门口右侧、高空航拍",
      "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦",
      "framing": "构图方式，如居中、三分法、对称、框架式",
      "lighting": "光线特征，如自然光、柔光、逆光、霓虹灯光",
      "color_tone": "色调，如暖黄、冷蓝、高饱和、黑白、青橙",
      "location": "场景地点",
      "time_of_day": "白天/傍晚/夜晚/室内灯光",
      "characters": ["角色名或主体"],
      "key_objects": ["画面关键物体，如电视、汽车、咖啡杯"],
      "action": "主要动作/事件",
      "emotion": "整体情绪",
      "dialogue": "关键台词（如有）",
      "style": "风格，如纪实、广告、电影感、vlog、剧情",
      "atmosphere": "氛围，如紧张、温馨、宏大、压抑、梦幻",
      "culture": "文化背景或视觉符号",
      "tags": ["标签1", "标签2"]
    }},
    ...
  ]
}}

## 分析要求
1. segments 必须连续且不重叠，覆盖整个视频 [0.0, {duration:.1f}]。
2. 若整段视频是强连贯的一镜到底，请只输出一个 segment，并设置 coherence_score=0.95, is_long_take=true。
3. 只有 coherence_score < 0.4 的边界，才放入 suggested_cut_points。
4. 对机位和 camera_position 的描述要准确、稳定，便于后续判断镜头是否切换。
5. 时间戳必须基于视频帧时间点，不得超过 {duration:.1f} 秒。
6. 只输出 JSON，不要包含其他文字。"""

    def _extract_json(self, content: str) -> Dict:
        """从模型返回中提取 JSON"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())

    def _parse_result(self, data: Dict) -> VideoAnalysisResult:
        """解析 JSON 为 VideoAnalysisResult"""
        segments = []
        for seg in data.get("segments", []):
            segments.append(VideoSegment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                description=seg.get("description", ""),
                coherence_score=float(seg.get("coherence_score", 0.0)),
                is_long_take=bool(seg.get("is_long_take", False)),
                shot_size=seg.get("shot_size", "未知"),
                camera_position=seg.get("camera_position", "未知"),
                camera_movement=seg.get("camera_movement", "未知"),
                framing=seg.get("framing", ""),
                lighting=seg.get("lighting", ""),
                color_tone=seg.get("color_tone", ""),
                key_objects=seg.get("key_objects", []),
                location=seg.get("location", ""),
                time_of_day=seg.get("time_of_day", ""),
                characters=seg.get("characters", []),
                action=seg.get("action", ""),
                emotion=seg.get("emotion", ""),
                dialogue=seg.get("dialogue", ""),
                style=seg.get("style", ""),
                atmosphere=seg.get("atmosphere", ""),
                culture=seg.get("culture", ""),
                tags=seg.get("tags", []),
            ))

        return VideoAnalysisResult(
            segments=segments,
            suggested_cut_points=[float(t) for t in data.get("suggested_cut_points", [])],
            overall_summary=data.get("overall_summary", ""),
            overall_coherence=float(data.get("overall_coherence", 0.0)),
        )
