"""
本地 VLM 服务封装

把 video-intelligence-extractor 的 VisionEngine 适配成项目 VLMService 的接口，
供 Phase 1 在 `models.vlm.provider == "local"` 时调用。
"""
import os
from typing import List, Dict, Tuple, Any

from src.models import Segment
from src.utils import logger


class LocalVLMService:
    """基于 Qwen2.5-VL 的本地视觉理解服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local_cfg = config.get("models", {}).get("local", {})
        vision_cfg = local_cfg.get("vision", {})

        self.enabled = bool(local_cfg.get("enabled", False))
        self.device = local_cfg.get("device", "cuda")
        self.cache_dir = local_cfg.get("cache_dir")
        self.model_id = vision_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.model_path = vision_cfg.get("model_path")
        self.load_in_4bit = vision_cfg.get("load_in_4bit", True)
        self.max_new_tokens = vision_cfg.get("max_new_tokens", 512)
        self.keyframe_count = vision_cfg.get("keyframe_count", 3)

        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from src.local_models.vision_engine import VisionEngine
            self._engine = VisionEngine(
                model_id=self.model_id,
                model_path=self.model_path,
                device=self.device,
                load_in_4bit=self.load_in_4bit,
                max_new_tokens=self.max_new_tokens,
                cache_dir=self.cache_dir,
            )
        return self._engine

    def sample_frames(self, video_path: str, duration: float, temp_dir: str) -> List[Tuple[float, str]]:
        """本地服务不依赖外部 base64 帧，返回空列表即可；关键帧由 engine 内部抽取"""
        return []

    def analyze_whole_video(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> Dict[str, Any]:
        """对单个视频片段（或已切分 Shot）做完整内容分析"""
        if not self.enabled:
            logger.warning("[LocalVLM] 本地模型未启用，跳过本地视觉分析")
            return {}

        logger.info(f"[LocalVLM] 本地视觉分析: {os.path.basename(video_path)}")
        try:
            engine = self._get_engine()
            result = engine.process_video(video_path, keyframe_count=self.keyframe_count)
            return self._map_to_shot_fields(result)
        except Exception as e:
            logger.error(f"[LocalVLM] 本地视觉分析失败: {e}")
            return {}

    def analyze_temporal_segments(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> List[Segment]:
        """本地模型目前按整段分析，返回单个覆盖全长的 segment"""
        description = self.analyze_whole_video(video_path, frames, duration)
        return [Segment(
            start=0.0,
            end=duration,
            description=description.get("action", ""),
            location=description.get("location", ""),
            time_of_day=description.get("time_of_day", ""),
            characters=description.get("characters", []),
            action=description.get("action", ""),
            emotion=description.get("emotion", ""),
            dialogue=description.get("dialogue", ""),
            camera_position=description.get("camera_position", ""),
            camera_movement=description.get("camera_movement", ""),
            shot_size=description.get("shot_size", ""),
            framing=description.get("framing", ""),
            lighting=description.get("lighting", ""),
            color_tone=description.get("color_tone", ""),
            style=description.get("style", ""),
            atmosphere=description.get("atmosphere", ""),
            culture=description.get("culture", ""),
            key_objects=description.get("key_objects", []),
            tags=description.get("tags", []),
            direction=description.get("direction", ""),
            performance=description.get("performance", ""),
            action_details=description.get("action_details", ""),
            continuity_score=float(description.get("continuity_score", 0.0) or 0.0),
            continuity_notes=description.get("continuity_notes", ""),
            coherence_score=0.0,
            is_long_take=False,
        )]

    def validate_cut_candidates(
        self,
        candidates: List[Tuple[float, List[Tuple[float, str]]]],
    ) -> Dict[float, bool]:
        """本地模型暂不支持切点验证，返回空，由 Phase 0 CV 独立处理"""
        logger.debug("[LocalVLM] 本地模型不支持 VLM 切点验证，跳过")
        return {}

    @staticmethod
    def _map_to_shot_fields(result: Dict[str, Any]) -> Dict[str, Any]:
        """把 VisionEngine 输出字段映射到 Shot 模型字段"""
        return {
            "location": result.get("location", ""),
            "time_of_day": result.get("time_of_day", ""),
            "characters": [],  # 由 Face 服务填充
            "action": result.get("action", ""),
            "action_details": result.get("action_details", ""),
            "emotion": result.get("emotion", ""),
            "dialogue": "",
            "camera_position": result.get("camera_position", ""),
            "camera_movement": result.get("camera_movement", ""),
            "shot_size": result.get("shot_size", ""),
            "framing": result.get("framing", ""),
            "lighting": result.get("lighting", ""),
            "color_tone": result.get("color_tone", ""),
            "style": result.get("style", ""),
            "atmosphere": result.get("atmosphere", ""),
            "culture": result.get("culture", ""),
            "key_objects": result.get("key_objects", []),
            "tags": result.get("tags", []),
            "direction": result.get("direction", ""),
            "performance": result.get("performance", ""),
            "continuity_score": float(result.get("continuity_score", 0.0) or 0.0),
            "continuity_notes": result.get("continuity_notes", ""),
            "notes": result.get("notes", ""),
            "internal_segments": result.get("key_frames", []),
        }
