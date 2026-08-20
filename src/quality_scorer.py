"""
统一质量分计算模块

职责：
- 为每个 Shot 计算综合质量分，用于 Phase 2 的 take 选择
- 归一化 visual_quality / stability
- 融入剧本匹配置信度、台词、时长、元数据完整度

与去重逻辑解耦，只负责评分。
"""
from typing import Dict, Any

from src.models import Shot


class QualityScorer:
    """镜头综合质量评分器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # 支持 weights（推荐）或旧版 quality_weights
        # 默认权重：剧情匹配置信度最高，其次视觉质量与稳定性
        self.weights = self.config.get("weights") or self.config.get("quality_weights", {
            "script_confidence": 0.35,
            "visual_quality": 0.20,
            "stability": 0.15,
            "dialogue": 0.10,
            "duration": 0.10,
            "metadata_complete": 0.10,
        })

    def score(self, shot: Shot) -> float:
        """计算单个 Shot 的综合质量分 (0-10)"""
        w = self.weights

        # 1. 视觉质量（假设 1-5，归一化到 0-1）
        visual_q = self._normalize_1_5(shot.visual_quality)

        # 2. 稳定性（假设 1-5，归一化到 0-1）
        stability = self._normalize_1_5(shot.stability)

        # 3. 剧本匹配置信度（0-1）
        script_conf = 0.0
        if shot.script_anchor:
            script_conf = float(shot.script_anchor.get("confidence", 0.0))

        # 4. 台词信息（有台词加分，避免空镜头占优）
        dialogue_bonus = 0.5 if (shot.asr_text or shot.dialogue) else 0.0

        # 5. 时长适中度（1-30 秒最佳，过短或过长扣分）
        duration_score = self._duration_score(shot.duration_sec)

        # 6. 元数据完整度（关键字段越多越完整）
        metadata_complete = self._metadata_completeness(shot)

        score = (
            w.get("visual_quality", 0.25) * visual_q
            + w.get("stability", 0.20) * stability
            + w.get("script_confidence", 0.20) * script_conf
            + w.get("dialogue", 0.15) * dialogue_bonus
            + w.get("duration", 0.10) * duration_score
            + w.get("metadata_complete", 0.10) * metadata_complete
        )

        # 映射到 0-10 制，便于阅读
        return round(score * 10, 2)

    @staticmethod
    def _normalize_1_5(value: Any) -> float:
        if value is None:
            return 0.5
        try:
            v = float(value)
            if v <= 0:
                return 0.0
            return min(1.0, v / 5.0)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _duration_score(duration: float) -> float:
        if duration <= 0:
            return 0.0
        if duration < 1.0:
            return 0.3
        if duration <= 30.0:
            return 1.0
        if duration <= 60.0:
            return 0.8
        return 0.6

    @staticmethod
    def _metadata_completeness(shot: Shot) -> float:
        """评估关键元数据字段完整度 (0-1)"""
        fields = [
            shot.shot_size,
            shot.camera_position,
            shot.camera_movement,
            shot.location,
            shot.emotion,
            shot.action,
            shot.characters,
        ]
        filled = 0
        for f in fields:
            if f is not None and f != "" and f != [] and f != "未知":
                filled += 1
        return filled / len(fields)
