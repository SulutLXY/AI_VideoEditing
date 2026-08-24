"""
剧情导向的镜头综合质量评分器

职责：
- 为每个 Shot 计算综合质量分，用于 Phase 2 的 take 选择
- 评分维度面向剧情匹配、视觉质量、叙事连贯、时长适配、元数据完整
- 与去重逻辑解耦，只负责评分

注：
- "内容多样性"由 Phase 2 在组内选择时根据上下文额外调整，不在此处计算。
"""
from typing import Dict, Any, List, Optional

from src.models import Shot


class QualityScorer:
    """镜头综合质量评分器（剧情导向 v0.3）"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # 默认权重：剧情匹配度最高，叙事连贯次之
        self.weights = self.config.get("weights") or {
            "script_match": 0.35,
            "visual_quality": 0.25,
            "narrative_coherence": 0.20,
            "duration_fit": 0.10,
            "metadata_complete": 0.10,
        }

    def score(self, shot: Shot, context: Optional["ScoreContext"] = None) -> float:
        """
        计算单个 Shot 的综合质量分 (0-10)

        Args:
            shot: 待评分镜头
            context: 可选上下文（上一个已选镜头、目标段落时长）
        """
        w = self.weights

        # 1. 剧情匹配度 (0-1)
        script_match = self._script_match_score(shot)

        # 2. 视觉质量 (0-1)
        visual_q = self._visual_quality_score(shot)

        # 3. 叙事连贯 (0-1)
        narrative = self._narrative_coherence_score(shot, context)

        # 4. 时长适配 (0-1)
        duration_fit = self._duration_fit_score(shot, context)

        # 5. 元数据完整度 (0-1)
        metadata_complete = self._metadata_completeness(shot)

        score = (
            w.get("script_match", 0.35) * script_match
            + w.get("visual_quality", 0.25) * visual_q
            + w.get("narrative_coherence", 0.20) * narrative
            + w.get("duration_fit", 0.10) * duration_fit
            + w.get("metadata_complete", 0.10) * metadata_complete
        )

        return round(score * 10, 2)

    def _script_match_score(self, shot: Shot) -> float:
        """剧情匹配度：剧本锚定置信度 + 行为/动作/台词匹配信号"""
        score = 0.0

        # 剧本锚定置信度
        if shot.script_anchor:
            score += float(shot.script_anchor.get("confidence", 0.0)) * 0.6

        # 行为与动作越具体，匹配潜力越高
        behavior = (shot.behavior or "") + " " + (shot.action or "") + " " + (shot.action_details or "")
        if behavior.strip():
            # 关键词越丰富，置信度越高（简单启发式）
            keyword_count = sum(1 for kw in ["奔跑", "追逐", "跃起", "变身", "接", "抱", "抚摸", "换装", "扔", "追", "跑", "跳"] if kw in behavior)
            score += min(0.25, keyword_count * 0.05)

        # 有台词且情绪明确加分
        if (shot.dialogue or shot.asr_text) and shot.emotion:
            score += 0.10

        # 主体明确加分
        if shot.primary_subject and shot.primary_subject not in ["", "未知"]:
            score += 0.05

        return min(1.0, score)

    def _visual_quality_score(self, shot: Shot) -> float:
        """视觉质量：综合画质、稳定、曝光、对焦"""
        visual_q = self._normalize_1_5(shot.visual_quality)
        stability = self._normalize_1_5(shot.stability)

        exposure_ok = 0.5
        if shot.exposure:
            exp = str(shot.exposure).lower()
            if "过曝" in exp or "欠曝" in exp or "曝光不足" in exp:
                exposure_ok = 0.0
            elif "正常" in exp:
                exposure_ok = 1.0

        focus_ok = 0.5
        if shot.focus:
            foc = str(shot.focus).lower()
            if "模糊" in foc or "失焦" in foc:
                focus_ok = 0.0
            elif "清晰" in foc:
                focus_ok = 1.0

        return (visual_q * 0.4 + stability * 0.3 + exposure_ok * 0.15 + focus_ok * 0.15)

    def _narrative_coherence_score(self, shot: Shot, context: Optional["ScoreContext"]) -> float:
        """叙事连贯：镜头内部连续性 + 与上下镜头的方向/景别衔接"""
        score = float(shot.continuity_score or 0.0) * 0.6

        # 方向明确加分
        if shot.direction and shot.direction not in ["", "未知", "静止"]:
            score += 0.15

        # 与上一个镜头的连贯性（如果提供上下文）
        if context and context.prev_shot:
            prev = context.prev_shot
            # 同角色或同主体加分
            if prev.primary_subject and shot.primary_subject:
                if prev.primary_subject == shot.primary_subject:
                    score += 0.10
                else:
                    # 主体变化但属于自然切换（如云琛 → 黑猫）也加分
                    score += 0.05
            # 方向连续性：避免越轴
            if prev.direction and shot.direction:
                score += self._direction_continuity(prev.direction, shot.direction) * 0.15

        return min(1.0, score)

    @staticmethod
    def _direction_continuity(dir_a: str, dir_b: str) -> float:
        """方向连续性：同向或自然过渡给高分，明显反向（越轴风险）给低分"""
        a = str(dir_a).lower()
        b = str(dir_b).lower()
        # 明确相反方向
        if ("左到右" in a and "右到左" in b) or ("右到左" in a and "左到右" in b):
            return 0.0
        # 同向
        if ("左到右" in a and "左到右" in b) or ("右到左" in a and "右到左" in b):
            return 1.0
        # 静止过渡自然
        if "静止" in a or "静止" in b:
            return 0.8
        return 0.5

    def _duration_fit_score(self, shot: Shot, context: Optional["ScoreContext"]) -> float:
        """时长适配：结合全局最佳时长与段落目标时长"""
        dur = shot.duration_sec
        if dur <= 0:
            return 0.0

        # 全局最佳区间 1~15 秒
        if dur < 1.0:
            return 0.3
        if dur <= 15.0:
            base = 1.0
        elif dur <= 30.0:
            base = 0.8
        elif dur <= 60.0:
            base = 0.6
        else:
            base = 0.4

        # 如果有段落目标时长，进一步判断变速后能否适配
        if context and context.target_segment_duration:
            target = context.target_segment_duration
            # 0.75~1.5 倍变速可覆盖
            if target * 0.75 <= dur <= target * 1.5:
                base = max(base, 0.9)
            elif target * 0.5 <= dur <= target * 2.0:
                base = max(base, 0.6)
            else:
                base *= 0.7

        return base

    def _metadata_completeness(self, shot: Shot) -> float:
        """评估关键元数据字段完整度 (0-1)"""
        fields = [
            shot.shot_size,
            shot.camera_position,
            shot.camera_movement,
            shot.location,
            shot.emotion,
            shot.action,
            shot.behavior,
            shot.pace,
            shot.primary_subject,
            shot.characters,
        ]
        filled = 0
        for f in fields:
            if f is not None and f != "" and f != [] and f != "未知":
                filled += 1
        return filled / len(fields)

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


class ScoreContext:
    """评分上下文"""

    def __init__(
        self,
        prev_shot: Optional[Shot] = None,
        target_segment_duration: Optional[float] = None,
    ):
        self.prev_shot = prev_shot
        self.target_segment_duration = target_segment_duration
