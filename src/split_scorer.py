"""
切分积分计算器

职责：
- 根据 Segment 语义描述和 CV 候选切分点计算切分积分
- 应用长镜头/一镜到底保护
- 合并过短片段

与 VLM/LLM/ASR 解耦，只依赖 Segment 和配置。
"""
from typing import List, Dict, Any, Optional

from src.models import Segment, Boundary
from src.utils import logger


class SplitScorer:
    """切分积分计算器"""

    def __init__(self, split_scoring_config: Dict[str, Any]):
        self.config = split_scoring_config
        self.weights = self.config.get("weights", {
            "camera_change": 0.35,
            "subject_change": 0.25,
            "emotion_break": 0.15,
            "plot_shift": 0.10,
            "action_break": 0.10,
            "dialogue_break": 0.05,
        })
        self.thresholds = self.config.get("thresholds", {"high": 0.55, "medium": 0.40})
        self.protection = self.config.get("long_take_protection", {
            "enabled": True,
            "min_duration": 30.0,
            "threshold_boost": 0.15,
        })

    def compute_boundaries(
        self,
        segments: List[Segment],
        candidates: List[float],
        duration: float,
    ) -> List[Boundary]:
        """对每个候选切分点计算切分积分"""
        boundaries = []
        for t in candidates:
            if t <= 0 or t >= duration:
                continue

            left_seg = self._find_segment_at(t - 0.1, segments)
            right_seg = self._find_segment_at(t + 0.1, segments)
            if not left_seg or not right_seg or left_seg == right_seg:
                continue

            score, reasons = self._score_boundary(left_seg, right_seg)
            confidence = self._confidence_from_score(score)

            boundaries.append(Boundary(
                time_sec=t,
                score=round(score, 3),
                reason="; ".join(reasons) if reasons else "无明显变化",
                confidence=confidence,
            ))

        boundaries = sorted(boundaries, key=lambda b: b.time_sec)
        logger.info(f"候选切分点: {len(boundaries)} 个")
        for b in boundaries:
            logger.info(f"  t={b.time_sec:.2f}, score={b.score}, confidence={b.confidence}, reason={b.reason}")
        return boundaries

    def apply_long_take_protection(
        self,
        boundaries: List[Boundary],
        segments: List[Segment],
        duration: float,
    ) -> List[Boundary]:
        """长镜头/一镜到底保护：提升阈值"""
        if not self.protection.get("enabled", True):
            return boundaries

        min_duration = self.protection.get("min_duration", 30.0)
        boost = self.protection.get("threshold_boost", 0.15)

        cut_times = [0.0] + [b.time_sec for b in boundaries] + [duration]
        protected = []

        for i, b in enumerate(boundaries):
            prev_cut = cut_times[i]
            next_cut = cut_times[i + 2]
            segment_duration = next_cut - prev_cut

            seg = self._find_segment_at(b.time_sec + 0.01, segments)
            is_long_take = seg and getattr(seg, "is_long_take", False)
            high_coherence = seg and getattr(seg, "coherence_score", 0.0) >= 0.85

            if segment_duration >= min_duration or is_long_take or high_coherence:
                b.score = round(b.score - boost, 3)
                b.confidence = self._confidence_from_score(b.score)
                b.reason += " [长镜头/高连贯保护]"
                protected.append(b)

        logger.info(f"长镜头/高连贯保护: {len(protected)} 个边界被调整")
        return boundaries

    def merge_short_segments(self, cut_times: List[float], min_duration: float) -> List[float]:
        """合并过短片段"""
        if len(cut_times) < 3:
            return cut_times

        merged = [cut_times[0]]
        for i in range(1, len(cut_times) - 1):
            if cut_times[i] - merged[-1] < min_duration:
                continue
            merged.append(cut_times[i])
        merged.append(cut_times[-1])
        return merged

    def _score_boundary(self, left: Segment, right: Segment) -> tuple[float, List[str]]:
        """对左右两个段落计算切分积分，同时考虑段落内部连贯性"""
        score = 0.0
        reasons = []

        # 基础语义变化积分
        # 1. 机位切换
        if left.camera_position and right.camera_position and left.camera_position != right.camera_position:
            score += self.weights.get("camera_change", 0.35)
            reasons.append(f"机位变化: {left.camera_position} → {right.camera_position}")

        # 2. 主体/角色变化
        left_chars = set(left.characters or [])
        right_chars = set(right.characters or [])
        if left_chars and right_chars and left_chars != right_chars:
            score += self.weights.get("subject_change", 0.25)
            reasons.append(f"主体变化: {left_chars} → {right_chars}")

        # 3. 情绪断裂
        if left.emotion and right.emotion and left.emotion != right.emotion:
            score += self.weights.get("emotion_break", 0.15)
            reasons.append(f"情绪变化: {left.emotion} → {right.emotion}")

        # 4. 动作不连续
        if left.action and right.action:
            left_actions = set(left.action)
            right_actions = set(right.action)
            overlap = len(left_actions & right_actions) / max(len(left_actions), 1)
            if overlap < 0.3:
                score += self.weights.get("action_break", 0.10)
                reasons.append(f"动作不连续: {left.action} vs {right.action}")

        # 5. 场景/地点变化（剧情节点偏移）
        if left.location and right.location and left.location != right.location:
            score += self.weights.get("plot_shift", 0.10)
            reasons.append(f"场景变化: {left.location} → {right.location}")

        # 6. 对话/音频变化
        if left.dialogue and right.dialogue and left.dialogue != right.dialogue:
            score += self.weights.get("dialogue_break", 0.05)
            reasons.append("对话/台词变化")

        # 连贯性抑制：如果两侧段落本身强连贯，降低切分意愿
        left_coherence = getattr(left, "coherence_score", 0.0)
        right_coherence = getattr(right, "coherence_score", 0.0)
        if left_coherence > 0.7 and right_coherence > 0.7:
            score -= 0.25
            reasons.append("两侧段落内部连贯性高，抑制切分")

        return max(0.0, score), reasons

    def _confidence_from_score(self, score: float) -> str:
        high = self.thresholds.get("high", 0.55)
        medium = self.thresholds.get("medium", 0.40)
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

    @staticmethod
    def _find_segment_at(t: float, segments: List[Segment]) -> Optional[Segment]:
        for seg in segments:
            if seg.start <= t < seg.end:
                return seg
        return segments[-1] if segments else None
