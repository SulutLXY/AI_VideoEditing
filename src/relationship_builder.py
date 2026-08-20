"""
镜头关系图构建器

职责：
- 按源文件和时间排序 Shot
- 分析相邻镜头的连贯性
- 生成 Relationship 对象（关系类型 + 连贯性分数）

与模型/LLM 解耦，只基于 Shot 元数据做启发式判断。
"""
from typing import List, Tuple

from src.models import Shot, Relationship, Relationships
from src.utils import tc_to_sec, logger


class RelationshipBuilder:
    """镜头关系图构建器"""

    def build(self, shots: List[Shot]) -> List[Shot]:
        """对 Shot 列表排序并建立前后关系"""
        shots = sorted(shots, key=lambda s: (s.source_file, tc_to_sec(s.tc_in, s.fps)))

        for i in range(len(shots)):
            if i > 0:
                prev = shots[i - 1]
                curr = shots[i]
                rel_type, score = self._classify(prev, curr)
                curr.relationships.prev = Relationship(
                    shot_id=prev.shot_id,
                    relationship_type=rel_type,
                    coherence_score=score,
                )
                prev.relationships.next = Relationship(
                    shot_id=curr.shot_id,
                    relationship_type=rel_type,
                    coherence_score=score,
                )

        logger.info(f"关系图构建完成: {len(shots)} 个 Shot")
        return shots

    def _classify(self, prev: Shot, curr: Shot) -> Tuple[str, float]:
        """判断两个相邻 Shot 的关系类型和连贯性分数"""
        score = 0.8
        reasons = []

        # 机位变化 → 连贯性低
        if prev.camera_position and curr.camera_position and prev.camera_position != curr.camera_position:
            score -= 0.3
            reasons.append("机位变化")

        # 场景变化 → 连贯性低
        if prev.location and curr.location and prev.location != curr.location:
            score -= 0.35
            reasons.append("场景变化")

        # 情绪变化 → 中等影响
        if prev.emotion and curr.emotion and prev.emotion != curr.emotion:
            score -= 0.2
            reasons.append("情绪变化")

        # 主体变化 → 中等影响
        prev_chars = set(prev.characters or [])
        curr_chars = set(curr.characters or [])
        if prev_chars and curr_chars and prev_chars != curr_chars:
            score -= 0.15
            reasons.append("主体变化")

        # 连续对话 → 连贯性高
        if prev.asr_text and curr.asr_text and (prev.asr_text in curr.asr_text or curr.asr_text in prev.asr_text):
            score += 0.1
            reasons.append("对话连续")

        score = max(0.0, min(1.0, score))

        if score >= 0.75:
            return ("情绪延续" if not reasons else "自然过渡"), round(score, 2)

        # 按原因优先级判断关系类型
        if "场景变化" in reasons:
            return "场景切换", round(score, 2)
        if "机位变化" in reasons:
            return "机位跳切", round(score, 2)
        if "主体变化" in reasons:
            return "动作衔接", round(score, 2)
        if "情绪变化" in reasons:
            return "情绪转折", round(score, 2)

        return "时间跳跃", round(score, 2)
