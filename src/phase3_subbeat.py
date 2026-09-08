"""
Phase 3 Sub-beat 拆分模块

把 Phase 2 生成的 beat 进一步拆分为语义更窄的 sub-beat，
作为 Phase 3 镜头分配的最小单元。

拆分依据：
1. dialogue_entries：每个对白条目对应一个 sub-beat
2. gender_transition：状态转换单独成 sub-beat
3. key_actions：剩余动作按关键动作拆分
"""
import copy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.utils import logger


@dataclass
class SubBeat:
    """sub-beat：比 beat 更细的剧情单元"""
    sub_beat_id: str
    parent_beat_id: str
    act: str
    scene: str
    content: str
    key_actions: List[str] = field(default_factory=list)
    key_dialogue: str = ""
    emotion: str = ""
    pace: str = "正常"
    gender_state: str = ""
    gender_transition: str = ""
    estimated_duration: float = 0.0
    start_in_parent: float = 0.0
    end_in_parent: float = 0.0
    priority: int = 0
    dialogue_entries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SubBeatSplitter:
    """把 Phase 2 的 beat 拆分为 sub-beat"""

    def __init__(self, beats_analysis: Dict[str, Any]):
        self.beats_analysis = beats_analysis

    def split(self) -> List[SubBeat]:
        """执行拆分，返回全局有序的 sub-beat 列表"""
        beats = self.beats_analysis.get("beats", [])
        sub_beats: List[SubBeat] = []

        for beat in beats:
            beat_id = beat.get("beat_id", "")
            if not beat_id:
                continue
            sub_beats.extend(self._split_beat(beat))

        logger.info(f"SubBeat 拆分完成: {len(beats)} 个 beat -> {len(sub_beats)} 个 sub-beat")
        for sb in sub_beats:
            logger.debug(
                f"  {sb.sub_beat_id}: actions={sb.key_actions}, dialogue={sb.key_dialogue[:30]}..."
            )
        return sub_beats

    def _split_beat(self, beat: Dict[str, Any]) -> List[SubBeat]:
        """拆分单个 beat"""
        beat_id = beat.get("beat_id", "")
        act = beat.get("act", "")
        scene = beat.get("scene", "")
        content = beat.get("content", "")
        emotion = beat.get("emotion", "")
        pace = beat.get("pace", "正常")
        gender_state = beat.get("gender_state", "")
        gender_transition = beat.get("gender_transition", "")
        beat_duration = float(beat.get("estimated_duration", 0.0) or 0.0)
        priority = int(beat.get("priority", 0) or 0)
        key_actions = list(beat.get("key_actions", []) or [])
        key_dialogue = beat.get("key_dialogue", "")
        dialogue_entries = list(beat.get("dialogue_entries", []) or [])

        # 锁定的 beat 不再拆分：用户指定的节点（含一键分镜头插入的）保持完整
        if beat.get("locked"):
            return [SubBeat(
                sub_beat_id=f"{beat_id}-1",
                parent_beat_id=beat_id,
                act=act,
                scene=scene,
                content=content,
                key_actions=key_actions,
                key_dialogue=key_dialogue,
                emotion=emotion,
                pace=pace,
                gender_state=gender_state,
                gender_transition=gender_transition,
                estimated_duration=beat_duration,
                start_in_parent=0.0,
                end_in_parent=beat_duration,
                priority=priority,
                dialogue_entries=dialogue_entries,
            )]

        sub_beats: List[SubBeat] = []
        cursor = 0.0

        # 1. 按 dialogue_entries 拆分，每个 entry 一个 sub-beat
        # 注意：不按 key_actions 二次拆分，避免素材不足时拆出无候选的 sub-beat
        if dialogue_entries:
            total_dialogue_dur = sum(
                float(e.get("estimated_duration", 0.0) or 0.0) for e in dialogue_entries
            ) or beat_duration

            for idx, entry in enumerate(dialogue_entries):
                entry_dur = float(entry.get("estimated_duration", 0.0) or 0.0)
                if total_dialogue_dur > 0 and beat_duration > 0:
                    alloc = beat_duration * (entry_dur / total_dialogue_dur)
                else:
                    alloc = beat_duration / len(dialogue_entries) if dialogue_entries else beat_duration

                # 该 sub-beat 的关键动作：按 entry 数量平均分配
                actions_for_this = self._allocate_actions_for_entry(key_actions, idx, len(dialogue_entries))

                sb = SubBeat(
                    sub_beat_id=f"{beat_id}-{idx + 1}",
                    parent_beat_id=beat_id,
                    act=act,
                    scene=scene,
                    content=content,
                    key_actions=actions_for_this,
                    key_dialogue=entry.get("text", ""),
                    emotion=entry.get("emotion", emotion),
                    pace=entry.get("pace", pace),
                    gender_state=entry.get("gender_state", gender_state),
                    gender_transition="",
                    estimated_duration=alloc,
                    start_in_parent=cursor,
                    end_in_parent=cursor + alloc,
                    priority=priority,
                    dialogue_entries=[entry],
                )
                sub_beats.append(sb)
                cursor += alloc

        # 2. 如果有 gender_transition 且未被 dialogue_entries 覆盖，单独拆分
        # 暂时简化：如果 beat 有 gender_transition，且没有对应的 entry，插入一个转换 sub-beat
        if gender_transition and not any(sb.gender_transition for sb in sub_beats):
            # 在 beat 中间插入一个短的转换 sub-beat
            transition_dur = min(2.0, beat_duration * 0.25) if beat_duration > 0 else 2.0
            # 找到最合适的位置：通常是 key_actions 中出现 "女装/男装/变身" 的位置
            transition_actions = [a for a in key_actions if any(k in a for k in ["女装", "男装", "变身", "换装"])]
            if not transition_actions:
                transition_actions = [gender_transition]

            sb = SubBeat(
                sub_beat_id=f"{beat_id}-T",
                parent_beat_id=beat_id,
                act=act,
                scene=scene,
                content=f"状态转换: {gender_transition}",
                key_actions=transition_actions,
                key_dialogue="",
                emotion=emotion,
                pace="爆发",
                gender_state=gender_state,
                gender_transition=gender_transition,
                estimated_duration=transition_dur,
                start_in_parent=max(0.0, beat_duration / 2 - transition_dur / 2),
                end_in_parent=min(beat_duration, beat_duration / 2 + transition_dur / 2),
                priority=priority,
            )
            sub_beats.append(sb)

        # 3. 如果既没有 dialogue_entries 也没有 gender_transition，整个 beat 作为一个 sub-beat
        # 不再按 key_actions 拆分，避免拆出无候选的 sub-beat
        if not sub_beats and key_actions:
            sb = SubBeat(
                sub_beat_id=f"{beat_id}-1",
                parent_beat_id=beat_id,
                act=act,
                scene=scene,
                content=content,
                key_actions=key_actions,
                key_dialogue=key_dialogue,
                emotion=emotion,
                pace=pace,
                gender_state=gender_state,
                gender_transition="",
                estimated_duration=beat_duration,
                start_in_parent=0.0,
                end_in_parent=beat_duration,
                priority=priority,
            )
            sub_beats.append(sb)

        # 4. 兜底：如果以上都没有，整个 beat 作为一个 sub-beat
        if not sub_beats:
            sb = SubBeat(
                sub_beat_id=f"{beat_id}-1",
                parent_beat_id=beat_id,
                act=act,
                scene=scene,
                content=content,
                key_actions=key_actions,
                key_dialogue=key_dialogue,
                emotion=emotion,
                pace=pace,
                gender_state=gender_state,
                gender_transition=gender_transition,
                estimated_duration=beat_duration,
                start_in_parent=0.0,
                end_in_parent=beat_duration,
                priority=priority,
            )
            sub_beats.append(sb)

        # 5. 重新归一化时间，确保子区间覆盖整个 beat
        sub_beats = self._normalize_durations(sub_beats, beat_duration)

        # 6. 按 start_in_parent 排序
        sub_beats.sort(key=lambda x: x.start_in_parent)
        return sub_beats

    @staticmethod
    def _allocate_actions_for_entry(
        key_actions: List[str], entry_index: int, total_entries: int
    ) -> List[str]:
        """把 beat 的 key_actions 分配给各个 dialogue entry"""
        if not key_actions:
            return []
        if total_entries <= 1:
            return key_actions

        # 简单策略：平均分配
        chunk_size = max(1, len(key_actions) // total_entries)
        start = entry_index * chunk_size
        end = start + chunk_size if entry_index < total_entries - 1 else len(key_actions)
        return key_actions[start:end]

    @staticmethod
    def _normalize_durations(sub_beats: List[SubBeat], parent_duration: float) -> List[SubBeat]:
        """归一化 sub-beat 时长，使其之和等于 parent_duration"""
        if not sub_beats or parent_duration <= 0:
            return sub_beats

        total = sum(sb.estimated_duration for sb in sub_beats)
        if total <= 0:
            for sb in sub_beats:
                sb.estimated_duration = parent_duration / len(sub_beats)
                sb.start_in_parent = 0.0
                sb.end_in_parent = sb.estimated_duration
            return sub_beats

        # 按比例缩放
        for sb in sub_beats:
            sb.estimated_duration = parent_duration * (sb.estimated_duration / total)

        # 重新计算 start/end
        cursor = 0.0
        for sb in sub_beats:
            sb.start_in_parent = cursor
            sb.end_in_parent = cursor + sb.estimated_duration
            cursor = sb.end_in_parent

        # 修正浮点误差
        if sub_beats:
            sub_beats[-1].end_in_parent = parent_duration

        return sub_beats


def load_sub_beats(output_dir: str) -> List[SubBeat]:
    """从 Phase 2 输出加载 sub-beat，优先使用预生成的 sub_beats"""
    import os
    from src.utils import load_json

    path = os.path.join(output_dir, "script_beats_analysis.json")
    if not os.path.exists(path):
        logger.warning(f"未找到剧本节奏分析文件: {path}")
        return []

    data = load_json(path)

    # 优先读取 Phase 2 预生成的 sub_beats
    sub_beats: List[SubBeat] = []
    for beat in data.get("beats", []):
        for sb_data in beat.get("sub_beats", []):
            try:
                sub_beats.append(SubBeat(**sb_data))
            except Exception as e:
                logger.warning(f"加载 sub-beat 失败: {sb_data.get('sub_beat_id', '?')} - {e}")

    if sub_beats:
        logger.info(f"从 script_beats_analysis.json 加载 {len(sub_beats)} 个预生成 sub-beat")
        return sub_beats

    # 回退：动态拆分（兼容旧数据）
    logger.info("未找到预生成 sub-beat，使用 SubBeatSplitter 动态拆分")
    splitter = SubBeatSplitter(data)
    return splitter.split()
