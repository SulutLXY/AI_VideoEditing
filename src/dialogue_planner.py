"""
DialoguePlanner: 剧本对白规划器

职责：
1. 解析 ScriptBeat.key_dialogue，提取说话人、状态提示、对白文本。
2. 维护角色状态机（如云琛男装/女装切换）。
3. 为每条对白估算 1x 语速时长，并计算最大语速下的最短时长。
4. 输出 voice_cast.json 角色-音色映射，供 Phase 4 TTS 使用。
5. 生成结构化 dialogue_plan.json，供 Phase 3/4 做音画同步约束。

输入：ScriptBeat 列表
输出：更新后的 ScriptBeat 列表（含 dialogue_entries）+ voice_cast 字典
"""
import re
import os
from typing import List, Dict, Any, Optional, Tuple

from src.models import ScriptBeat, DialogueEntry
from src.utils import logger, save_json


class DialoguePlanner:
    """对白规划器"""

    # 默认语速：中文字/秒
    PACE_CHARS_PER_SEC = {
        "爆发": 5.5,
        "快": 5.0,
        "正常": 4.5,
        "慢": 3.5,
        "静止": 3.0,
    }

    # 默认角色音色映射（Edge TTS）
    DEFAULT_VOICE_CAST = {
        "云琛-男装": {
            "role": "中性",
            "voice_id": "zh-CN-YunxiNeural",
            "description": "云琛男装状态，偏中性/男声",
        },
        "云琛-女装": {
            "role": "少女",
            "voice_id": "zh-CN-XiaoxiaoNeural",
            "description": "云琛女装状态，少女声",
        },
        "小六": {
            "role": "少年",
            "voice_id": "zh-CN-YunjianNeural",
            "description": "小六，少年声",
        },
        "default": {
            "role": "默认",
            "voice_id": "zh-CN-YunxiNeural",
            "description": "未识别角色，使用默认音色",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        audio_cfg = config.get("audio", {})
        self.max_speed = float(audio_cfg.get("max_dialogue_speed", 2.0))
        self.base_chars_per_sec = float(audio_cfg.get("base_chars_per_sec", 4.5))
        self.voice_cast_override = audio_cfg.get("voice_cast", {}) or {}
        self.output_dir = config.get("paths", {}).get("output", "./output")

        # 合并用户覆盖的音色映射
        self.voice_cast = self._build_voice_cast()

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def plan(self, beats: List[ScriptBeat]) -> Tuple[List[ScriptBeat], Dict[str, Any]]:
        """为所有 beat 解析对白，生成 dialogue_entries 和 voice_cast"""
        logger.info("=" * 60)
        logger.info("DialoguePlanner: 解析剧本对白并规划配音")
        logger.info("=" * 60)

        # 维护角色状态机
        character_states: Dict[str, str] = {}
        total_entries = 0
        total_dialogue_duration_1x = 0.0

        for beat in beats:
            entries, character_states = self._parse_beat_dialogue(
                beat, character_states
            )
            beat.dialogue_entries = entries
            total_entries += len(entries)
            total_dialogue_duration_1x += sum(e.estimated_duration for e in entries)

        report = {
            "total_beats": len(beats),
            "beats_with_dialogue": sum(1 for b in beats if b.dialogue_entries),
            "total_entries": total_entries,
            "total_dialogue_duration_1x": round(total_dialogue_duration_1x, 2),
            "total_dialogue_duration_max_speed": round(
                total_dialogue_duration_1x / self.max_speed, 2
            ),
            "max_speed": self.max_speed,
            "base_chars_per_sec": self.base_chars_per_sec,
            "voice_cast": self.voice_cast,
            "beats": [
                {
                    "beat_id": b.beat_id,
                    "entries": [e.to_dict() for e in b.dialogue_entries],
                    "beat_dialogue_duration_1x": round(
                        sum(e.estimated_duration for e in b.dialogue_entries), 2
                    ),
                    "beat_dialogue_duration_min": round(
                        sum(e.estimated_duration for e in b.dialogue_entries) / self.max_speed, 2
                    ),
                }
                for b in beats
            ],
        }

        save_json(report, os.path.join(self.output_dir, "dialogue_plan.json"))
        logger.info(
            f"对白规划完成: {total_entries} 条对白, "
            f"1x 总时长 {total_dialogue_duration_1x:.1f}s, "
            f"最大语速 {self.max_speed}x 下最短 {total_dialogue_duration_1x / self.max_speed:.1f}s"
        )
        return beats, report

    # ------------------------------------------------------------------
    # 解析单条 beat 的对白
    # ------------------------------------------------------------------
    def _parse_beat_dialogue(
        self,
        beat: ScriptBeat,
        character_states: Dict[str, str],
    ) -> Tuple[List[DialogueEntry], Dict[str, str]]:
        """解析一个 beat 的 key_dialogue，返回 dialogue_entries 和更新后的状态机"""
        entries: List[DialogueEntry] = []

        # 优先使用 beat 显式标记的性别状态/切换
        character_states = self._apply_beat_gender_state(
            beat, character_states, before_dialogue=True
        )

        if not beat.key_dialogue:
            # 无对白也要应用状态切换的后半部分，保证下一 beat 状态正确
            character_states = self._apply_beat_gender_state(
                beat, character_states, before_dialogue=False
            )
            return entries, character_states

        # 关键台词可能是纯引号文本（无说话人前缀），先尝试拆出所有引号内对白
        raw_text = beat.key_dialogue.strip()

        # 优先尝试解析带说话人前缀的整行/多行对白
        dialogue_items = []
        has_explicit_speaker = False
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = self._parse_dialogue_line(line)
            if parsed:
                dialogue_items.append(parsed)
                has_explicit_speaker = True

        # 如果没有解析出带说话人前缀的对白，再尝试从引号中提取纯文本
        if not has_explicit_speaker:
            quoted_segments = self._extract_quoted_segments(raw_text)
            if quoted_segments:
                for segment in quoted_segments:
                    parsed = self._parse_dialogue_line(segment)
                    if parsed:
                        dialogue_items.append(parsed)
                    else:
                        # 只有纯文本，没有说话人前缀，从 beat.content 推断
                        speaker, stage_direction = self._infer_speaker_and_direction(beat.content)
                        text = segment.strip('""').strip()
                        dialogue_items.append((speaker, stage_direction, text))
            else:
                # 既没有引号也没有说话人前缀，整段按无对白处理
                pass

        for speaker, stage_direction, text in dialogue_items:
            if not text:
                continue

            # 更新状态机：以当前 speaker 的最新状态为准
            character_states = self._update_state_from_direction(
                speaker, stage_direction, character_states
            )

            # 确定当前 speaker 的性别/状态
            gender_state = self._resolve_gender_state(speaker, character_states)

            # 目标音色角色
            target_voice_role = self._resolve_voice_role(speaker, gender_state)

            # 估算时长
            pace = self._resolve_pace(beat.pace, stage_direction)
            chars_per_sec = self.PACE_CHARS_PER_SEC.get(pace, self.base_chars_per_sec)
            estimated_duration = len(text) / chars_per_sec if chars_per_sec > 0 else 0.0

            entry = DialogueEntry(
                speaker=speaker,
                text=text,
                gender_state=gender_state,
                start_in_beat=0.0,  # Phase 3 再根据镜头排布写入具体时间
                estimated_duration=round(estimated_duration, 2),
                pace=pace,
                emotion=self._resolve_emotion(beat.emotion, stage_direction),
                is_offscreen="画外音" in stage_direction or "画外" in stage_direction,
                target_voice_role=target_voice_role,
            )
            entries.append(entry)

        # beat.content 里描述的状态变化（如云琛换装）应影响下一 beat 的对白，
        # 而不是当前 beat 已经说出的对白。
        character_states = self._update_states_from_content(
            beat.content, character_states
        )

        # 应用 beat 显式标记的状态切换后半部分（如男装→女装最终切到女装）
        character_states = self._apply_beat_gender_state(
            beat, character_states, before_dialogue=False
        )

        return entries, character_states

    @staticmethod
    def _parse_dialogue_line(line: str) -> Optional[Tuple[str, str, str]]:
        """
        解析单行对白。
        支持格式：
        - 云琛（边跑边喊）：天天又叫又跑的？
        - 小六（画外音）：霍帮招聘护卫了！
        - 云琛（小声自言自语）：五年了...
        - 云琛（画外音，故意压低嗓音，用低沉的男人腔）：接着——
        - 小六: 什么？
        """
        # 尝试匹配「说话人（状态）：文本」
        pattern = r"^(.+?)[（(](.+?)[）)][:：]\s*(.+)$"
        m = re.match(pattern, line)
        if m:
            speaker = m.group(1).strip()
            stage_direction = m.group(2).strip()
            text = m.group(3).strip().strip('""').strip()
            return speaker, stage_direction, text

        # 没有状态括号，只匹配「说话人：文本」
        pattern2 = r"^(.+?)[:：]\s*(.+)$"
        m2 = re.match(pattern2, line)
        if m2:
            speaker = m2.group(1).strip()
            text = m2.group(2).strip().strip('""').strip()
            return speaker, "", text

        # 完全无法解析，视作无对白
        return None

    @staticmethod
    def _extract_quoted_segments(text: str) -> List[str]:
        """从关键台词字段中提取所有被引号包裹的对白片段"""
        segments = []
        # 匹配中文引号 ""xxx"" 或英文引号 "xxx"
        for quote in ('"', '"', '"'):
            if quote in text:
                pattern = re.compile(rf"{re.escape(quote)}(.*?){re.escape(quote)}")
                segments.extend(pattern.findall(text))
        # 去重并保持顺序
        seen = set()
        result = []
        for s in segments:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return result

    @staticmethod
    def _infer_speaker_and_direction(content: str) -> Tuple[str, str]:
        """根据 beat.content 推断说话人和舞台提示"""
        if not content:
            return "云琛", ""

        content = str(content)

        # 小六相关：如果内容只讲小六说话且没提到云琛说话
        xiaoliu_speaking = any(k in content for k in ["小六", "小六的"])
        yunchen_speaking = "云琛" in content

        if xiaoliu_speaking and not yunchen_speaking:
            speaker = "小六"
        else:
            # 默认主角云琛
            speaker = "云琛"

        # 推断舞台提示
        directions = []
        if "画外音" in content or "画外" in content:
            directions.append("画外音")
        if "压低嗓音" in content or "男人腔" in content:
            directions.append("压低嗓音，用男人腔")
        elif "低声" in content or "小声" in content or "自言自语" in content:
            directions.append("低声")
        elif "边跑边喊" in content:
            directions.append("边跑边喊")
        elif "喊" in content:
            directions.append("喊")
        elif "吐槽" in content:
            directions.append("吐槽")

        stage_direction = "，".join(directions)
        return speaker, stage_direction

    # ------------------------------------------------------------------
    # 角色状态机
    # ------------------------------------------------------------------
    def _update_states_from_content(
        self,
        content: str,
        character_states: Dict[str, str],
    ) -> Dict[str, str]:
        """根据 beat.content 更新角色状态（如云琛换装）"""
        if not content:
            return character_states

        content = str(content)

        # 云琛状态切换关键词
        # 先判断明确的换装/变声过渡：女装/男装过渡关键字不要互相覆盖
        female_transition = ["露出女装", "变回女装", "变作女装", "换上女装", "换成女装"]
        male_transition = ["穿上男装", "恢复男装", "换上男装", "换成男装"]

        if any(k in content for k in female_transition):
            character_states["云琛"] = "女装"
        elif any(k in content for k in male_transition):
            character_states["云琛"] = "男装"
        # 其次根据整体描述（无明确过渡时）：仙女/少女/娇声→女装；压低嗓音/男人腔/男声→男装
        elif any(k in content for k in ["仙女", "少女", "娇声", "柔声"]):
            character_states["云琛"] = "女装"
        elif any(k in content for k in ["压低嗓音", "男人腔", "男声", "粗声"]):
            character_states["云琛"] = "男装"

        return character_states

    def _apply_beat_gender_state(
        self,
        beat: ScriptBeat,
        character_states: Dict[str, str],
        before_dialogue: bool = True,
    ) -> Dict[str, str]:
        """
        应用 beat 显式标记的 gender_state / gender_transition。
        - before_dialogue=True：若存在 transition，取前半部分作为当前状态；否则用 gender_state。
        - before_dialogue=False：若存在 transition，取后半部分作为下一 beat 的起始状态。
        """
        transition = (beat.gender_transition or "").strip()
        if transition:
            parts = [p.strip() for p in re.split(r"[→\-\>\/]", transition) if p.strip()]
            if len(parts) >= 2:
                if before_dialogue:
                    character_states["云琛"] = parts[0]
                else:
                    character_states["云琛"] = parts[-1]
            elif len(parts) == 1:
                character_states["云琛"] = parts[0]
        elif beat.gender_state:
            character_states["云琛"] = beat.gender_state

        return character_states

    def _update_state_from_direction(
        self,
        speaker: str,
        stage_direction: str,
        character_states: Dict[str, str],
    ) -> Dict[str, str]:
        """根据括号内的舞台提示更新状态"""
        if not stage_direction or speaker != "云琛":
            return character_states

        d = str(stage_direction)
        female_cues = ["女装", "少女", "仙女", "柔声", "娇声"]
        male_cues = ["男装", "男人腔", "低沉", "压低嗓音", "男声", "粗声"]

        if any(c in d for c in female_cues):
            character_states["云琛"] = "女装"
        elif any(c in d for c in male_cues):
            character_states["云琛"] = "男装"

        return character_states

    def _resolve_gender_state(
        self, speaker: str, character_states: Dict[str, str]
    ) -> str:
        """确定 speaker 当前的性别/状态"""
        if speaker == "云琛":
            return character_states.get("云琛", "男装")
        return ""

    # ------------------------------------------------------------------
    # 音色 / 语速 / 情绪解析
    # ------------------------------------------------------------------
    def _resolve_voice_role(self, speaker: str, gender_state: str) -> str:
        """把 speaker + 状态映射到 voice_cast 中的角色键"""
        if speaker == "云琛":
            return f"云琛-{gender_state}"
        return speaker

    def _resolve_pace(self, beat_pace: str, stage_direction: str) -> str:
        """结合 beat 节奏和舞台提示决定对白语速档位"""
        d = str(stage_direction).lower()
        if any(k in d for k in ["喊", "急", "快", "急促", "爆发"]):
            return "快"
        if any(k in d for k in ["慢", "低声", "小声", "自言自语", "缓缓"]):
            return "慢"
        if beat_pace in self.PACE_CHARS_PER_SEC:
            return beat_pace
        return "正常"

    @staticmethod
    def _resolve_emotion(beat_emotion: str, stage_direction: str) -> str:
        """从舞台提示提取情绪，回退到 beat 情绪"""
        d = str(stage_direction)
        emotion_map = {
            "焦虑": "焦虑",
            "紧张": "紧张",
            "愤怒": "愤怒",
            "开心": "开心",
            "悲伤": "悲伤",
            "压低嗓音": "阴沉",
            "低声": "低沉",
            "喊": "激动",
            "急": "焦急",
        }
        for cue, emotion in emotion_map.items():
            if cue in d:
                return emotion
        return beat_emotion or ""

    # ------------------------------------------------------------------
    # voice_cast 构建
    # ------------------------------------------------------------------
    def _build_voice_cast(self) -> Dict[str, Any]:
        """合并默认音色映射和用户覆盖配置"""
        cast = dict(self.DEFAULT_VOICE_CAST)

        # 用户覆盖：可以是 {speaker: voice_id} 或完整 dict
        for key, value in self.voice_cast_override.items():
            if key in cast and isinstance(value, str):
                cast[key]["voice_id"] = value
            elif isinstance(value, dict):
                cast[key] = {**cast.get(key, {}), **value}
            else:
                cast[key] = {"role": key, "voice_id": value, "description": "用户自定义"}

        return cast

    def get_voice_id(self, target_voice_role: str) -> str:
        """根据 target_voice_role 返回实际 TTS voice_id"""
        role_cfg = self.voice_cast.get(target_voice_role) or self.voice_cast.get("default")
        if isinstance(role_cfg, dict):
            return role_cfg.get("voice_id", self.DEFAULT_VOICE_CAST["default"]["voice_id"])
        return role_cfg or self.DEFAULT_VOICE_CAST["default"]["voice_id"]
