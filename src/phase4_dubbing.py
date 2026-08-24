"""
Phase 4 扩展：配音配乐合成

从 ai-voice-dubber 合并的音频能力，负责：
- 根据 Phase 3 剪辑决策时间线，为每段镜头合成配音
- 根据情绪/氛围匹配或生成 BGM
- 将语音、BGM、视频合成为最终成片

输入：
- config: 全局配置（需包含 audio/dubber/tts/music 段）
- shots: Phase 1 输出的 Shot 列表
- decisions: Phase 3 输出的 EditDecision 列表

输出：
- final_with_dubbing.mp4
- mixed_audio.wav
- dub_info.json
"""
import os
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

from src.models import Shot, ScriptBeat
from src.utils import logger, ensure_dir, tc_to_sec, sec_to_tc
from src.audio import VoiceManager, VoiceEngine, MusicEngine, Dubber


class Phase4Dubbing:
    """Phase 4 配音配乐合成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("paths", {}).get("output", "./output")
        self.audio_cfg = config.get("audio", {})
        self.dubber_cfg = config.get("dubber", {})
        ensure_dir(self.output_dir)

    def run(
        self,
        shots: List[Shot],
        decisions: List[Any],
        script_beats: Optional[List[ScriptBeat]] = None,
    ) -> Dict[str, Any]:
        """执行配音配乐合成

        参数:
            shots: Phase 1/2 输出的镜头列表
            decisions: Phase 3 剪辑决策
            script_beats: 剧本情节点列表，用于生成剧本驱动的配音文本
        """
        logger.info("=" * 60)
        logger.info("Phase 4: 配音配乐合成")
        logger.info("=" * 60)

        if not decisions:
            logger.warning("Phase 4: 剪辑决策为空，跳过配音")
            return {}

        # 建立 shot_id -> Shot 映射
        shot_map = {s.shot_id: s for s in shots}
        beat_map = {b.beat_id: b for b in (script_beats or [])}

        # 构建 dubber 需要的时间线
        timeline = self._build_timeline(decisions, shot_map, beat_map)
        if not timeline:
            logger.warning("Phase 4: 没有可配音的有效段落，跳过")
            return {}

        # 初始化音频引擎
        voice_manager = VoiceManager(self.config)
        voice_engine = VoiceEngine(self.config)
        music_engine = MusicEngine(self.config)
        dubber = Dubber(self.config)

        # 音色选择
        default_voice_id = self.audio_cfg.get("default_voice_id") or self.dubber_cfg.get("voice_id")
        if default_voice_id:
            voice = voice_manager.get_voice(default_voice_id) or voice_manager.get_voice_by_name(default_voice_id)
            if not voice:
                logger.warning(f"指定音色 {default_voice_id} 未找到，使用默认 Edge TTS 音色")
                default_voice_id = None

        # BGM 模式
        bgm_mode = self.audio_cfg.get("bgm_mode", "match")
        keep_original = self.audio_cfg.get("keep_original_audio", False)

        # 执行合成
        result = dubber.dub_from_timeline(
            timeline=timeline,
            voice_engine=voice_engine,
            music_engine=music_engine,
            voice_manager=voice_manager,
            output_dir=self.output_dir,
            bgm_mode=bgm_mode,
            default_voice_id=default_voice_id,
            keep_original_audio=keep_original,
        )

        logger.info("Phase 4 完成")
        if result.get("video"):
            logger.info(f"成片: {result['video']}")
        if result.get("audio"):
            logger.info(f"混合音频: {result['audio']}")

        return result

    def _build_timeline(
        self,
        decisions: List[Any],
        shot_map: Dict[str, Shot],
        beat_map: Dict[str, ScriptBeat],
    ) -> List[Dict[str, Any]]:
        """把 Phase 3 决策 + Shot 信息转成 dubber 时间线

        配音文本按 beat 分配，每个 beat 的台词只出现一次，避免同一 beat 内多镜头重复配音。
        """
        # 统一 decisions 为 dict
        decision_dicts = []
        for d in decisions:
            if hasattr(d, "to_dict"):
                decision_dicts.append(d.to_dict())
            elif isinstance(d, dict):
                decision_dicts.append(d)
            else:
                decision_dicts.append(d.__dict__)

        logger.info(f"[Phase4] 构建配音时间线，共 {len(decision_dicts)} 个决策")

        # 按 beat 分组，并预分配每句台词给最合适的镜头
        beat_dialogue_assignments = self._assign_dialogues_to_beats(
            decision_dicts, shot_map, beat_map
        )
        logger.info(f"[Phase4] 对话分配结果: {sum(1 for v in beat_dialogue_assignments.values() if v)} 个 primary 镜头")
        for (bid, sid), is_primary in beat_dialogue_assignments.items():
            if is_primary:
                logger.info(f"[Phase4]   primary dialogue -> beat={bid}, shot={sid}")

        timeline = []
        current_time = 0.0

        for d in decision_dicts:
            shot_id = d.get("shot_id", "")
            shot = shot_map.get(shot_id)

            # 获取对应 beat
            beat_id = ""
            beat = None
            if shot and shot.script_anchor:
                beat_id = shot.script_anchor.get("beat", "")
                beat = beat_map.get(beat_id)
            elif d.get("beat_id"):
                beat_id = d.get("beat_id")
                beat = beat_map.get(beat_id)

            # 生成配音文本：只有被指定为对话镜头的才生成
            is_primary = beat_dialogue_assignments.get((beat_id, shot_id), False)
            dialogue, text_source = self._generate_dialogue_for_decision(shot, beat, is_primary)
            logger.info(
                f"[Phase4] timeline item shot={shot_id}, beat={beat_id}, "
                f"primary={is_primary}, source={text_source}, dialogue={dialogue[:30]!r}"
            )

            # 提取台词和情绪
            emotion = ""
            speaker = ""
            if shot:
                emotion = shot.emotion or (beat.emotion if beat else "")
                speaker = ", ".join(shot.characters) if shot.characters else ""

            # 优先使用 Phase1 切分好的独立片段；否则回退到原始素材
            clip_path = d.get("clip_path") or d.get("source_clip") or d.get("video_path") or ""
            if not clip_path and shot:
                split_path = None
                if shot.cv_metadata:
                    split_path = shot.cv_metadata.get("shot_config", {}).get("split_clip_path")
                clip_path = split_path or shot.source_path
                if split_path and not os.path.exists(split_path):
                    logger.warning(
                        f"[Phase4] 切分片段不存在，回退到原始素材: {shot.shot_id}"
                    )
                    clip_path = shot.source_path

            # 计算实际入点/出点/速度，并按速度折算成片时长
            fps = getattr(shot, "fps", 24.0) if shot else 24.0
            tc_in = d.get("tc_in") or (shot.tc_in if shot else "00:00:00:00")
            tc_out = d.get("tc_out") or (shot.tc_out if shot else sec_to_tc(getattr(shot, "duration_sec", 0.0), fps))
            speed_str = d.get("speed", "1x")
            speed_mult = self._parse_speed(speed_str)

            try:
                src_in_sec = tc_to_sec(tc_in, fps)
                src_out_sec = tc_to_sec(tc_out, fps)
                src_duration = max(0.0, src_out_sec - src_in_sec)
            except Exception:
                src_in_sec = 0.0
                src_out_sec = src_in_sec + (getattr(shot, "duration_sec", 0.0) if shot else 0.0)
                src_duration = max(0.0, src_out_sec - src_in_sec)

            duration_sec = src_duration / speed_mult if speed_mult > 0 else src_duration

            start_time = current_time
            end_time = current_time + duration_sec
            current_time = end_time

            item = {
                "sequence": d.get("sequence", len(timeline) + 1),
                "shot_id": shot_id,
                "clip_path": clip_path,
                "source_clip": clip_path,
                "video_path": clip_path,
                "tc_in": tc_in,
                "tc_out": tc_out,
                "src_in_sec": src_in_sec,
                "src_out_sec": src_out_sec,
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "src_duration": src_duration,
                "dialogue": dialogue,
                "dialogue_source": text_source,
                "narration": d.get("narration", ""),
                "emotion": emotion,
                "speaker": speaker,
                "speed": speed_str,
                "speed_mult": speed_mult,
            }
            timeline.append(item)

        return timeline

    def _assign_dialogues_to_beats(
        self,
        decision_dicts: List[Dict],
        shot_map: Dict[str, Shot],
        beat_map: Dict[str, ScriptBeat],
    ) -> Dict[Tuple[str, str], bool]:
        """预分配每个 beat 的 key_dialogue 给最合适的镜头，返回 {(beat_id, shot_id): is_primary}

        策略：
        - 若 beat 没有 key_dialogue，所有镜头都不是对话镜头。
        - 若 beat 只有 1 个镜头，该镜头承担全部台词。
        - 若 beat 有多个镜头，优先选 function='对话镜头' 的镜头；
          否则选 action/asr 与 beat.key_dialogue 重叠度最高的。
        """
        from collections import defaultdict

        assignments: Dict[tuple, bool] = {}
        beat_groups: Dict[str, List[Dict]] = defaultdict(list)

        for d in decision_dicts:
            shot_id = d.get("shot_id", "")
            shot = shot_map.get(shot_id)
            beat_id = ""
            if shot and shot.script_anchor:
                beat_id = shot.script_anchor.get("beat", "")
            elif d.get("beat_id"):
                beat_id = d.get("beat_id")
            if beat_id:
                beat_groups[beat_id].append(d)

        def _simple_tokens(text: str) -> Set[str]:
            """简单中文分词：按标点切分 + 2-gram"""
            if not text:
                return set()
            delimiters = set("，、。！？；：""''（）(),.!?;:\"'() ")
            text = str(text).lower()
            words: Set[str] = set()
            current = ""
            for ch in text:
                if ch in delimiters:
                    if len(current) >= 2:
                        words.add(current)
                    current = ""
                else:
                    current += ch
            if len(current) >= 2:
                words.add(current)
            for i in range(len(text) - 1):
                bg = text[i:i + 2]
                if bg[0] not in delimiters and bg[1] not in delimiters:
                    words.add(bg)
            return words

        for beat_id, group in beat_groups.items():
            beat = beat_map.get(beat_id)
            if not beat or not (beat.key_dialogue or beat.content):
                continue

            if len(group) == 1:
                assignments[(beat_id, group[0].get("shot_id"))] = True
                continue

            # 优先选 function='对话镜头'
            dialogue_shots = []
            for d in group:
                shot = shot_map.get(d.get("shot_id"))
                if shot and shot.script_anchor:
                    if shot.script_anchor.get("function") == "对话镜头":
                        dialogue_shots.append(d)

            if len(dialogue_shots) == 1:
                assignments[(beat_id, dialogue_shots[0].get("shot_id"))] = True
                continue
            if len(dialogue_shots) > 1:
                group = dialogue_shots

            # 按与 key_dialogue 的 token 重叠度排序
            beat_text = beat.key_dialogue or beat.content or ""
            beat_tokens = _simple_tokens(beat_text)

            def _dialogue_fit(d: Dict) -> float:
                shot = shot_map.get(d.get("shot_id"))
                if not shot or not beat_tokens:
                    return 0.0
                shot_text = " ".join(filter(None, [
                    shot.action or "",
                    getattr(shot, "action_details", "") or "",
                    shot.asr_text or "",
                    shot.dialogue or "",
                ]))
                shot_tokens = _simple_tokens(shot_text)
                if not shot_tokens:
                    return 0.0
                overlap = shot_tokens & beat_tokens
                recall = len(overlap) / len(beat_tokens)
                precision = len(overlap) / len(shot_tokens)
                if recall + precision <= 0:
                    return 0.0
                return 2 * recall * precision / (recall + precision)

            group_sorted = sorted(group, key=_dialogue_fit, reverse=True)
            assignments[(beat_id, group_sorted[0].get("shot_id"))] = True

        return assignments

    def _generate_dialogue_for_decision(
        self,
        shot: Optional[Shot],
        beat: Optional[ScriptBeat],
        is_primary_dialogue: bool,
    ) -> tuple:
        """为当前决策生成配音文本，避免重复原视频 ASR 对白。

        只有被指定为 primary 的对话镜头才生成 key_dialogue，其余镜头不生成对白。
        """
        if not beat:
            # 没有对应 beat 的镜头：不生成对白（避免 ASR 噪音）
            return "", "无对白"

        # 只有主对话镜头生成 key_dialogue
        if is_primary_dialogue:
            if beat.key_dialogue and beat.key_dialogue.strip():
                return beat.key_dialogue.strip(), "剧本对白"
            if beat.content and beat.content.strip():
                return self._summarize_to_line(beat.content, max_chars=40), "剧本旁白"

        # 非主镜头：不生成对白
        return "", "无对白"

    @staticmethod
    def _summarize_to_line(content: str, max_chars: int = 40) -> str:
        """把 beat.content 提炼成一句适合配音的短句"""
        if not content:
            return ""
        # 去掉特殊符号和多余空格
        text = content.replace("△", "").replace("□", "").replace("\n", " ").strip()
        # 截取前 max_chars 字符，尽量在句末结束
        if len(text) <= max_chars:
            return text
        # 在 max_chars 前找最后一个标点
        cut = max_chars
        for i in range(max_chars - 1, max_chars // 2, -1):
            if text[i] in "，。！？；,!?;":
                cut = i + 1
                break
        return text[:cut].strip()


    @staticmethod
    def _parse_speed(speed_str: str) -> float:
        """解析速度字符串为倍率：2x/200% -> 2.0，50% -> 0.5"""
        if not speed_str:
            return 1.0
        s = str(speed_str).strip().lower()
        if s == "删除":
            return 0.0
        if "%" in s:
            try:
                return float(s.replace("%", "")) / 100.0
            except Exception:
                return 1.0
        if "x" in s:
            try:
                return float(s.replace("x", ""))
            except Exception:
                return 1.0
        try:
            return float(s)
        except Exception:
            return 1.0

    @staticmethod
    def _estimate_duration(tc_in: str, tc_out: str) -> float:
        """粗略从时间码估算时长"""
        try:
            from src.utils import tc_to_sec
            return tc_to_sec(tc_out) - tc_to_sec(tc_in)
        except Exception:
            return 0.0
