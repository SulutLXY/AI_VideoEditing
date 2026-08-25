"""
Phase 4 扩展：配音配乐合成

从 ai-voice-dubber 合并的音频能力，负责：
- 根据 Phase 3 剪辑决策时间线，为每段镜头合成配音
- 根据情绪/氛围匹配或生成 BGM
- 将语音、BGM、视频合成为最终成片

输入：
- config: 全局配置（需包含 audio/dubber/tts/music 段）
- shots: Phase 1/2 输出的 Shot 列表
- decisions: Phase 3 输出的 EditDecision 列表

输出：
- final_with_dubbing.mp4
- mixed_audio.wav
- dub_info.json
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models import Shot, ScriptBeat, DialogueEntry
from src.utils import logger, ensure_dir, tc_to_sec, sec_to_tc, load_json
from src.audio import VoiceManager, VoiceEngine, MusicEngine, Dubber


class Phase4Dubbing:
    """Phase 4 配音配乐合成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get("paths", {}).get("output", "./output")
        self.audio_cfg = config.get("audio", {})
        self.dubber_cfg = config.get("dubber", {})
        self.max_speed = float(self.audio_cfg.get("max_dialogue_speed", 2.0))
        ensure_dir(self.output_dir)

    def run(
        self,
        shots: List[Shot],
        decisions: List[Any],
        script_beats: Optional[List[ScriptBeat]] = None,
    ) -> Dict[str, Any]:
        """执行配音配乐合成"""
        logger.info("=" * 60)
        logger.info("Phase 4: 配音配乐合成")
        logger.info("=" * 60)

        if not decisions:
            logger.warning("Phase 4: 剪辑决策为空，跳过配音")
            return {}

        # 建立 shot_id -> Shot 映射
        shot_map = {s.shot_id: s for s in shots}

        # 加载剧本情节点（含 dialogue_entries）。
        # 注意：文件中的分析结果（dialogue_entries）优先于 main.py 传入的 bare script_beats。
        file_beats = self._load_script_beats()
        script_beats = file_beats if file_beats else (script_beats or [])
        beat_map = {b.beat_id: b for b in script_beats}
        voice_cast = self._load_voice_cast()

        # 构建视频时间线（只含视频信息，不再按镜头分配整段 key_dialogue）
        timeline = self._build_timeline(decisions, shot_map, beat_map)
        if not timeline:
            logger.warning("Phase 4: 没有可配音的有效段落，跳过")
            return {}

        # 基于 dialogue_entries 构建精确对白时间轴
        dialogue_segments = self._build_dialogue_segments(timeline, beat_map, voice_cast)
        if not dialogue_segments:
            logger.warning("Phase 4: 未生成任何对白段落，将只输出 BGM/视频")

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
            dialogue_segments=dialogue_segments,
        )

        logger.info("Phase 4 完成")
        if result.get("video"):
            logger.info(f"成片: {result['video']}")
        if result.get("audio"):
            logger.info(f"混合音频: {result['audio']}")

        return result

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load_script_beats(self) -> List[ScriptBeat]:
        """从 script_beats_analysis.json 加载剧本情节点"""
        path = os.path.join(self.output_dir, "script_beats_analysis.json")
        if not os.path.exists(path):
            logger.warning(f"未找到剧本分析文件: {path}")
            return []
        try:
            data = load_json(path)
            beats = []
            for b in data.get("beats", []):
                beat = ScriptBeat(
                    act=b.get("act", ""),
                    scene=b.get("scene", ""),
                    beat_id=b.get("beat_id", ""),
                    location=b.get("location", ""),
                    time=b.get("time", ""),
                    content=b.get("content", ""),
                    emotion=b.get("emotion", ""),
                    key_actions=b.get("key_actions", []),
                    key_dialogue=b.get("key_dialogue", ""),
                    estimated_duration=float(b.get("estimated_duration", 0.0) or 0.0),
                    pace=b.get("pace", "正常"),
                    emotion_intensity=float(b.get("emotion_intensity", 0.0) or 0.0),
                    priority=int(b.get("priority", 3) or 3),
                    required_shots_count=int(b.get("required_shots_count", 1) or 1),
                )
                beat.dialogue_entries = [
                    DialogueEntry.from_dict(e) for e in b.get("dialogue_entries", [])
                ]
                beats.append(beat)
            return beats
        except Exception as e:
            logger.error(f"加载剧本情节点失败: {e}")
            return []

    def _load_voice_cast(self) -> Dict[str, Any]:
        """加载 voice_cast 映射，优先 dialogue_plan.json，其次 config"""
        path = os.path.join(self.output_dir, "dialogue_plan.json")
        if os.path.exists(path):
            try:
                data = load_json(path)
                return data.get("voice_cast", {})
            except Exception as e:
                logger.warning(f"加载 dialogue_plan.json 失败: {e}")
        return self.audio_cfg.get("voice_cast", {})

    # ------------------------------------------------------------------
    # 视频时间线构建
    # ------------------------------------------------------------------
    def _build_timeline(
        self,
        decisions: List[Any],
        shot_map: Dict[str, Shot],
        beat_map: Dict[str, ScriptBeat],
    ) -> List[Dict[str, Any]]:
        """把 Phase 3 决策转成纯视频时间线（不含对白分配）"""
        decision_dicts = []
        for d in decisions:
            if hasattr(d, "to_dict"):
                decision_dicts.append(d.to_dict())
            elif isinstance(d, dict):
                decision_dicts.append(d)
            else:
                decision_dicts.append(d.__dict__)

        logger.info(f"[Phase4] 构建视频时间线，共 {len(decision_dicts)} 个决策")

        timeline = []
        current_time = 0.0

        for d in decision_dicts:
            shot_id = d.get("shot_id", "")
            shot = shot_map.get(shot_id)

            beat_id = ""
            if shot and shot.script_anchor:
                beat_id = shot.script_anchor.get("beat", "")
            elif d.get("beat_id"):
                beat_id = d.get("beat_id")

            clip_path = d.get("clip_path") or d.get("source_clip") or d.get("video_path") or ""
            if not clip_path and shot:
                split_path = None
                if shot.cv_metadata:
                    split_path = shot.cv_metadata.get("shot_config", {}).get("split_clip_path")
                clip_path = split_path or shot.source_path
                if split_path and not os.path.exists(split_path):
                    logger.warning(f"[Phase4] 切分片段不存在，回退到原始素材: {shot.shot_id}")
                    clip_path = shot.source_path

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
                "beat_id": beat_id,
                "speed": speed_str,
                "speed_mult": speed_mult,
            }
            timeline.append(item)

        return timeline

    # ------------------------------------------------------------------
    # 对白时间轴构建
    # ------------------------------------------------------------------
    def _build_dialogue_segments(
        self,
        timeline: List[Dict[str, Any]],
        beat_map: Dict[str, ScriptBeat],
        voice_cast: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        将每个 beat 的 dialogue_entries 映射到绝对时间轴，并计算所需语速。

        策略：
        - beat 内对白总时长 D，视频时长 B。
        - 若 D <= B：按 1x 顺序播放，剩余时间作为间隙。
        - 若 D > B：需要加速，目标语速 s = clamp(D / B, 1.0, max_speed)。
          若 D / B > max_speed，则音频会溢出到下一个 beat（允许）。
        """
        # 计算每个 beat 在成片时间轴上的起止时间
        beat_time_ranges: Dict[str, Dict[str, float]] = {}
        for item in timeline:
            bid = item.get("beat_id", "")
            if not bid or bid == "UNMATCHED":
                continue
            if bid not in beat_time_ranges:
                beat_time_ranges[bid] = {"start": item["start_time"], "end": item["end_time"]}
            else:
                beat_time_ranges[bid]["start"] = min(beat_time_ranges[bid]["start"], item["start_time"])
                beat_time_ranges[bid]["end"] = max(beat_time_ranges[bid]["end"], item["end_time"])

        segments = []
        for beat_id, time_range in sorted(beat_time_ranges.items(), key=lambda x: x[1]["start"]):
            beat = beat_map.get(beat_id)
            if not beat:
                continue
            entries = getattr(beat, "dialogue_entries", []) or []
            if not entries:
                continue

            beat_start = time_range["start"]
            beat_end = time_range["end"]
            beat_duration = beat_end - beat_start
            if beat_duration <= 0:
                continue

            total_dialogue_1x = sum(e.estimated_duration for e in entries)
            if total_dialogue_1x <= 0:
                continue

            # 计算 beat 内统一目标语速
            required_speed = total_dialogue_1x / beat_duration
            if required_speed <= 1.0:
                target_speed = 1.0
            elif required_speed <= self.max_speed:
                target_speed = required_speed
            else:
                target_speed = self.max_speed

            overflow = required_speed > self.max_speed
            if overflow:
                logger.info(
                    f"[Phase4] beat {beat_id} 对白 {total_dialogue_1x:.1f}s 超出视频 "
                    f"{beat_duration:.1f}s，将以 {self.max_speed}x 语速溢出"
                )

            # Edge TTS rate 字符串
            if abs(target_speed - 1.0) < 0.05:
                rate_str = "+0%"
            else:
                rate_pct = int(round((target_speed - 1.0) * 100))
                rate_str = f"+{rate_pct}%"

            cursor = beat_start
            for idx, entry in enumerate(entries):
                if not entry.text:
                    continue

                # 本条对白的起始时间
                entry_start = cursor
                actual_audio_duration = entry.estimated_duration / target_speed if target_speed > 0 else entry.estimated_duration

                voice_id = self._resolve_voice_id(entry.target_voice_role, voice_cast)

                seg = {
                    "id": f"{beat_id}_d{idx:02d}",
                    "text": entry.text,
                    "speaker": entry.speaker,
                    "emotion": entry.emotion or beat.emotion,
                    "start": round(entry_start, 3),
                    "end": round(entry_start + actual_audio_duration, 3),
                    "beat_id": beat_id,
                    "beat_start": beat_start,
                    "beat_end": beat_end,
                    "voice_id": voice_id,
                    "rate": rate_str,
                    "target_speed": round(target_speed, 2),
                    "estimated_duration_1x": entry.estimated_duration,
                    "overflow": overflow,
                }
                segments.append(seg)

                cursor = entry_start + actual_audio_duration

        logger.info(f"[Phase4] 生成 {len(segments)} 条对白段落")
        for seg in segments:
            overflow_note = " [溢出]" if seg.get("overflow") else ""
            logger.info(
                f"[Phase4]   {seg['id']}: t={seg['start']:.2f}s~{seg['end']:.2f}s, "
                f"speed={seg['target_speed']:.2f}x, voice={seg['voice_id']}, "
                f"text={seg['text'][:24]!r}{overflow_note}"
            )

        return segments

    def _resolve_voice_id(self, target_voice_role: str, voice_cast: Dict[str, Any]) -> str:
        """把 target_voice_role 解析成 Edge TTS voice_id"""
        if not target_voice_role:
            return ""

        # 1) 先查 voice_cast
        cfg = voice_cast.get(target_voice_role)
        if isinstance(cfg, dict):
            raw = cfg.get("voice_id", "")
        elif isinstance(cfg, str):
            raw = cfg
        else:
            raw = ""

        # 2) 角色默认兜底
        if not raw:
            defaults = {
                "云琛-男装": "zh-CN-YunxiNeural",
                "云琛-女装": "zh-CN-XiaoxiaoNeural",
                "小六": "zh-CN-YunjianNeural",
            }
            raw = defaults.get(target_voice_role, "")

        # 3) 补全 edge_ 前缀
        if raw and not raw.startswith("edge_"):
            raw = f"edge_{raw}"
        return raw

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
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
