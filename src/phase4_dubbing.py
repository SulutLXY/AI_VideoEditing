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
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models import Shot
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

    def run(self, shots: List[Shot], decisions: List[Any]) -> Dict[str, Any]:
        """执行配音配乐合成"""
        logger.info("=" * 60)
        logger.info("Phase 4: 配音配乐合成")
        logger.info("=" * 60)

        if not decisions:
            logger.warning("Phase 4: 剪辑决策为空，跳过配音")
            return {}

        # 建立 shot_id -> Shot 映射
        shot_map = {s.shot_id: s for s in shots}

        # 构建 dubber 需要的时间线
        timeline = self._build_timeline(decisions, shot_map)
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

    def _build_timeline(self, decisions: List[Any], shot_map: Dict[str, Shot]) -> List[Dict[str, Any]]:
        """把 Phase 3 决策 + Shot 信息转成 dubber 时间线"""
        timeline = []
        current_time = 0.0

        for d in decisions:
            # 兼容 dataclass 和 dict
            if hasattr(d, "to_dict"):
                d = d.to_dict()
            elif not isinstance(d, dict):
                d = d.__dict__

            shot_id = d.get("shot_id", "")
            shot = shot_map.get(shot_id)

            # 提取台词和情绪
            dialogue = ""
            emotion = ""
            speaker = ""
            if shot:
                dialogue = shot.dialogue or shot.asr_text or ""
                emotion = shot.emotion or ""
                speaker = ", ".join(shot.characters) if shot.characters else ""

            # 优先用决策里可能带的路径，否则回退到 shot.source_path
            clip_path = d.get("clip_path") or d.get("source_clip") or d.get("video_path") or ""
            if not clip_path and shot:
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
                "narration": d.get("narration", ""),
                "emotion": emotion,
                "speaker": speaker,
                "speed": speed_str,
                "speed_mult": speed_mult,
            }
            timeline.append(item)

        return timeline

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
