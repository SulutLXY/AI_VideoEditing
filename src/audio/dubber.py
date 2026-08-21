"""
Dubber: 配音合成引擎（从 ai-voice-dubber 合并并适配）

适配目标：不再直接读取 script.txt，而是接收 AI_VideoEditing Phase 3 的输出：
- timeline: 剪辑决策时间线（含每段的 emotion / dialogue / source_clip 等）
- script_beats: 剧本情节点（用于提取台词）
- video_segments: 实际视频片段路径列表

输出：
- 带配音 + BGM 的最终成片
- 配音干声、BGM、混合音频中间文件
"""
import os
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.utils import logger, ensure_dir
from .voice_manager import VoiceProfile


class Dubber:
    """配音合成器"""

    def __init__(self, config: dict):
        self.config = config
        self.dubber_cfg = config.get("dubber", {})
        self.paths = config.get("paths", {})
        self.temp_dir = Path(self.paths.get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 音频合成（FFmpeg 直接处理）
    # ------------------------------------------------------------------
    def mix_audio(self, voice_files: List[str],
                  bgm_file: Optional[str] = None,
                  output_path: str = "output/mixed_audio.wav",
                  voice_timing: Optional[List[Tuple[float, float]]] = None) -> str:
        """
        混合语音和BGM（使用 FFmpeg）

        Args:
            voice_files: 语音文件列表
            bgm_file: BGM文件路径
            output_path: 输出路径
            voice_timing: 每段语音的开始时间 [(start_sec, end_sec), ...]

        Returns:
            输出文件路径
        """
        # 先按时间线合并语音为一个文件
        voice_merged = self._merge_voice_files(voice_files, voice_timing)

        if bgm_file and Path(bgm_file).exists():
            self._mix_with_ffmpeg(voice_merged, bgm_file, output_path)
        else:
            import shutil
            shutil.copy2(voice_merged, output_path)

        # 清理临时文件
        if Path(voice_merged).exists() and voice_merged != output_path:
            os.remove(voice_merged)

        logger.info(f"[Dubber] 音频合成完成: {output_path}")
        return output_path

    def _merge_voice_files(self, voice_files: List[str],
                           timing: Optional[List[Tuple[float, float]]] = None) -> str:
        """合并语音文件（FFmpeg concat 或按时间线 pad）"""
        if not voice_files:
            silent = str(self.temp_dir / "silent.wav")
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", "1", "-acodec", "pcm_s16le", silent
            ], capture_output=True)
            return silent

        # 单段直接返回
        if len(voice_files) == 1:
            return voice_files[0]

        # 如果提供了时间线，使用 adelay + amix 拼接（保留空白间隔）
        if timing and len(timing) == len(voice_files):
            return self._merge_with_timing(voice_files, timing)

        # 否则直接首尾相接
        merged = str(self.temp_dir / "merged_voice.wav")
        concat_list = str(self.temp_dir / "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for vf in voice_files:
                abs_path = str(Path(vf).resolve()).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            merged
        ], capture_output=True, check=False)

        return merged

    def _merge_with_timing(self, voice_files: List[str],
                           timing: List[Tuple[float, float]]) -> str:
        """按时间线合并语音，保留间隔"""
        inputs = []
        filters = []
        for i, (vf, (start, _)) in enumerate(zip(voice_files, timing)):
            inputs.extend(["-i", vf])
            filters.append(f"[{i}:a]adelay=delays={int(start * 1000)}|{int(start * 1000)}[a{i}]")

        mix_inputs = "".join(f"[a{i}]" for i in range(len(voice_files)))
        total_end = max(t[1] for t in timing)
        filters.append(f"{mix_inputs}amix=inputs={len(voice_files)}:duration=longest:dropout_transition=2[aout]")

        output = str(self.temp_dir / "timed_voice.wav")
        cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(filters), "-map", "[aout]", output]
        subprocess.run(cmd, capture_output=True, check=False)

        if not Path(output).exists() or self._get_audio_duration(output) < total_end * 0.5:
            # fallback 到首尾相接
            logger.warning("[Dubber] 时间线合并失败，回退到首尾相接")
            return self._merge_voice_files(voice_files, None)

        return output

    def _mix_with_ffmpeg(self, voice_file: str, bgm_file: str, output_path: str):
        """使用 FFmpeg 混合语音和BGM"""
        voice_duration = self._get_audio_duration(voice_file)
        bgm_duration = self._get_audio_duration(bgm_file)

        bgm_db = self.dubber_cfg.get("bgm_volume", -20)
        voice_db = self.dubber_cfg.get("voice_volume", 0)
        fade_in = self.dubber_cfg.get("fade_in", 1.0)
        fade_out = self.dubber_cfg.get("fade_out", 2.0)

        # BGM 处理：音量调整 + 淡入淡出 + 循环/截断
        bgm_processed = str(self.temp_dir / "bgm_processed.wav")
        if bgm_duration < voice_duration:
            loop_count = int(voice_duration / bgm_duration) + 1
            subprocess.run([
                "ffmpeg", "-y",
                "-stream_loop", str(loop_count),
                "-i", bgm_file,
                "-t", str(voice_duration),
                "-af", f"volume={bgm_db}dB,fade=t=in:st=0:d={fade_in},fade=t=out:st={voice_duration-fade_out}:d={fade_out}",
                "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                bgm_processed
            ], capture_output=True, check=False)
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", bgm_file,
                "-t", str(voice_duration),
                "-af", f"volume={bgm_db}dB,fade=t=in:st=0:d={fade_in},fade=t=out:st={voice_duration-fade_out}:d={fade_out}",
                "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                bgm_processed
            ], capture_output=True, check=False)

        # 混合语音和BGM
        os.makedirs(Path(output_path).parent, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", voice_file,
            "-i", bgm_processed,
            "-filter_complex", f"[0:a]volume={voice_db}dB[v];[v][1:a]amix=inputs=2:duration=longest:dropout_transition=2[aout]",
            "-map", "[aout]",
            "-acodec", "pcm_s16le", "-ar", str(self.dubber_cfg.get("sample_rate", 24000)),
            output_path
        ], capture_output=True, check=False)

        if Path(bgm_processed).exists():
            os.remove(bgm_processed)

    def _get_audio_duration(self, file_path: str) -> float:
        """获取音频时长"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, check=False
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _generate_silent_wav(self, output_path: str, duration: float,
                             sample_rate: int = 24000, channels: int = 1) -> str:
        """生成指定时长的静音 WAV 占位文件"""
        import subprocess
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"anullsrc=r={sample_rate}:cl={'mono' if channels == 1 else 'stereo'}",
            "-t", str(duration),
            "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
            output_path
        ], capture_output=True, check=False)
        return output_path

    # ------------------------------------------------------------------
    # 视频合成
    # ------------------------------------------------------------------
    def dub_video(self, video_path: str, mixed_audio: str,
                  output_path: str = "output/dubbed_video.mp4",
                  keep_original_audio: bool = False,
                  original_volume: float = 0.3) -> str:
        """将配音音频合成到视频中"""
        video_path = Path(video_path)
        mixed_audio = Path(mixed_audio)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")
        if not mixed_audio.exists():
            raise FileNotFoundError(f"音频不存在: {mixed_audio}")

        os.makedirs(output_path.parent, exist_ok=True)

        try:
            if keep_original_audio:
                temp_original = str(self.temp_dir / "original_lower.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-vn", "-af", f"volume={original_volume}",
                    "-acodec", "pcm_s16le", temp_original
                ], capture_output=True, check=False)

                final_audio = str(self.temp_dir / "final_mixed.wav")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", temp_original,
                    "-i", str(mixed_audio),
                    "-filter_complex", "amix=inputs=2:duration=longest",
                    "-acodec", "aac", "-b:a", "192k",
                    final_audio
                ], capture_output=True, check=False)

                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", final_audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest",
                    str(output_path)
                ], capture_output=True, check=False)

                for f in [temp_original, final_audio]:
                    if Path(f).exists():
                        os.remove(f)
            else:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(mixed_audio),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-shortest",
                    str(output_path)
                ], capture_output=True, check=False)

            logger.info(f"[Dubber] 视频合成完成: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"[Dubber] FFmpeg 错误: {e}")
            raise

    # ------------------------------------------------------------------
    # 适配 AI_VideoEditing 的配音流程
    # ------------------------------------------------------------------
    def dub_from_timeline(
        self,
        timeline: List[Dict],
        voice_engine,
        music_engine,
        voice_manager,
        output_dir: str = "output",
        bgm_mode: str = "match",
        default_voice_id: Optional[str] = None,
        keep_original_audio: bool = False,
    ) -> Dict:
        """
        从 Phase 3 剪辑决策时间线生成配音成片

        Args:
            timeline: Phase 3 输出的 timeline 决策项，每项可包含 dialogue / narration / emotion 等字段
            voice_engine: VoiceEngine 实例
            music_engine: MusicEngine 实例
            voice_manager: VoiceManager 实例
            output_dir: 输出目录
            bgm_mode: "match" 本地库匹配 / "generate" AI生成 / "none" 无BGM
            default_voice_id: 默认音色 ID，未提供时使用首个 edge_tts 音色
            keep_original_audio: 是否保留原视频音频并降低音量混合

        Returns:
            {"video": ..., "audio": ..., "voice_files": [...], "bgm": ..., "meta": ...}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_dir(str(output_dir))

        # 情绪 -> 音色映射（Edge TTS 多音色）
        def _emotion_to_voice_id(emotion: str) -> Optional[str]:
            """根据情绪选择 Edge TTS 音色 ID"""
            emotion = (emotion or "").lower()
            mapping = {
                "温馨": "edge_zh-CN-YunxiNeural",
                "平静": "edge_zh-CN-YunxiNeural",
                "专注": "edge_zh-CN-YunxiNeural",
                "好奇": "edge_zh-CN-XiaoxuanNeural",
                "开心": "edge_zh-CN-XiaoxiaoNeural",
                "兴奋": "edge_zh-CN-YunjianNeural",
                "紧张": "edge_zh-CN-YunjianNeural",
                "恐惧": "edge_zh-CN-YunjianNeural",
                "悲伤": "edge_zh-CN-XiaoyiNeural",
                "难过": "edge_zh-CN-XiaoyiNeural",
                "愤怒": "edge_zh-CN-YunjianNeural",
                "冷静": "edge_zh-CN-YunxiNeural",
            }
            for k, v in mapping.items():
                if k in emotion:
                    return v
            return None

        def _choose_voice(segment: dict) -> Optional[VoiceProfile]:
            """为单段文本选择音色：优先按情绪映射，否则 fallback 到 default_voice_id"""
            # 1) 按情绪选音色
            voice_id = _emotion_to_voice_id(segment.get("emotion", ""))
            if voice_id:
                voice = voice_manager.get_voice(voice_id)
                if voice:
                    return voice
            # 2) 按 speaker 名字选（如果有 speaker 且配置中存在对应音色）
            speaker = segment.get("speaker", "")
            if speaker:
                voice = voice_manager.get_voice_by_name(speaker) or voice_manager.get_voice(f"edge_{speaker}")
                if voice:
                    return voice
            # 3) fallback 到 default_voice_id
            if default_voice_id:
                voice = voice_manager.get_voice(default_voice_id)
                if voice:
                    return voice
            # 4) 最终兜底：第一个可用 edge_tts 音色
            voices = voice_manager.list_voices(engine="edge_tts")
            return voices[0] if voices else None

        # 提取有台词的段落
        def _clean_tts_text(text: str) -> str:
            """清洗 TTS 文本：移除 SenseVoice / Whisper 等 ASR 特殊标记"""
            import re
            # 移除 <|zh|>, <|NEUTRAL|>, <|BGM|>, <|Speech|>, <|withitn|> 等标记
            # 使用贪心匹配，兼容多字符标签（如 <|EMO_UNKNOWN|>）
            text = re.sub(r"<\|[A-Za-z0-9_]+\|>", "", text)
            # 合并多余空白
            text = re.sub(r"\s+", " ", text).strip()
            return text

        segments = []
        for i, item in enumerate(timeline):
            text = item.get("dialogue") or item.get("narration") or ""
            text = _clean_tts_text(text)
            if text:
                segments.append({
                    "id": f"seg_{i:03d}",
                    "text": text,
                    "emotion": item.get("emotion", ""),
                    "speaker": item.get("speaker", ""),
                    "start": item.get("start_time", 0.0),
                    "end": item.get("end_time", 0.0),
                })

        logger.info(f"[Dubber] 共 {len(segments)} 段文本待合成")

        # Step 1: 合成语音
        voice_dir = self.temp_dir / "voices"
        voice_dir.mkdir(exist_ok=True)

        voice_files = []
        voice_timing = []
        failed_segments = []
        last_voice = None
        for i, seg in enumerate(segments):
            output_file = str(voice_dir / f"{seg['id']}.wav")
            duration = max(seg["end"] - seg["start"], 1.0)
            voice = _choose_voice(seg)
            if not voice:
                logger.error(f"[Dubber] 未找到可用音色 [{seg['id']}]")
                failed_segments.append({"id": seg["id"], "text": seg["text"], "error": "未找到可用音色"})
                self._generate_silent_wav(output_file, duration)
                voice_files.append(output_file)
                voice_timing.append((seg["start"], seg["end"]))
                continue
            last_voice = voice
            seg["voice_id"] = voice.id
            seg["voice_name"] = voice.name
            try:
                logger.info(f"[Dubber] [{seg['id']}] 情绪'{seg.get('emotion', '')}' -> 音色'{voice.name}'({voice.id})")
                voice_engine.synthesize(seg["text"], voice, output_file, duration=duration)
            except Exception as e:
                logger.error(f"[Dubber] 语音合成失败 [{seg['id']}]: {e}")
                failed_segments.append({"id": seg["id"], "text": seg["text"], "error": str(e), "voice_id": voice.id, "voice_name": voice.name})
                # 如果引擎未生成占位文件，则在此处生成一个静音文件以保证时间线不中断
                if not Path(output_file).exists():
                    self._generate_silent_wav(output_file, duration)
            voice_files.append(output_file)
            voice_timing.append((seg["start"], seg["end"]))
            # 请求间隔，降低 Edge TTS 触发限流/无音频返回的概率
            if i < len(segments) - 1:
                time.sleep(0.5)

        voice = last_voice  # 用于元数据兜底

        # Step 2: 配乐
        bgm_file = None
        total_duration = max((t.get("end_time", t.get("end", 0.0)) for t in timeline), default=0.0)

        if bgm_mode in ("match", "generate") and total_duration > 0:
            emotions = music_engine.analyze_segments_emotion(segments) if segments else [("", "平静")]
            if emotions:
                dominant_emotion = max([e for _, e in emotions], key=lambda x: sum(1 for _, e in emotions if e == x))
            else:
                dominant_emotion = "平静"

            if bgm_mode == "generate":
                try:
                    bgm_file = str(music_engine.generate_music(
                        f"{dominant_emotion}的背景音乐，适合视频配音",
                        int(total_duration) + 5
                    ))
                except Exception as e:
                    logger.warning(f"[Dubber] AI生成BGM失败: {e}，尝试本地库匹配")
                    bgm = music_engine.match_from_library(dominant_emotion, total_duration)
                    if bgm:
                        bgm_file = str(bgm)
            else:
                bgm = music_engine.match_from_library(dominant_emotion, total_duration)
                if bgm:
                    bgm_file = str(bgm)

        # Step 3: 混合音频
        mixed_audio = str(output_dir / "mixed_audio.wav")
        self.mix_audio(voice_files, bgm_file, mixed_audio, voice_timing)

        # Step 4: 合并视频片段（先 concat 成一个完整视频）
        final_video_path = str(output_dir / "final_with_dubbing.mp4")
        merged_video = self._merge_video_segments(timeline)

        if merged_video and Path(merged_video).exists():
            self.dub_video(merged_video, mixed_audio, final_video_path, keep_original_audio)
            result_video = final_video_path
        else:
            logger.warning("[Dubber] 未找到可合并的视频片段，仅输出音频")
            result_video = None

        # 保存元数据
        meta_path = str(output_dir / "dub_info.json")
        dub_info = {
            "audio": mixed_audio,
            "video": result_video,
            "voice_used": voice.name if voice else "",
            "voice_id": voice.id if voice else "",
            "bgm_used": bgm_file,
            "bgm_mode": bgm_mode,
            "segments": [
                {"id": s["id"], "text": s["text"], "speaker": s["speaker"],
                 "emotion": s["emotion"], "start": s["start"], "end": s["end"],
                 "voice_id": s.get("voice_id", ""), "voice_name": s.get("voice_name", "")}
                for s in segments
            ],
            "failed_segments": failed_segments,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(dub_info, f, ensure_ascii=False, indent=2)

        logger.info(f"[Dubber] 配音完成！输出目录: {output_dir}")
        return {
            "video": result_video,
            "audio": mixed_audio,
            "voice_files": voice_files,
            "bgm": bgm_file,
            "meta": meta_path,
            "info": dub_info,
        }

    def _merge_video_segments(self, timeline: List[Dict]) -> Optional[str]:
        """按时间线合并视频片段：对每个片段截取入点/出点并应用变速，再统一分辨率拼接"""
        # 目标输出分辨率（可在 config 中扩展，默认 1080p）
        target_width = self.config.get("video", {}).get("output_width", 1920)
        target_height = self.config.get("video", {}).get("output_height", 1080)
        target_resolution = f"{target_width}x{target_height}"

        valid_items = []
        for item in timeline:
            clip = item.get("clip_path") or item.get("source_clip") or item.get("video_path")
            if clip and Path(clip).exists():
                valid_items.append(item)

        if not valid_items:
            return None

        temp_segments = []
        for idx, item in enumerate(valid_items):
            clip = item["clip_path"]
            tc_in = item.get("src_in_sec", 0.0)
            tc_out = item.get("src_out_sec", 0.0)
            speed_mult = item.get("speed_mult", 1.0)
            if speed_mult is None or speed_mult <= 0:
                speed_mult = 1.0

            # 输出片段路径
            segment_path = str(self.temp_dir / f"video_segment_{idx:03d}.mp4")

            # 截取 + 变速 + 统一分辨率（静音，配音轨后续覆盖）
            # setpts=PTS/speed 调整视频速度；atempo 仅当速度在 0.5~2 范围可用
            filters = [f"setpts=PTS/{speed_mult}", "scale={}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2:black".format(target_resolution, target_width, target_height)]

            # 音频也变速（后续会被配音覆盖，但保持同步）
            audio_filter = None
            if 0.5 <= speed_mult <= 2.0:
                audio_filter = f"atempo={speed_mult}"
            elif 2.0 < speed_mult <= 4.0:
                audio_filter = f"atempo=2.0,atempo={speed_mult / 2.0}"

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(tc_in),
                "-to", str(tc_out),
                "-i", clip,
                "-vf", ",".join(filters),
            ]
            if audio_filter:
                cmd.extend(["-af", audio_filter])
            else:
                cmd.extend(["-an"])
            cmd.extend([
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-ar", "24000", "-ac", "1",
                "-r", "30",
                segment_path,
            ])

            logger.info(f"[Dubber] 处理片段 {idx + 1}/{len(valid_items)}: {Path(clip).name} "
                        f"({tc_in:.2f}s-{tc_out:.2f}s, speed={speed_mult:.2f})")
            result = subprocess.run(cmd, capture_output=True, check=False)
            if not Path(segment_path).exists():
                logger.warning(f"[Dubber] 片段处理失败，跳过: {clip}\n{result.stderr.decode('utf-8', errors='ignore')[:200]}")
                continue
            temp_segments.append(segment_path)

        if not temp_segments:
            return None

        if len(temp_segments) == 1:
            return temp_segments[0]

        # concat demuxer 拼接
        concat_file = str(self.temp_dir / "video_concat_list.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in temp_segments:
                f.write(f"file '{seg.replace(chr(92), '/')}'\n")

        merged = str(self.temp_dir / "merged_video.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            merged,
        ], capture_output=True, check=False)

        if Path(merged).exists():
            return merged
        return None
