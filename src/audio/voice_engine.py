"""
Voice Engine: TTS 推理引擎（从 ai-voice-dubber 合并）
支持：Edge TTS (免安装) / GPT-SoVITS (需额外安装)
"""
import os
import tempfile
import asyncio
from typing import Optional, List
from pathlib import Path

from src.utils import logger
from .voice_manager import VoiceProfile


class VoiceEngine:
    """语音合成引擎"""

    def __init__(self, config: dict):
        self.config = config
        self.tts_config = config.get("tts", {})
        self.default_engine = self.tts_config.get("default_engine", "edge_tts")

    def synthesize(self, text: str, voice: VoiceProfile,
                   output_path: str,
                   rate: Optional[str] = None,
                   volume: Optional[str] = None,
                   duration: Optional[float] = None) -> str:
        """
        合成单条语音

        Args:
            text: 要合成的文本
            voice: 音色配置
            output_path: 输出文件路径
            rate: 语速（如 "+10%" / "-5%"）
            volume: 音量（如 "+0%" / "-10%"）
            duration: 期望音频时长（秒），失败时用于生成静音占位

        Returns:
            输出文件路径
        """
        if voice.engine == "edge_tts":
            return self._synthesize_edge(text, voice, output_path, rate, volume, duration)
        elif voice.engine == "gpt_sovits":
            return self._synthesize_gpt_sovits(text, voice, output_path, duration)
        elif voice.engine == "rvc":
            return self._synthesize_rvc(text, voice, output_path, duration)
        else:
            raise ValueError(f"不支持的引擎: {voice.engine}")

    # ------------------------------------------------------------------
    # Edge TTS (在线，免安装)
    # ------------------------------------------------------------------
    def _synthesize_edge(self, text: str, voice: VoiceProfile,
                         output_path: str,
                         rate: Optional[str] = None,
                         volume: Optional[str] = None,
                         duration: Optional[float] = None) -> str:
        """使用 Edge TTS 合成（带重试、备用音色、文件校验，失败时回退静音占位）"""
        import edge_tts
        import time

        voice_id = voice.edge_voice or self.tts_config.get("edge_tts", {}).get("default_voice", "zh-CN-YunxiNeural")
        rate = rate or self.tts_config.get("edge_tts", {}).get("rate", "+0%")
        volume = volume or self.tts_config.get("edge_tts", {}).get("volume", "+0%")

        # 备用音色：当主音色连续失败时尝试其他稳定音色
        backup_voices = ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural"]
        if voice_id not in backup_voices:
            backup_voices.insert(0, voice_id)
        else:
            backup_voices = [voice_id] + [v for v in backup_voices if v != voice_id]

        request_timeout = self.tts_config.get("edge_tts", {}).get("timeout", 90)
        max_retries = 5
        last_error = None

        def _cleanup_bad_file(path: str):
            """删除可能已损坏的输出文件"""
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        def _is_valid_audio(path: str) -> bool:
            """校验音频文件是否有效（存在且非极小）"""
            if not os.path.exists(path):
                return False
            size = os.path.getsize(path)
            if size < 1024:  # 小于 1KB 视为无效
                return False
            return True

        for voice_idx, current_voice in enumerate(backup_voices):
            for attempt in range(1, max_retries + 1):
                try:
                    _cleanup_bad_file(output_path)
                    communicate = edge_tts.Communicate(
                        text=text,
                        voice=current_voice,
                        rate=rate,
                        volume=volume,
                    )
                    asyncio.run(
                        asyncio.wait_for(communicate.save(output_path), timeout=request_timeout)
                    )
                    if _is_valid_audio(output_path):
                        if current_voice != voice_id:
                            logger.info(f"[VoiceEngine] Edge TTS 使用备用音色 {current_voice} 合成完成: {output_path}")
                        else:
                            logger.info(f"[VoiceEngine] Edge TTS 合成完成: {output_path}")
                        return output_path
                    else:
                        raise RuntimeError("合成文件无效或为空")
                except Exception as e:
                    last_error = e
                    _cleanup_bad_file(output_path)
                    logger.warning(
                        f"[VoiceEngine] Edge TTS 合成失败 (音色 {current_voice}, "
                        f"尝试 {attempt}/{max_retries}): {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(min(2 ** attempt, 16))  # 指数退避，最大 16s

            logger.warning(f"[VoiceEngine] 音色 {current_voice} 连续 {max_retries} 次失败，尝试下一个备用音色")

        # 全部失败后生成静音占位，避免整个配音流程崩溃
        fallback_duration = duration or max(len(text.strip()) * 0.25, 1.0)
        logger.warning(
            f"[VoiceEngine] Edge TTS 所有音色均失败，生成静音占位: "
            f"{output_path} (时长 {fallback_duration:.1f}s)"
        )
        self._generate_silent_wav(output_path, fallback_duration)
        return output_path

    def _generate_silent_wav(self, output_path: str, duration: float,
                             sample_rate: int = 24000, channels: int = 1) -> str:
        """使用 FFmpeg 生成指定时长的静音 WAV"""
        import subprocess
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"anullsrc=r={sample_rate}:cl={'mono' if channels == 1 else 'stereo'}",
            "-t", str(duration),
            "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
            output_path
        ], capture_output=True, check=True)
        return output_path

    # ------------------------------------------------------------------
    # GPT-SoVITS (本地/API)
    # ------------------------------------------------------------------
    def _synthesize_gpt_sovits(self, text: str, voice: VoiceProfile,
                               output_path: str,
                               duration: Optional[float] = None) -> str:
        """使用 GPT-SoVITS 合成（支持声音克隆）"""
        gpt_cfg = self.tts_config.get("gpt_sovits", {})
        api_url = gpt_cfg.get("api_url")

        if api_url:
            return self._synthesize_gpt_sovits_api(text, voice, output_path, api_url, duration)

        if voice.gpt_model and voice.sovits_model:
            return self._synthesize_gpt_sovits_local(text, voice, output_path, duration)

        if voice.ref_audio:
            return self._synthesize_gpt_sovits_local(text, voice, output_path, duration)

        fallback_duration = duration or max(len(text.strip()) * 0.25, 1.0)
        logger.warning(
            f"[VoiceEngine] GPT-SoVITS 配置不完整，生成静音占位 "
            f"({fallback_duration:.1f}s): {output_path}"
        )
        self._generate_silent_wav(output_path, fallback_duration)
        return output_path

    def _synthesize_gpt_sovits_api(self, text: str, voice: VoiceProfile,
                                    output_path: str, api_url: str,
                                    duration: Optional[float] = None) -> str:
        """通过 GPT-SoVITS API 合成（失败时回退到静音占位）"""
        import requests

        payload = {
            "text": text,
            "text_lang": voice.language or "zh",
        }

        if voice.ref_audio:
            payload["ref_audio_path"] = voice.ref_audio
            payload["prompt_text"] = voice.ref_text or ""
            payload["prompt_lang"] = voice.language or "zh"

        try:
            response = requests.post(f"{api_url}/tts", json=payload, timeout=300)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            logger.info(f"[VoiceEngine] GPT-SoVITS API 合成完成: {output_path}")
            return output_path

        except Exception as e:
            fallback_duration = duration or max(len(text.strip()) * 0.25, 1.0)
            logger.warning(
                f"[VoiceEngine] GPT-SoVITS API 失败: {e}，生成静音占位 "
                f"({fallback_duration:.1f}s): {output_path}"
            )
            self._generate_silent_wav(output_path, fallback_duration)
            return output_path

    def _synthesize_gpt_sovits_local(self, text: str, voice: VoiceProfile,
                                      output_path: str,
                                      duration: Optional[float] = None) -> str:
        """本地加载 GPT-SoVITS 模型推理（需完整安装 GPT-SoVITS，未安装则回退静音）"""
        fallback_duration = duration or max(len(text.strip()) * 0.25, 1.0)
        logger.warning(
            "[VoiceEngine] 本地 GPT-SoVITS 推理未实现，生成静音占位 "
            f"({fallback_duration:.1f}s): {output_path}"
        )
        self._generate_silent_wav(output_path, fallback_duration)
        return output_path

    # ------------------------------------------------------------------
    # RVC (实时变声，需安装)
    # ------------------------------------------------------------------
    def _synthesize_rvc(self, text: str, voice: VoiceProfile,
                        output_path: str,
                        duration: Optional[float] = None) -> str:
        """RVC 变声（先用基础 TTS 生成，再用 RVC 转换音色；RVC 未实现则回退静音）"""
        temp_path = tempfile.mktemp(suffix=".wav")
        try:
            self._synthesize_edge(text, voice, temp_path, duration=duration)
        except Exception:
            pass
        fallback_duration = duration or max(len(text.strip()) * 0.25, 1.0)
        logger.warning(
            "[VoiceEngine] RVC 推理未实现，生成静音占位 "
            f"({fallback_duration:.1f}s): {output_path}"
        )
        self._generate_silent_wav(output_path, fallback_duration)
        return output_path

    # ------------------------------------------------------------------
    # 批量合成
    # ------------------------------------------------------------------
    def synthesize_segments(self, segments: List[dict],
                           voice: VoiceProfile,
                           output_dir: str) -> List[str]:
        """
        批量合成多段文本

        Args:
            segments: [{"text": "...", "id": "seg_001"}, ...]
            voice: 音色
            output_dir: 输出目录

        Returns:
            输出文件路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        for i, seg in enumerate(segments):
            text = seg["text"]
            seg_id = seg.get("id", f"seg_{i:03d}")
            output_path = os.path.join(output_dir, f"{seg_id}.wav")
            duration = seg.get("end") and seg.get("start") and (seg["end"] - seg["start"])

            self.synthesize(text, voice, output_path, duration=duration)
            output_files.append(output_path)

        return output_files
