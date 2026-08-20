"""
ASR（自动语音识别）服务层

职责：
- 提取视频音频
- 调用 Whisper 或其他 ASR 模型转录
- 返回带时间戳的文本片段
"""
import os
from typing import List, Dict, Any

from src.utils import run_ffmpeg, ensure_dir, logger


class ASRService:
    """语音识别服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("models", {}).get("asr", {})
        self.provider = self.config.get("provider", "whisper")
        self.model_name = self.config.get("model", "large-v3")
        self.language = self.config.get("language")
        self.local_service = None
        if self.provider == "local":
            from src.services.local_asr_service import LocalASRService
            self.local_service = LocalASRService(config)

    def transcribe(self, video_path: str, temp_dir: str) -> List[Dict[str, Any]]:
        """转录视频音频，返回带时间戳的文本片段"""
        if self.local_service is not None:
            return self.local_service.transcribe(video_path, temp_dir)
        audio_dir = os.path.join(temp_dir, "audio", os.path.basename(video_path))
        ensure_dir(audio_dir)
        audio_path = os.path.join(audio_dir, "audio.wav")

        try:
            run_ffmpeg([
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                audio_path,
            ])
        except Exception as e:
            logger.warning(f"提取音频失败: {e}")
            return []

        if not os.path.exists(audio_path):
            return []

        if self.provider == "whisper":
            return self._transcribe_with_whisper(audio_path)

        logger.warning(f"ASR provider {self.provider} 尚未实现")
        return []

    def _transcribe_with_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            import whisper
            model = whisper.load_model(self.model_name)
            result = model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=True,
            )
            return [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                for seg in result.get("segments", [])
            ]
        except Exception as e:
            logger.warning(f"Whisper 转录失败: {e}")
            return []
