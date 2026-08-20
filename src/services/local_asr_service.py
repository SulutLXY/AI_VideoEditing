"""
本地 ASR 服务封装

把 video-intelligence-extractor 的 ASREngine（SenseVoice）适配成项目 ASRService 接口。
"""
import os
from typing import List, Dict, Any

from src.utils import logger


class LocalASRService:
    """基于 SenseVoice 的本地语音识别服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local_cfg = config.get("models", {}).get("local", {})
        asr_cfg = local_cfg.get("asr", {})

        self.enabled = bool(local_cfg.get("enabled", False))
        self.device = local_cfg.get("device", "cuda")
        self.cache_dir = local_cfg.get("cache_dir")
        self.model_id = asr_cfg.get("model_id", "iic/SenseVoiceSmall")
        self.batch_size = asr_cfg.get("batch_size", 1)

        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from src.local_models.asr_engine import ASREngine
            self._engine = ASREngine(
                model_id=self.model_id,
                device=self.device,
                batch_size=self.batch_size,
                cache_dir=self.cache_dir,
            )
        return self._engine

    def transcribe(self, video_path: str, temp_dir: str) -> List[Dict[str, Any]]:
        """转录视频音频，返回带时间戳的文本片段"""
        if not self.enabled:
            logger.warning("[LocalASR] 本地模型未启用，跳过本地 ASR")
            return []

        logger.info(f"[LocalASR] 本地 ASR 转录: {os.path.basename(video_path)}")
        try:
            engine = self._get_engine()
            result = engine.process_video(video_path)
            return result.get("transcriptions", [])
        except Exception as e:
            logger.error(f"[LocalASR] 本地 ASR 失败: {e}")
            return []

    def unload(self):
        """卸载模型释放显存"""
        if self._engine is not None:
            self._engine.unload()
            self._engine = None
