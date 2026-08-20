"""
MaterialProcessor: 素材状态分发器

根据素材状态（RAW / PROCESSED / ANALYZED）选择对应处理器，
将具体处理逻辑下放到各状态处理器。
"""
from typing import List, Dict, Any, Optional, Callable

from src.models import Shot
from src.processors import RawProcessor, AnalyzedProcessor
from src.processors.phase1_analysis_processor import Phase1AnalysisProcessor
from src.services.vlm_service import VLMService
from src.services.video_vlm_service import VideoVLMService
from src.services.asr_service import ASRService
from src.split_scorer import SplitScorer
from src.utils import logger


class MaterialProcessor:
    """素材状态分发器"""

    def __init__(
        self,
        vlm_service: VLMService,
        video_vlm_service: VideoVLMService,
        asr_service: ASRService,
        split_scorer: SplitScorer,
        output_dir: str,
        temp_dir: str,
        processing_config: Dict[str, Any],
        config: Dict[str, Any] = None,
    ):
        self.vlm_service = vlm_service
        self.video_vlm_service = video_vlm_service
        self.asr_service = asr_service
        self.split_scorer = split_scorer
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.processing_config = processing_config
        self.config = config or {}

        # 注意：Phase 0 粗剪不依赖 VLM/ASR/SplitScorer，保留参数仅为了兼容旧接口
        self.raw_processor = RawProcessor(
            vlm_service=vlm_service,
            video_vlm_service=video_vlm_service,
            asr_service=asr_service,
            split_scorer=split_scorer,
            output_dir=output_dir,
            temp_dir=temp_dir,
            keyframe_strategy=processing_config.get("keyframe_strategy", "adaptive"),
            keyframe_interval=processing_config.get("keyframe_interval", 2.0),
            min_shot_duration=processing_config.get("min_shot_duration", 1.0),
            config=self.config,
        )
        self.phase1_processor = Phase1AnalysisProcessor(
            vlm_service=vlm_service,
            asr_service=asr_service,
            output_dir=output_dir,
            temp_dir=temp_dir,
            keyframe_strategy=processing_config.get("keyframe_strategy", "adaptive"),
            keyframe_interval=processing_config.get("keyframe_interval", 2.0),
        )
        self.analyzed_processor = AnalyzedProcessor(
            output_dir=output_dir,
            temp_dir=temp_dir,
        )

    def process(
        self,
        video_path: str,
        state: str,
        meta_format: Optional[str],
        next_shot_id_func: Callable[[], str],
    ) -> List[Shot]:
        """根据状态分发处理"""
        logger.info(f"[MaterialProcessor] 分发: {state} -> {video_path}")
        if state == "RAW":
            return self.raw_processor.process(video_path, next_shot_id_func)
        elif state == "PROCESSED":
            return self.phase1_processor.process(
                video_path=video_path,
                next_shot_id_func=next_shot_id_func,
                state="PROCESSED",
            )
        elif state == "ANALYZED":
            return self.analyzed_processor.process(video_path, meta_format, next_shot_id_func)
        else:
            logger.error(f"未知素材状态: {state}")
            return []
