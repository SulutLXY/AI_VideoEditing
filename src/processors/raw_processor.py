"""
RAW 素材处理器（Phase 0 粗剪入口）

职责：
- 对 RAW 原始素材只做 Phase 0 粗剪。
- 不调用 VLM/LLM/ASR，不建立关系图。
- 调用 RoughCutAnalyzer 完成切分并输出粗剪片段。

保留 VLM/ASR 等参数是为了兼容现有 MaterialProcessor 的调用接口，
但 Phase 0 不会使用它们。
"""
import os
from typing import List, Any, Dict, Callable

from src.models import Shot
from src.phase0_rough_cut import RoughCutAnalyzer
from src.utils import logger


class RawProcessor:
    """处理 RAW 原始素材：仅做 Phase 0 粗剪"""

    def __init__(
        self,
        vlm_service=None,
        video_vlm_service=None,
        asr_service=None,
        split_scorer=None,
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        keyframe_strategy: str = "adaptive",
        keyframe_interval: float = 2.0,
        min_shot_duration: float = 1.0,
        config: Dict[str, Any] = None,
    ):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.keyframe_strategy = keyframe_strategy
        self.keyframe_interval = keyframe_interval
        self.min_shot_duration = min_shot_duration
        self.config = config or {}

    def process(self, video_path: str, next_shot_id_func: Callable[[], str]) -> List[Shot]:
        """处理单个 RAW 视频：只做 Phase 0 粗剪"""
        filename = os.path.basename(video_path)
        logger.info(f"[RAW] Phase 0 粗剪: {filename}")

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer._process_video(video_path)

        # 重新分配 shot_id 以匹配外部计数器
        result = []
        for shot in shots:
            shot.shot_id = next_shot_id_func()
            result.append(shot)

        return result
