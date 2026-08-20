"""
Phase 2 素材库轻量清点

职责：
- 当 Phase 2 没有现成的 phase1_analysis.json 时，从用户指定的素材库目录
  做传统 CV 扫描，生成仅含物理属性的基础 Shot 清单。
- 不调用任何 VLM/LLM，只做 ffprobe / OpenCV 层面的清点。
- 为每个视频生成一个 Shot，状态标记为 RAW，并注明未经视觉语义分析。
"""
import os
from typing import List, Optional

from src.cv_utils import cv_pre_scan
from src.models import Shot
from src.utils import get_video_files, sec_to_tc, logger


class MaterialInventoryBuilder:
    """从素材文件夹构建轻量 Shot 清单"""

    def __init__(self):
        self._counter = 0

    def _next_shot_id(self) -> str:
        self._counter += 1
        return f"S{self._counter:03d}"

    def build_shots_from_directory(self, directory: str) -> List[Shot]:
        """扫描目录下所有视频文件，返回基础 Shot 列表"""
        if not directory or not os.path.isdir(directory):
            logger.warning(f"素材库目录不存在或为空: {directory}")
            return []

        video_paths = get_video_files(directory)
        if not video_paths:
            logger.warning(f"素材库目录中未找到视频文件: {directory}")
            return []

        logger.info(f"Phase 2 素材库清点: 在 {directory} 发现 {len(video_paths)} 个视频")

        shots = []
        for video_path in video_paths:
            shot = self._build_shot(video_path)
            if shot:
                shots.append(shot)

        logger.info(f"Phase 2 素材库清点完成: 生成 {len(shots)} 个基础 Shot")
        return shots

    def _build_shot(self, video_path: str) -> Optional[Shot]:
        """对单个视频做 CV 预扫描并生成 Shot"""
        try:
            cv_meta = cv_pre_scan(video_path)
        except Exception as e:
            logger.warning(f"素材库清点跳过无法解析的视频: {video_path} ({e})")
            return None

        duration = float(cv_meta.get("duration", 0.0))
        fps = float(cv_meta.get("fps", 24.0))
        resolution = cv_meta.get("resolution", (1920, 1080))
        width, height = resolution[0], resolution[1]
        aspect_ratio = cv_meta.get("aspect_ratio", f"{width}:{height}")

        shot_id = self._next_shot_id()
        shot = Shot(
            shot_id=shot_id,
            state="RAW",
            source_file=os.path.basename(video_path),
            source_path=os.path.abspath(video_path),
            tc_in="00:00:00:00",
            tc_out=sec_to_tc(duration, fps),
            duration_sec=duration,
            fps=fps,
            resolution=(width, height),
            aspect_ratio=aspect_ratio,
            bitrate=cv_meta.get("bitrate"),
            codec=cv_meta.get("codec"),
            visual_quality=cv_meta.get("visual_quality"),
            cv_metadata=cv_meta,
            needs_review=True,
            # 明确标记该镜头未做视觉语义分析
            tags=["cv_inventory", "no_vlm_analysis"],
        )
        return shot
