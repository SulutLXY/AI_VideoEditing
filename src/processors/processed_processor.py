"""
PROCESSED 素材处理器

职责：
- 对已经人工处理好的视频只做整体内容分析
- 不切分
- 生成统一配置文件与可直接使用的物理片段（原视频拷贝）
- 返回单个 Shot 对象
"""
import os
from typing import List, Dict, Any

from src.models import Shot, Provenance
from src.services.vlm_service import VLMService
from src.services.asr_service import ASRService
from src.cv_utils import cv_pre_scan, extract_keyframes, split_video
from src.utils import logger, sec_to_tc, ensure_dir
from src.processors.common import build_shot_config


class ProcessedProcessor:
    """处理 PROCESSED 已处理素材"""

    def __init__(
        self,
        vlm_service: VLMService,
        asr_service: ASRService,
        temp_dir: str,
        output_dir: str = "./output",
    ):
        self.vlm_service = vlm_service
        self.asr_service = asr_service
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.split_clips_dir = os.path.join(output_dir, "phase1_split_clips")
        ensure_dir(self.split_clips_dir)

    def process(self, video_path: str, next_shot_id_func) -> List[Shot]:
        filename = os.path.basename(video_path)
        logger.info(f"[PROCESSED] 处理: {filename}")

        cv_meta = cv_pre_scan(video_path)
        fps = cv_meta.get("fps", 24.0)
        duration = cv_meta.get("duration", 0.0)

        frames = self.vlm_service.sample_frames(video_path, duration, self.temp_dir)
        description = self.vlm_service.analyze_whole_video(video_path, frames, duration)

        asr_segments = self.asr_service.transcribe(video_path, self.temp_dir)
        dialogue_text = " ".join([seg["text"] for seg in asr_segments])

        shot = Shot(
            shot_id=next_shot_id_func(),
            state="PROCESSED",
            source_file=filename,
            source_path=video_path,
            tc_in=sec_to_tc(0.0, fps),
            tc_out=sec_to_tc(duration, fps),
            duration_sec=duration,
            fps=fps,
            resolution=cv_meta.get("resolution", (1920, 1080)),
            aspect_ratio=cv_meta.get("aspect_ratio", "16:9"),
            bitrate=cv_meta.get("bitrate"),
            codec=cv_meta.get("codec"),
            location=description.get("location", ""),
            time_of_day=description.get("time_of_day", ""),
            characters=description.get("characters", []),
            action=description.get("action", ""),
            emotion=description.get("emotion", ""),
            dialogue=dialogue_text,
            shot_size=description.get("shot_size", ""),
            camera_position=description.get("camera_position", ""),
            camera_movement=description.get("camera_movement", ""),
            visual_quality=cv_meta.get("visual_quality"),
            framing=description.get("framing", ""),
            lighting=description.get("lighting", ""),
            color_tone=description.get("color_tone", ""),
            do_not_split=True,
            asr_text=dialogue_text,
            provenance=Provenance(
                state="PROCESSED",
                generated_by="vlm_analysis",
                split_decision={
                    "score": 0.0,
                    "reason": "PROCESSED 素材不拆分",
                    "protected": True,
                },
            ),
        )

        if description.get("internal_segments"):
            shot.internal_segments = description.get("internal_segments")

        # 物理片段：PROCESSED 不重新编码，直接拷贝整段视频
        ext = os.path.splitext(video_path)[1] or ".mp4"
        split_clip_path = os.path.join(self.split_clips_dir, f"{shot.shot_id}{ext}")
        try:
            split_video(video_path, split_clip_path, 0.0, duration, copy=True)
        except Exception as e:
            logger.warning(f"PROCESSED 视频拷贝失败 {shot.shot_id}: {e}")
            split_clip_path = ""

        # 统一 shot 配置
        shot.cv_metadata = build_shot_config(shot, cv_meta, description, split_clip_path)

        # 提取关键帧供后续阶段使用
        keyframes_dir = os.path.join(self.output_dir, "phase1_keyframes")
        ensure_dir(keyframes_dir)
        shot.keyframes = extract_keyframes(
            video_path=video_path,
            shot_id=shot.shot_id,
            tc_in=shot.tc_in,
            tc_out=shot.tc_out,
            fps=shot.fps,
            output_dir=keyframes_dir,
            strategy="adaptive",
            interval=2.0,
        )

        return [shot]
