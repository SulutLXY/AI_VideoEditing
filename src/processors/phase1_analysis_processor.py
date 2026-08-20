"""
Phase 1 分析处理器

职责：
- 对已经切好的视频片段（Phase 0 输出 或 PROCESSED/ANALYZED 素材）
  调用 VLM 做镜头内容细节分析。
- 调用 ASR 做台词转录。
- 提取关键帧。
- 生成完整 Shot 与配置文件。
- 不做切分。
"""
import os
from typing import List, Dict, Any, Optional, Callable

from src.models import Shot, Provenance
from src.services.vlm_service import VLMService
from src.services.asr_service import ASRService
from src.cv_utils import cv_pre_scan, extract_keyframes
from src.utils import logger, sec_to_tc, ensure_dir
from src.processors.common import build_shot_config


class Phase1AnalysisProcessor:
    """Phase 1：对单个视频片段做多模态语义分析"""

    def __init__(
        self,
        vlm_service: VLMService,
        asr_service: ASRService,
        face_service: Optional[Any] = None,
        output_dir: str = "./output",
        temp_dir: str = "./temp",
        keyframe_strategy: str = "adaptive",
        keyframe_interval: float = 2.0,
    ):
        self.vlm_service = vlm_service
        self.asr_service = asr_service
        self.face_service = face_service
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.keyframe_strategy = keyframe_strategy
        self.keyframe_interval = keyframe_interval
        self.split_clips_dir = os.path.join(output_dir, "phase1_split_clips")
        ensure_dir(self.split_clips_dir)

    def process(
        self,
        video_path: str,
        next_shot_id_func: Callable[[], str],
        shot_id: Optional[str] = None,
        state: str = "RAW",
        cv_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Shot]:
        """分析单个视频片段，返回完整 Shot"""
        filename = os.path.basename(video_path)
        logger.info(f"[Phase 1] 分析: {filename}")

        # 1. CV 预扫描（若未传入则重新扫描）
        if cv_meta is None:
            cv_meta = cv_pre_scan(video_path)
        fps = cv_meta.get("fps", 24.0)
        duration = cv_meta.get("duration", 0.0)

        # 2. VLM 采样 + 分析
        frames = self.vlm_service.sample_frames(video_path, duration, self.temp_dir)
        description = self.vlm_service.analyze_whole_video(video_path, frames, duration)

        # 3. ASR 转录
        asr_segments = self.asr_service.transcribe(video_path, self.temp_dir)
        dialogue_text = " ".join([seg["text"] for seg in asr_segments])

        # 4. 人脸识别（若启用本地模型）
        face_characters = []
        if self.face_service is not None:
            try:
                face_characters = self.face_service.identify_characters(
                    video_path, start_sec=0.0, end_sec=duration
                )
            except Exception as e:
                logger.warning(f"[Phase 1] 人脸识别失败: {e}")

        # 合并 VLM 与 Face 识别出的角色（VLM 可能用通用描述，Face 给出具体身份）
        characters = list(dict.fromkeys(description.get("characters", []) or []))
        for name in face_characters:
            if name not in characters:
                characters.append(name)

        # 5. 生成 Shot
        if shot_id is None:
            shot_id = next_shot_id_func()

        shot = Shot(
            shot_id=shot_id,
            state=state,
            source_file=filename,
            source_path=os.path.abspath(video_path),
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
            characters=characters,
            action=description.get("action", ""),
            emotion=description.get("emotion", ""),
            dialogue=dialogue_text,
            asr_text=dialogue_text,
            shot_size=description.get("shot_size", ""),
            camera_position=description.get("camera_position", ""),
            camera_movement=description.get("camera_movement", ""),
            framing=description.get("framing", ""),
            lighting=description.get("lighting", ""),
            color_tone=description.get("color_tone", ""),
            style=description.get("style", ""),
            atmosphere=description.get("atmosphere", ""),
            culture=description.get("culture", ""),
            tags=description.get("tags", []),
            key_objects=description.get("key_objects", []),
            direction=description.get("direction", ""),
            performance=description.get("performance", ""),
            action_details=description.get("action_details", ""),
            continuity_score=float(description.get("continuity_score", 0.0) or 0.0),
            continuity_notes=description.get("continuity_notes", ""),
            visual_quality=cv_meta.get("visual_quality"),
            do_not_split=True,
            needs_review=False,
            cv_metadata=cv_meta,
            provenance=Provenance(
                state=state,
                generated_by="phase1_analysis",
                split_decision={
                    "score": 0.0,
                    "reason": "Phase 1 只做分析不切分",
                    "protected": True,
                },
            ),
        )

        if description.get("internal_segments"):
            shot.internal_segments = description.get("internal_segments")

        # 6. 拷贝/物理片段到 phase1_split_clips（用于后续阶段统一读取）
        ext = os.path.splitext(video_path)[1] or ".mp4"
        split_clip_path = os.path.join(self.split_clips_dir, f"{shot.shot_id}{ext}")
        try:
            from src.cv_utils import split_video
            split_video(video_path, split_clip_path, 0.0, duration, copy=True)
        except Exception as e:
            logger.warning(f"Phase 1 视频拷贝失败 {shot.shot_id}: {e}")
            split_clip_path = ""

        # 7. 生成统一 shot 配置
        content_meta = {
            "content_summary": description.get("action", ""),
            "shot_type": description.get("shot_size", ""),
            "camera_position": description.get("camera_position", ""),
            "camera_movement": description.get("camera_movement", ""),
            "location": description.get("location", ""),
            "time_of_day": description.get("time_of_day", ""),
            "characters": characters,
            "action": description.get("action", ""),
            "emotion": description.get("emotion", ""),
            "dialogue": dialogue_text,
            "framing": description.get("framing", ""),
            "lighting": description.get("lighting", ""),
            "color_tone": description.get("color_tone", ""),
            "style": description.get("style", ""),
            "atmosphere": description.get("atmosphere", ""),
            "culture": description.get("culture", ""),
            "tags": description.get("tags", []),
            "key_objects": description.get("key_objects", []),
            "direction": description.get("direction", ""),
            "performance": description.get("performance", ""),
            "action_details": description.get("action_details", ""),
            "continuity_score": float(description.get("continuity_score", 0.0) or 0.0),
            "continuity_notes": description.get("continuity_notes", ""),
            "is_long_take": False,
            "coherence_score": 0.0,
        }
        shot.cv_metadata = build_shot_config(shot, cv_meta, content_meta, split_clip_path)

        # 8. 提取关键帧
        keyframes_dir = os.path.join(self.output_dir, "phase1_keyframes")
        ensure_dir(keyframes_dir)
        shot.keyframes = extract_keyframes(
            video_path=video_path,
            shot_id=shot.shot_id,
            tc_in=shot.tc_in,
            tc_out=shot.tc_out,
            fps=shot.fps,
            output_dir=keyframes_dir,
            strategy=self.keyframe_strategy,
            interval=self.keyframe_interval,
        )

        return [shot]
