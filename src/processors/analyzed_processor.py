"""
ANALYZED 素材处理器

职责：
- 查找与视频对应的分析文件（支持 autocut_v1 / custom_v1 / csv）
- 通过 Adapter 转换为标准 Shot 结构
- 不分析、不切分
- 拷贝原视频到 split_clips 并生成统一配置
- 标记 needs_review=true（当关键字段缺失时）
"""
import os
from typing import List, Optional

from src.models import Shot, Provenance
from src.cv_utils import cv_pre_scan, extract_keyframes, split_video
from src.adapters import find_adapter_for_file, get_adapter
from src.utils import logger, sec_to_tc, ensure_dir
from src.processors.common import build_shot_config


class AnalyzedProcessor:
    """处理 ANALYZED 已有分析结果素材"""

    def __init__(self, output_dir: str = "./output", temp_dir: str = "./temp"):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.split_clips_dir = os.path.join(output_dir, "phase1_split_clips")
        ensure_dir(self.split_clips_dir)

    def process(self, video_path: str, meta_format: Optional[str], next_shot_id_func) -> List[Shot]:
        filename = os.path.basename(video_path)
        logger.info(f"[ANALYZED] 处理: {filename}")

        base_name = os.path.splitext(video_path)[0]
        meta_path = None

        # 按指定格式查找分析文件
        format_to_ext = {
            "custom_v1": "_meta.json",
            "autocut_v1": "_config.json",
            "csv": "_meta.csv",
            "generic_json": "_meta.json",
        }
        if meta_format:
            ext = format_to_ext.get(meta_format)
            if ext:
                candidate = base_name + ext
                if os.path.exists(candidate):
                    meta_path = candidate

        if not meta_path or not os.path.exists(meta_path):
            for ext in ["_config.json", "_meta.json", ".json", "_meta.csv", ".csv"]:
                candidate = base_name + ext
                if os.path.exists(candidate):
                    meta_path = candidate
                    break

        shot = None
        adapter_name = "fallback"
        if meta_path and os.path.exists(meta_path):
            try:
                adapter = find_adapter_for_file(meta_path)
                if not adapter:
                    logger.error(f"无法识别分析文件格式: {meta_path}")
                else:
                    shot = adapter.read(meta_path, video_path=video_path)
                    adapter_name = adapter.name
            except Exception as e:
                logger.error(f"Adapter 读取失败，使用 fallback: {e}")

        if shot is None:
            logger.warning(f"未找到/读取分析文件，将生成空 ANALYZED Shot 并标记 needs_review: {video_path}")
            shot = self._build_fallback_shot(video_path)

        shot.shot_id = next_shot_id_func()
        shot.state = "ANALYZED"
        shot.do_not_split = True

        # CV 扫描补全/校验物理字段
        try:
            cv_meta = cv_pre_scan(video_path)
            shot.cv_metadata = cv_meta
            shot.fps = cv_meta.get("fps", shot.fps or 24.0)
            shot.resolution = cv_meta.get("resolution", shot.resolution)
            shot.aspect_ratio = cv_meta.get("aspect_ratio", shot.aspect_ratio)
            shot.bitrate = cv_meta.get("bitrate", shot.bitrate)
            shot.codec = cv_meta.get("codec", shot.codec)
            shot.visual_quality = cv_meta.get("visual_quality", shot.visual_quality)
            duration = cv_meta.get("duration", shot.duration_sec)
            shot.duration_sec = duration
            shot.tc_in = sec_to_tc(0.0, shot.fps)
            shot.tc_out = sec_to_tc(duration, shot.fps)
        except Exception as e:
            logger.warning(f"ANALYZED CV 扫描失败: {e}")
            cv_meta = shot.cv_metadata or {}

        # 物理片段：直接拷贝原视频
        ext = os.path.splitext(video_path)[1] or ".mp4"
        split_clip_path = os.path.join(self.split_clips_dir, f"{shot.shot_id}{ext}")
        try:
            split_video(video_path, split_clip_path, 0.0, shot.duration_sec, copy=True)
        except Exception as e:
            logger.warning(f"ANALYZED 视频拷贝失败 {shot.shot_id}: {e}")
            split_clip_path = ""

        # 只有当关键语义字段缺失时才标记 needs_review
        has_critical_fields = bool(shot.action or shot.emotion or shot.location)
        shot.needs_review = not has_critical_fields

        # 统一 shot 配置
        content_meta = self._shot_to_content_meta(shot)
        shot.cv_metadata = build_shot_config(shot, cv_meta, content_meta, split_clip_path)

        # 溯源信息
        shot.provenance = Provenance(
            state="ANALYZED",
            generated_by=adapter_name,
            conversion={
                "from_format": adapter_name,
                "adapter": adapter_name,
                "missing_fields": self._missing_fields(shot),
                "needs_review": shot.needs_review,
            },
        )

        # 提取关键帧
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

    def _build_fallback_shot(self, video_path: str) -> Shot:
        cv_meta = cv_pre_scan(video_path)
        fps = cv_meta.get("fps", 24.0)
        duration = cv_meta.get("duration", 0.0)
        filename = os.path.basename(video_path)

        return Shot(
            shot_id="S000",
            state="ANALYZED",
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
            visual_quality=cv_meta.get("visual_quality"),
            do_not_split=True,
            needs_review=True,
            cv_metadata=cv_meta,
            provenance=Provenance(
                state="ANALYZED",
                generated_by="fallback",
                conversion={
                    "from_format": "unknown",
                    "adapter": "fallback",
                    "missing_fields": ["all semantic fields"],
                    "needs_review": True,
                },
            ),
        )

    @staticmethod
    def _shot_to_content_meta(shot: Shot) -> dict:
        """把 Shot 的语义字段打包成 build_shot_config 可读的 dict"""
        return {
            "content_summary": shot.action,
            "shot_type": shot.shot_size,
            "camera_position": shot.camera_position,
            "camera_movement": shot.camera_movement,
            "location": shot.location,
            "time_of_day": shot.time_of_day,
            "characters": shot.characters,
            "action": shot.action,
            "emotion": shot.emotion,
            "dialogue": shot.dialogue,
            "framing": shot.framing,
            "lighting": shot.lighting,
            "color_tone": shot.color_tone,
            "style": getattr(shot, "style", ""),
            "atmosphere": getattr(shot, "atmosphere", ""),
            "culture": getattr(shot, "culture", ""),
            "tags": getattr(shot, "tags", []),
            "key_objects": shot.key_objects,
            "direction": getattr(shot, "direction", ""),
            "performance": getattr(shot, "performance", ""),
            "action_details": getattr(shot, "action_details", ""),
            "continuity_score": getattr(shot, "continuity_score", 0.0),
            "continuity_notes": getattr(shot, "continuity_notes", ""),
            "is_long_take": False,
            "coherence_score": getattr(shot, "coherence_score", 0.0),
        }

    @staticmethod
    def _missing_fields(shot: Shot) -> List[str]:
        missing = []
        if not shot.action:
            missing.append("action")
        if not shot.emotion:
            missing.append("emotion")
        if not shot.location:
            missing.append("location")
        if not shot.shot_size:
            missing.append("shot_size")
        return missing
