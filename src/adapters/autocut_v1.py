"""
autocut_v1 Adapter

用于读取 LLM-AutoCut 自身生成的 Shot 配置文件（*_config.json）。
当用户已有分析结果但格式为我们项目内部格式时，直接转译为标准 Shot 对象，
不重新分析、不切分。
"""
import json
import os
from typing import Optional

from src.cv_utils import cv_pre_scan
from src.utils import sec_to_tc
from src.models import Shot
from src.adapters.base import BaseMetaAdapter, build_analyzed_shot


class AutocutV1Adapter(BaseMetaAdapter):
    """读取 LLM-AutoCut 生成的 *_config.json"""

    @property
    def name(self) -> str:
        return "autocut_v1"

    def can_read(self, file_path: str) -> bool:
        if not file_path.endswith(".json"):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return False
            # 我们生成的配置至少包含 shot_id + clip_path 或 shot_config
            return "shot_id" in data and ("clip_path" in data or "shot_config" in data)
        except Exception:
            return False

    def read(self, file_path: str, video_path: Optional[str] = None) -> Shot:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 优先使用内嵌的 shot_config，否则使用顶层字段
        cfg = data.get("shot_config", data)

        filename = data.get("source_file") or cfg.get("source_file") or (
            video_path and os.path.basename(video_path) or "unknown.mp4"
        )
        source_path = video_path or data.get("source_path") or cfg.get("source_path") or (
            os.path.join(os.path.dirname(file_path), filename)
        )

        cv_meta = None
        duration = 0.0
        fps = 24.0
        try:
            if os.path.exists(source_path):
                cv_meta = cv_pre_scan(source_path)
                duration = cv_meta.get("duration", 0.0)
                fps = cv_meta.get("fps", 24.0)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"autocut_v1 adapter CV 扫描失败: {e}")

        # 如果配置里已有 duration/fps 等物理字段，也作为补充
        duration = data.get("duration_sec") or cfg.get("duration_sec") or duration
        fps = data.get("fps") or cfg.get("fps") or fps

        # 统一字段映射
        vlm_description = {
            "location": cfg.get("location", ""),
            "time_of_day": cfg.get("time_of_day", ""),
            "characters": cfg.get("characters", []),
            "action": cfg.get("action", cfg.get("content_summary", "")),
            "emotion": cfg.get("emotion", ""),
            "dialogue": cfg.get("dialogue", ""),
            "shot_size": cfg.get("shot_type", cfg.get("shot_size", "")),
            "camera_position": cfg.get("camera_position", ""),
            "camera_movement": cfg.get("camera_movement", ""),
            "framing": cfg.get("framing", ""),
            "lighting": cfg.get("lighting", ""),
            "color_tone": cfg.get("color_tone", ""),
        }

        # 收集缺失字段
        missing = [k for k, v in vlm_description.items() if not v and k not in ("framing", "lighting", "color_tone")]

        shot = build_analyzed_shot(
            shot_id="S000",
            source_file=filename,
            source_path=source_path,
            duration_sec=duration,
            vlm_description=vlm_description,
            adapter_name=self.name,
            missing_fields=missing,
            cv_metadata=cv_meta,
            style=cfg.get("style", ""),
            atmosphere=cfg.get("atmosphere", ""),
            culture=cfg.get("culture", ""),
            tags=cfg.get("tags", []),
            key_objects=cfg.get("key_objects", []),
            is_long_take=cfg.get("is_long_take", False),
            coherence_score=cfg.get("coherence_score", 0.0),
        )
        shot.fps = fps
        shot.resolution = tuple(cfg.get("resolution", cv_meta and cv_meta.get("resolution") or (1920, 1080)))
        shot.aspect_ratio = cfg.get("aspect_ratio", cv_meta and cv_meta.get("aspect_ratio") or "16:9")
        shot.bitrate = cfg.get("bitrate", cv_meta and cv_meta.get("bitrate"))
        shot.codec = cfg.get("codec", cv_meta and cv_meta.get("codec"))
        shot.visual_quality = cfg.get("visual_quality", cv_meta and cv_meta.get("visual_quality"))
        shot.tc_in = data.get("tc_in", sec_to_tc(0.0, fps))
        shot.tc_out = data.get("tc_out", sec_to_tc(duration, fps))

        # 补全新字段（兼容老配置）
        shot.direction = cfg.get("direction", "")
        shot.performance = cfg.get("performance", "")
        shot.action_details = cfg.get("action_details", "")
        shot.continuity_score = float(cfg.get("continuity_score", 0.0) or 0.0)
        shot.continuity_notes = cfg.get("continuity_notes", "")

        # 如果已有 split_clip_path，保留下来
        if cfg.get("split_clip_path"):
            shot.cv_metadata = cv_meta or {}
            shot.cv_metadata.setdefault("shot_config", {}).update({
                "split_clip_path": cfg.get("split_clip_path"),
            })

        return shot
