"""
custom_v1 Adapter

用于读取项目自定义的、较简单的旧版 JSON 分析结果。
输入示例：
{
  "filename": "scene_01.mp4",
  "description": "男主在咖啡馆等待，情绪焦虑",
  "tags": ["男主", "咖啡馆", "焦虑"]
}

输出：标准的 Shot 对象，标记 needs_review=true。
"""
import json
import os
from typing import Optional

from src.cv_utils import cv_pre_scan
from src.utils import sec_to_tc
from src.models import Shot
from src.adapters.base import BaseMetaAdapter, build_analyzed_shot


class CustomV1Adapter(BaseMetaAdapter):
    @property
    def name(self) -> str:
        return "custom_v1"

    def can_read(self, file_path: str) -> bool:
        if not file_path.endswith(".json"):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, dict) and "filename" in data and "description" in data
        except Exception:
            return False

    def read(self, file_path: str, video_path: Optional[str] = None) -> Shot:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = data.get("filename", video_path and os.path.basename(video_path) or "unknown.mp4")
        source_path = video_path or os.path.join(os.path.dirname(file_path), filename)

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
            logging.getLogger(__name__).warning(f"CustomV1 adapter CV 扫描失败: {e}")

        vlm_description = {
            "location": data.get("location", ""),
            "time_of_day": data.get("time_of_day", ""),
            "characters": data.get("characters", data.get("tags", [])),
            "action": data.get("action", ""),
            "emotion": data.get("emotion", ""),
            "dialogue": data.get("dialogue", ""),
            "shot_size": data.get("shot_size", ""),
            "camera_position": data.get("camera_position", ""),
            "camera_movement": data.get("camera_movement", ""),
        }

        missing = []
        if not vlm_description["location"]:
            missing.append("location")
        if not vlm_description["emotion"]:
            missing.append("emotion")

        shot = build_analyzed_shot(
            shot_id="S000",
            source_file=filename,
            source_path=source_path,
            duration_sec=duration,
            vlm_description=vlm_description,
            adapter_name=self.name,
            missing_fields=missing,
            cv_metadata=cv_meta,
        )
        # 补全可能存在的扩展字段
        shot.direction = data.get("direction", "")
        shot.performance = data.get("performance", "")
        shot.action_details = data.get("action_details", "")
        shot.continuity_score = float(data.get("continuity_score", 0.0) or 0.0)
        shot.continuity_notes = data.get("continuity_notes", "")
        shot.fps = fps
        shot.tc_out = sec_to_tc(duration, fps)
        return shot
