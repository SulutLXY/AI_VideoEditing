"""
csv_meta Adapter

用于读取 CSV 格式的分析结果。
输入示例：
filename,location,emotion,action,characters
scene_01.mp4,咖啡馆,焦虑,等待,男主

输出：标准的 Shot 对象，标记 needs_review=true。
"""
import csv
import os
from typing import Optional

from src.cv_utils import cv_pre_scan
from src.utils import sec_to_tc
from src.models import Shot
from src.adapters.base import BaseMetaAdapter, build_analyzed_shot


class CsvMetaAdapter(BaseMetaAdapter):
    @property
    def name(self) -> str:
        return "csv"

    def can_read(self, file_path: str) -> bool:
        return file_path.endswith(".csv")

    def read(self, file_path: str, video_path: Optional[str] = None) -> Shot:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"CSV 文件为空: {file_path}")

        data = rows[0]
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
            logging.getLogger(__name__).warning(f"CSV adapter CV 扫描失败: {e}")

        characters = []
        if data.get("characters"):
            characters = [c.strip() for c in data["characters"].split(",") if c.strip()]

        vlm_description = {
            "location": data.get("location", ""),
            "time_of_day": data.get("time_of_day", ""),
            "characters": characters,
            "action": data.get("action", ""),
            "emotion": data.get("emotion", ""),
            "dialogue": data.get("dialogue", ""),
            "shot_size": data.get("shot_size", ""),
            "camera_position": data.get("camera_position", ""),
            "camera_movement": data.get("camera_movement", ""),
        }

        missing = [k for k, v in vlm_description.items() if not v]

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
        # CSV 列若包含扩展字段则补全
        shot.direction = data.get("direction", "")
        shot.performance = data.get("performance", "")
        shot.action_details = data.get("action_details", "")
        shot.continuity_score = float(data.get("continuity_score", 0.0) or 0.0)
        shot.continuity_notes = data.get("continuity_notes", "")
        shot.fps = fps
        shot.tc_out = sec_to_tc(duration, fps)
        return shot
