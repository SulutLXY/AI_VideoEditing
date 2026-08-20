"""
generic_json Adapter

通用 JSON 分析结果适配器。
用于读取字段命名不统一的第三方/手工分析 JSON，自动映射到标准 Shot 结构。

匹配优先级最低：只有在 autocut_v1 / custom_v1 / csv 都识别失败后才启用。
"""
import json
import os
from typing import Any, Dict, List, Optional

from src.cv_utils import cv_pre_scan
from src.utils import sec_to_tc, logger
from src.models import Shot
from src.adapters.base import BaseMetaAdapter, build_analyzed_shot


# 常见字段别名映射（支持中英文及缩写）
_FIELD_ALIASES: Dict[str, List[str]] = {
    "shot_size": ["shot_size", "shot_type", "景别", "镜头类型", "shot"],
    "camera_position": ["camera_position", "机位", "camera_pos", "position"],
    "camera_movement": ["camera_movement", "运镜", "movement", "camera_move"],
    "framing": ["framing", "构图", "composition"],
    "lighting": ["lighting", "光效", "light"],
    "color_tone": ["color_tone", "色调", "color", "tone"],
    "location": ["location", "scene", "地点", "场景", "place"],
    "time_of_day": ["time_of_day", "time", "时间", "time_of_scene"],
    "characters": ["characters", "roles", "人物", "角色", "actors", "character"],
    "action": ["action", "content", "summary", "动作", "内容", "description", "describe"],
    "action_details": ["action_details", "动作细节", "action_detail", "details"],
    "emotion": ["emotion", "mood", "情绪", "情感", "feeling"],
    "dialogue": ["dialogue", "lines", "台词", "对白", "line"],
    "performance": ["performance", "表演", "acting", "performance_quality"],
    "direction": ["direction", "方向", "screen_direction", "movement_direction"],
    "style": ["style", "风格"],
    "atmosphere": ["atmosphere", "氛围", "mood_atmosphere"],
    "culture": ["culture", "文化", "culture_background"],
    "tags": ["tags", "标签", "tag"],
    "key_objects": ["key_objects", "props", "道具", "objects", "key_props"],
    "is_long_take": ["is_long_take", "long_take", "长镜头", "一镜到底"],
    "coherence_score": ["coherence_score", "连贯性", "coherence"],
    "continuity_score": ["continuity_score", "连续性评分", "continuity"],
    "continuity_notes": ["continuity_notes", "连续性说明", "continuity_note"],
    "duration_sec": ["duration_sec", "duration", "时长"],
    "fps": ["fps", "frame_rate", "帧率"],
    "resolution": ["resolution", "分辨率"],
    "aspect_ratio": ["aspect_ratio", "画幅比", "aspect"],
    "bitrate": ["bitrate", "码率"],
    "codec": ["codec", "编码"],
    "visual_quality": ["visual_quality", "画质", "quality"],
    "tc_in": ["tc_in", "timecode_in", "入点"],
    "tc_out": ["tc_out", "timecode_out", "出点"],
}


def _pick_field(data: Dict[str, Any], field: str) -> Any:
    """按别名从 dict 中取值"""
    for key in _FIELD_ALIASES.get(field, [field]):
        if key in data and data[key] is not None:
            return data[key]
    return None


def _pick_string(data: Dict[str, Any], field: str) -> str:
    value = _pick_field(data, field)
    return str(value) if value is not None else ""


def _pick_list(data: Dict[str, Any], field: str) -> List[str]:
    value = _pick_field(data, field)
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _pick_float(data: Dict[str, Any], field: str) -> float:
    value = _pick_field(data, field)
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick_bool(data: Dict[str, Any], field: str) -> bool:
    value = _pick_field(data, field)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "是", "1")
    return False


class GenericJsonAdapter(BaseMetaAdapter):
    """通用 JSON 适配器：自动映射常见字段名到标准 Shot"""

    @property
    def name(self) -> str:
        return "generic_json"

    def can_read(self, file_path: str) -> bool:
        if not file_path.endswith(".json"):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return isinstance(data, dict)
        except Exception:
            return False

    def read(self, file_path: str, video_path: Optional[str] = None) -> Shot:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 优先使用 shot_config 内嵌结构
        cfg = data.get("shot_config", data)

        filename = (
            _pick_string(data, "source_file")
            or _pick_string(cfg, "source_file")
            or (video_path and os.path.basename(video_path) or "unknown.mp4")
        )
        source_path = (
            video_path
            or _pick_string(data, "source_path")
            or _pick_string(cfg, "source_path")
            or os.path.join(os.path.dirname(file_path), filename)
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
            logger.warning(f"generic_json adapter CV 扫描失败: {e}")

        # 用配置中的物理字段覆盖 CV 结果
        duration = _pick_float(data, "duration_sec") or _pick_float(cfg, "duration_sec") or duration
        fps = _pick_float(data, "fps") or _pick_float(cfg, "fps") or fps

        vlm_description = {
            "location": _pick_string(cfg, "location"),
            "time_of_day": _pick_string(cfg, "time_of_day"),
            "characters": _pick_list(cfg, "characters"),
            "action": _pick_string(cfg, "action"),
            "emotion": _pick_string(cfg, "emotion"),
            "dialogue": _pick_string(cfg, "dialogue"),
            "shot_size": _pick_string(cfg, "shot_size"),
            "camera_position": _pick_string(cfg, "camera_position"),
            "camera_movement": _pick_string(cfg, "camera_movement"),
            "framing": _pick_string(cfg, "framing"),
            "lighting": _pick_string(cfg, "lighting"),
            "color_tone": _pick_string(cfg, "color_tone"),
            "direction": _pick_string(cfg, "direction"),
            "performance": _pick_string(cfg, "performance"),
            "action_details": _pick_string(cfg, "action_details"),
            "continuity_score": _pick_float(cfg, "continuity_score"),
            "continuity_notes": _pick_string(cfg, "continuity_notes"),
        }

        # 收集缺失字段（关键内容字段为空时记录）
        critical_fields = ["location", "action", "emotion", "shot_size"]
        missing = [k for k in critical_fields if not vlm_description.get(k)]

        shot = build_analyzed_shot(
            shot_id="S000",
            source_file=filename,
            source_path=source_path,
            duration_sec=duration,
            vlm_description=vlm_description,
            adapter_name=self.name,
            missing_fields=missing,
            cv_metadata=cv_meta,
            style=_pick_string(cfg, "style"),
            atmosphere=_pick_string(cfg, "atmosphere"),
            culture=_pick_string(cfg, "culture"),
            tags=_pick_list(cfg, "tags"),
            key_objects=_pick_list(cfg, "key_objects"),
            is_long_take=_pick_bool(cfg, "is_long_take"),
            coherence_score=_pick_float(cfg, "coherence_score"),
        )
        # build_analyzed_shot 未覆盖的新字段需手动补全
        shot.direction = _pick_string(cfg, "direction")
        shot.performance = _pick_string(cfg, "performance")
        shot.action_details = _pick_string(cfg, "action_details")
        shot.continuity_score = _pick_float(cfg, "continuity_score")
        shot.continuity_notes = _pick_string(cfg, "continuity_notes")
        shot.fps = fps
        shot.resolution = tuple(
            cfg.get("resolution")
            or cv_meta and cv_meta.get("resolution")
            or (1920, 1080)
        )
        shot.aspect_ratio = _pick_string(cfg, "aspect_ratio") or (
            cv_meta and cv_meta.get("aspect_ratio") or "16:9"
        )
        shot.bitrate = _pick_string(cfg, "bitrate") or (cv_meta and cv_meta.get("bitrate"))
        shot.codec = _pick_string(cfg, "codec") or (cv_meta and cv_meta.get("codec"))
        shot.visual_quality = _pick_float(cfg, "visual_quality") or (
            cv_meta and cv_meta.get("visual_quality")
        )
        shot.tc_in = _pick_string(cfg, "tc_in") or sec_to_tc(0.0, fps)
        shot.tc_out = _pick_string(cfg, "tc_out") or sec_to_tc(duration, fps)

        # 保留已有的 split_clip_path
        if cfg.get("split_clip_path"):
            shot.cv_metadata = cv_meta or {}
            shot.cv_metadata.setdefault("shot_config", {}).update({
                "split_clip_path": cfg.get("split_clip_path"),
            })

        return shot
