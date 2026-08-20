"""
处理器公共工具

把 RAW / PROCESSED / ANALYZED 三种状态最终输出的 Shot 配置统一到这里生成，
避免三套处理器各自维护一份字段列表。
"""
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from src.models import Segment, Shot


def _get_field(meta: Union[Segment, Dict[str, Any], None], key: str, default: Any = "") -> Any:
    """统一读取 Segment 或 dict 中的字段"""
    if meta is None:
        return default

    if isinstance(meta, Segment):
        # 字段别名
        if key == "shot_type":
            return getattr(meta, "shot_size", default) or default
        if key == "content_summary":
            return getattr(meta, "description", "") or getattr(meta, "action", "") or default
        return getattr(meta, key, default) or default

    if isinstance(meta, dict):
        if key in meta:
            value = meta[key]
            return value if value is not None else default
        # 别名映射
        aliases = {
            "shot_type": ["shot_size"],
            "content_summary": ["description", "action"],
        }
        for alias in aliases.get(key, []):
            if alias in meta and meta[alias] is not None:
                return meta[alias]
        return default

    return default


def _build_tags(meta: Union[Segment, Dict[str, Any], None]) -> List[str]:
    """从内容元数据中提取/聚合标签"""
    tags = _get_field(meta, "tags", [])
    if isinstance(tags, list) and tags:
        return [str(t) for t in tags]

    tags = []
    shot_type = _get_field(meta, "shot_type", "")
    if shot_type and shot_type != "未知":
        tags.append(shot_type)
    camera_movement = _get_field(meta, "camera_movement", "")
    if camera_movement and camera_movement != "未知":
        tags.append(camera_movement)
    emotion = _get_field(meta, "emotion", "")
    if emotion:
        tags.append(emotion)
    location = _get_field(meta, "location", "")
    if location:
        tags.append(location)
    time_of_day = _get_field(meta, "time_of_day", "")
    if time_of_day:
        tags.append(time_of_day)
    style = _get_field(meta, "style", "")
    if style:
        tags.append(style)
    return tags


def build_shot_config(
    shot: Shot,
    cv_meta: Optional[Dict[str, Any]],
    content_meta: Union[Segment, Dict[str, Any], None],
    split_clip_path: str = "",
) -> Dict[str, Any]:
    """为 Shot 生成统一的 JSON 配置

    参数：
        shot: 已建立基础字段的 Shot 对象
        cv_meta: CV 预扫描得到的物理属性
        content_meta: 内容语义来源，可以是 Segment 或 dict
        split_clip_path: 切分/拷贝后的物理片段路径
    """
    cv_meta = cv_meta or {}

    config = {
        # 物理属性
        "duration_sec": shot.duration_sec,
        "resolution": cv_meta.get("resolution", shot.resolution),
        "fps": shot.fps,
        "aspect_ratio": cv_meta.get("aspect_ratio", shot.aspect_ratio),
        "bitrate": cv_meta.get("bitrate", shot.bitrate),
        "codec": cv_meta.get("codec", shot.codec),
        "visual_quality": cv_meta.get("visual_quality", shot.visual_quality),

        # 镜头类型与机位
        "shot_type": _get_field(content_meta, "shot_type", shot.shot_size),
        "camera_position": _get_field(content_meta, "camera_position", shot.camera_position),
        "camera_movement": _get_field(content_meta, "camera_movement", shot.camera_movement),

        # 内容语义
        "content_summary": _get_field(content_meta, "content_summary", shot.action),
        "location": _get_field(content_meta, "location", shot.location),
        "time_of_day": _get_field(content_meta, "time_of_day", shot.time_of_day),
        "characters": _get_field(content_meta, "characters", shot.characters),
        "action": _get_field(content_meta, "action", shot.action),
        "emotion": _get_field(content_meta, "emotion", shot.emotion),
        "dialogue": _get_field(content_meta, "dialogue", shot.dialogue),

        # 影视风格与标签
        "framing": _get_field(content_meta, "framing", shot.framing),
        "lighting": _get_field(content_meta, "lighting", shot.lighting),
        "color_tone": _get_field(content_meta, "color_tone", shot.color_tone),
        "style": _get_field(content_meta, "style", getattr(shot, "style", "")),
        "atmosphere": _get_field(content_meta, "atmosphere", getattr(shot, "atmosphere", "")),
        "culture": _get_field(content_meta, "culture", getattr(shot, "culture", "")),
        "tags": _build_tags(content_meta),
        "key_objects": _get_field(content_meta, "key_objects", shot.key_objects),

        # 新增影视方向/表演/动作细节/连续性
        "direction": _get_field(content_meta, "direction", getattr(shot, "direction", "")),
        "performance": _get_field(content_meta, "performance", getattr(shot, "performance", "")),
        "action_details": _get_field(content_meta, "action_details", getattr(shot, "action_details", "")),
        "continuity_score": float(_get_field(content_meta, "continuity_score", getattr(shot, "continuity_score", 0.0)) or 0.0),
        "continuity_notes": _get_field(content_meta, "continuity_notes", getattr(shot, "continuity_notes", "")),

        # 物理来源与拆分
        "source_file": shot.source_file,
        "source_path": shot.source_path,
        "source_tc_in": shot.tc_in,
        "source_tc_out": shot.tc_out,
        "split_clip_path": split_clip_path,
        "is_long_take": bool(_get_field(content_meta, "is_long_take", False)),
        "coherence_score": float(_get_field(content_meta, "coherence_score", 0.0) or 0.0),

        # 处理状态
        "state": shot.state,
        "do_not_split": shot.do_not_split,
        "needs_review": shot.needs_review,
    }

    # 与原始 cv_meta 合并，保留原有字段
    base = dict(cv_meta) if cv_meta else {}
    base["shot_config"] = config
    return base
