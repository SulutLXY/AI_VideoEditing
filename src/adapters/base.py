"""
Adapter 基类与工具函数

提供 BaseMetaAdapter 和 build_analyzed_shot，避免子模块循环引用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from src.models import Shot, Provenance, Relationships


class BaseMetaAdapter(ABC):
    """元数据适配器基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器标识名"""
        pass

    @abstractmethod
    def can_read(self, file_path: str) -> bool:
        """判断是否能读取该文件"""
        pass

    @abstractmethod
    def read(self, file_path: str, video_path: Optional[str] = None) -> Shot:
        """读取文件并转换为 Shot 对象"""
        pass

    def write(self, shot: Shot, output_path: str) -> str:
        """可选：将 Shot 写回该格式（默认未实现）"""
        raise NotImplementedError(f"Adapter {self.name} 未实现 write")


@dataclass
class ConversionInfo:
    """转换信息"""
    from_format: str
    adapter: str
    missing_fields: list
    needs_review: bool

    def to_dict(self):
        return asdict(self)


def build_analyzed_shot(
    shot_id: str,
    source_file: str,
    source_path: str,
    duration_sec: float,
    vlm_description: Dict[str, Any],
    adapter_name: str,
    missing_fields: Optional[list] = None,
    cv_metadata: Optional[Dict[str, Any]] = None,
    **extra,
) -> Shot:
    """基于已有分析结果构建标准 Shot 对象

    extra 可传入 style / atmosphere / culture / tags / key_objects /
    is_long_take / coherence_score / framing / lighting / color_tone 等字段。
    """
    return Shot(
        shot_id=shot_id,
        state="ANALYZED",
        source_file=source_file,
        source_path=source_path,
        tc_in="00:00:00:00",
        tc_out="00:00:00:00",
        duration_sec=duration_sec,
        do_not_split=True,
        needs_review=True,
        location=vlm_description.get("location", ""),
        time_of_day=vlm_description.get("time_of_day", ""),
        characters=vlm_description.get("characters", []),
        action=vlm_description.get("action", ""),
        emotion=vlm_description.get("emotion", ""),
        dialogue=vlm_description.get("dialogue", ""),
        shot_size=vlm_description.get("shot_size", ""),
        camera_position=vlm_description.get("camera_position", ""),
        camera_movement=vlm_description.get("camera_movement", ""),
        framing=vlm_description.get("framing", extra.get("framing", "")),
        lighting=vlm_description.get("lighting", extra.get("lighting", "")),
        color_tone=vlm_description.get("color_tone", extra.get("color_tone", "")),
        style=extra.get("style", ""),
        atmosphere=extra.get("atmosphere", ""),
        culture=extra.get("culture", ""),
        tags=extra.get("tags", []),
        key_objects=extra.get("key_objects", []),
        is_long_take=extra.get("is_long_take", False),
        coherence_score=extra.get("coherence_score", 0.0),
        cv_metadata=cv_metadata,
        provenance=Provenance(
            state="ANALYZED",
            generated_by=adapter_name,
            conversion={
                "from_format": adapter_name,
                "adapter": adapter_name,
                "missing_fields": missing_fields or [],
                "needs_review": True,
            },
        ),
    )
