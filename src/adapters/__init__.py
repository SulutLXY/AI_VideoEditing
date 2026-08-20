"""
Adapter 模块：将外部分析结果转换为统一的 Shot 结构

v0.2 新增：支持 ANALYZED 状态的素材，已有分析结果但格式不统一时，
通过 Adapter 读取并转换为标准 Shot 对象，不重新分析、不切分。
"""
import os
from typing import Optional

from src.adapters.base import BaseMetaAdapter, build_analyzed_shot


def get_adapter(format_name: str) -> BaseMetaAdapter:
    """根据格式名获取适配器"""
    from src.adapters.custom_v1 import CustomV1Adapter
    from src.adapters.csv_meta import CsvMetaAdapter
    from src.adapters.autocut_v1 import AutocutV1Adapter
    from src.adapters.generic_json import GenericJsonAdapter

    adapters = {
        "custom_v1": CustomV1Adapter(),
        "csv": CsvMetaAdapter(),
        "autocut_v1": AutocutV1Adapter(),
        "generic_json": GenericJsonAdapter(),
    }
    if format_name not in adapters:
        raise ValueError(f"未知分析结果格式: {format_name}，可用: {list(adapters.keys())}")
    return adapters[format_name]


def find_adapter_for_file(file_path: str) -> Optional[BaseMetaAdapter]:
    """根据文件内容自动探测适配器"""
    from src.adapters.custom_v1 import CustomV1Adapter
    from src.adapters.csv_meta import CsvMetaAdapter
    from src.adapters.autocut_v1 import AutocutV1Adapter
    from src.adapters.generic_json import GenericJsonAdapter

    # 优先使用专用适配器，generic_json 作为最后兜底
    candidates = [AutocutV1Adapter(), CustomV1Adapter(), CsvMetaAdapter(), GenericJsonAdapter()]
    for adapter in candidates:
        if adapter.can_read(file_path):
            return adapter
    return None
