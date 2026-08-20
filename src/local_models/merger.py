"""
Merger: 整合三个引擎的结果，输出精简 JSON
"""
from typing import List, Dict
from datetime import datetime


def merge_video_result(
    filename: str,
    duration_sec: float,
    face_result: List[Dict],
    asr_result: Dict,
    vision_result: Dict,
    include_unknown: bool = True
) -> Dict:
    """
    合并单条视频的所有分析结果
    """
    # 人物列表（去重，过滤 unknown 如果不需要）
    persons = []
    for p in face_result:
        name = p["identity"]
        if name.startswith("unknown") and not include_unknown:
            continue
        persons.append(name)
    persons = list(dict.fromkeys(persons))  # 去重保序

    # 动作：Vision 的 actions + 语音存在时追加"说话"
    actions = list(vision_result.get("actions", []))
    if asr_result.get("has_speech") and "说话" not in actions:
        actions.append("说话")

    # 语音文本：取完整文本（去时间戳版）
    speech_text = asr_result.get("text", "")

    # 精简输出
    result = {
        "filename": filename,
        "duration_sec": round(duration_sec, 2),
        "persons": persons,
        "actions": actions,
        "speech": speech_text,
        "shot_info": {
            "shot_type": vision_result.get("shot_type", "无法判断"),
            "camera_movement": vision_result.get("camera_movement", "无法判断")
        },
        "scene": vision_result.get("scene", "")
    }
    return result


def build_batch_json(
    batch_id: str,
    video_results: List[Dict],
    reference_persons: List[str]
) -> Dict:
    """构建最终批次 JSON"""
    return {
        "batch_id": batch_id,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": len(video_results),
        "reference_persons": reference_persons,
        "videos": video_results
    }
