"""
Phase 2 镜头锚定手动校正接口

提供两个核心功能：
1. export_anchor_corrections(output_dir, csv_path): 从 phase2_selected_shots.json 导出可编辑 CSV
2. apply_anchor_corrections(output_dir, csv_path): 读取用户编辑后的 CSV 并写回 JSON

CSV 字段：
- shot_id: 镜头 ID（只读）
- source_file: 源文件名（只读，辅助识别）
- action: Phase1 识别出的主体动作（只读）
- asr_text: ASR 台词（只读）
- current_beat: 当前锚定的 beat
- current_confidence: 当前置信度
- corrected_beat: 用户修正后的 beat（空表示不修改）
- corrected_confidence: 用户修正后的置信度（空表示不修改）
- corrected_function: 用户修正后的 function（空表示不修改）
- corrected_reasoning: 用户修正后的 reasoning（空表示不修改）
- notes: 用户备注（仅保存到 CSV，不写回 JSON）
"""
import csv
import os
from typing import List, Dict, Any

from src.utils import load_json, save_json, logger


ANCHOR_CSV_FIELDS = [
    "shot_id",
    "source_file",
    "action",
    "asr_text",
    "current_beat",
    "current_confidence",
    "corrected_beat",
    "corrected_confidence",
    "corrected_function",
    "corrected_reasoning",
    "notes",
]


def _get_anchor_value(shot: Dict[str, Any], key: str, default: Any = "") -> Any:
    anchor = shot.get("script_anchor") or {}
    return anchor.get(key, default)


def _collect_valid_beats(output_dir: str) -> set:
    """收集当前可用的 beat_id"""
    beats = set()
    analysis_path = os.path.join(output_dir, "script_beats_analysis.json")
    if os.path.exists(analysis_path):
        try:
            data = load_json(analysis_path)
            for b in data.get("beats", []):
                if b.get("beat_id"):
                    beats.add(b.get("beat_id"))
        except Exception:
            pass
    return beats


def export_anchor_corrections(output_dir: str, csv_path: str) -> str:
    """导出 phase2_selected_shots.json 中的锚定信息为可编辑 CSV"""
    shots_path = os.path.join(output_dir, "phase2_selected_shots.json")
    if not os.path.exists(shots_path):
        raise FileNotFoundError(f"找不到 Phase 2 镜头文件: {shots_path}")

    data = load_json(shots_path)
    shots = data.get("shots", data.get("selected_shots", []))
    if not shots:
        raise ValueError(f"{shots_path} 中没有任何镜头")

    valid_beats = _collect_valid_beats(output_dir)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ANCHOR_CSV_FIELDS)
        writer.writeheader()
        for shot in shots:
            current_beat = _get_anchor_value(shot, "beat", "UNMATCHED")
            current_confidence = _get_anchor_value(shot, "confidence", 0.0)
            writer.writerow({
                "shot_id": shot.get("shot_id", ""),
                "source_file": shot.get("source_file", ""),
                "action": shot.get("action", ""),
                "asr_text": shot.get("asr_text", ""),
                "current_beat": current_beat,
                "current_confidence": current_confidence,
                "corrected_beat": "",
                "corrected_confidence": "",
                "corrected_function": "",
                "corrected_reasoning": "",
                "notes": "",
            })

    # 同时生成一个 beat 对照参考文件
    ref_path = os.path.join(os.path.dirname(csv_path), "phase2_anchor_beat_reference.txt")
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write("可用 beat_id 列表（corrected_beat 只能填写以下值或 UNMATCHED）：\n\n")
        for beat_id in sorted(valid_beats):
            f.write(f"  {beat_id}\n")
        f.write("\nfunction 可选值：主镜头 / 反应镜头 / 插入镜头 / 过渡 / 动作细节 / 情绪特写 / 环境交代 / 对话镜头 / UNMATCHED\n")

    logger.info(f"已导出锚定校正表: {csv_path}")
    logger.info(f"已生成 beat 参考文件: {ref_path}")
    return csv_path


def _read_corrections(csv_path: str) -> List[Dict[str, str]]:
    """读取校正 CSV"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def apply_anchor_corrections(output_dir: str, csv_path: str) -> str:
    """读取用户编辑后的 CSV，将修正写回 phase2_selected_shots.json"""
    shots_path = os.path.join(output_dir, "phase2_selected_shots.json")
    if not os.path.exists(shots_path):
        raise FileNotFoundError(f"找不到 Phase 2 镜头文件: {shots_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到校正 CSV 文件: {csv_path}")

    data = load_json(shots_path)
    shots = data.get("shots", data.get("selected_shots", []))
    shot_map = {s.get("shot_id"): s for s in shots}

    corrections = _read_corrections(csv_path)
    changed = 0
    invalid = []

    for corr in corrections:
        shot_id = corr.get("shot_id", "").strip()
        if not shot_id or shot_id not in shot_map:
            continue

        shot = shot_map[shot_id]
        anchor = shot.get("script_anchor") or {}
        updated = False

        # corrected_beat
        new_beat = corr.get("corrected_beat", "").strip()
        if new_beat:
            anchor["beat"] = new_beat
            updated = True

        # corrected_confidence
        new_conf = corr.get("corrected_confidence", "").strip()
        if new_conf:
            try:
                anchor["confidence"] = float(new_conf)
                updated = True
            except ValueError:
                invalid.append(f"{shot_id}: corrected_confidence '{new_conf}' 不是数字")

        # corrected_function
        new_func = corr.get("corrected_function", "").strip()
        if new_func:
            anchor["function"] = new_func
            updated = True

        # corrected_reasoning
        new_reason = corr.get("corrected_reasoning", "").strip()
        if new_reason:
            anchor["reasoning"] = new_reason
            updated = True

        if updated:
            shot["script_anchor"] = anchor
            changed += 1

    # 保存回 JSON
    save_json(data, shots_path)
    logger.info(f"锚定校正已应用: 修改 {changed} 个镜头, 输出 {shots_path}")

    if invalid:
        logger.warning("以下校正项格式错误，已跳过:")
        for msg in invalid:
            logger.warning(f"  {msg}")

    # 生成一个应用报告
    report_path = os.path.join(output_dir, "phase2_anchor_correction_report.json")
    save_json({
        "csv_path": csv_path,
        "shots_path": shots_path,
        "changed_count": changed,
        "invalid_count": len(invalid),
        "invalid_items": invalid,
    }, report_path)
    logger.info(f"校正报告: {report_path}")

    return shots_path
