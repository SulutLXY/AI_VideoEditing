# -*- coding: utf-8 -*-
"""存量镜头修复校验：检查每个镜头 frames.json 与其引用帧文件一致性，
thumbnails 是否指向 f_xxxx 序列，motion 是否存在；损坏的镜头重抽修复。"""
import glob
import json
import os
import sys
import io

import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.adaptive_frame_extractor import extract_adaptive_frames, EXTRACTION_VERSION

BASE = os.path.join(PROJECT_ROOT, "workspace", "output", "phase0_rough_clips")


def load_adaptive_config():
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f).get("phase0", {}).get("adaptive_frames", {})

def check_shot(shot_dir):
    """返回 (ok, reason)。ok=True 表示无需修复。"""
    sid = os.path.basename(shot_dir)
    mp4 = os.path.join(shot_dir, f"{sid}.mp4")
    frames_dir = os.path.join(shot_dir, "frames")
    frames_json = os.path.join(frames_dir, "frames.json")
    cfg_path = os.path.join(shot_dir, f"{sid}_config.json")

    if not os.path.exists(mp4):
        return True, "无 mp4（可能已清理），跳过"
    if not os.path.exists(frames_json):
        return False, "缺 frames.json"
    try:
        with open(frames_json, encoding="utf-8") as f:
            frames = json.load(f)
    except Exception as e:
        return False, f"frames.json 损坏: {e}"
    if not frames:
        return False, "frames.json 为空"
    missing = [fr["file"] for fr in frames
               if not os.path.exists(os.path.join(frames_dir, fr["file"]))]
    if missing:
        return False, f"缺 {len(missing)} 个帧文件"
    if not os.path.exists(os.path.join(frames_dir, "motion.json")):
        return False, "缺 motion.json"
    meta_path = os.path.join(frames_dir, "meta.json")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("extraction") != EXTRACTION_VERSION:
            return False, f"抽帧版本过旧: {meta.get('extraction', 'unknown')}"
    except Exception as e:
        return False, f"meta.json 损坏: {e}"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        thumbs = cfg.get("thumbnails", {})
        if not (thumbs.get("first", "").startswith("frames/f_")
                and thumbs.get("last", "").startswith("frames/f_")):
            return False, "thumbnails 未指向 f_xxxx 序列"
    except Exception as e:
        return False, f"config 损坏: {e}"
    return True, f"{len(frames)} 帧正常"

def repair(shot_dir, adaptive_cfg):
    sid = os.path.basename(shot_dir)
    mp4 = os.path.join(shot_dir, f"{sid}.mp4")
    frames_dir = os.path.join(shot_dir, "frames")
    cfg_path = os.path.join(shot_dir, f"{sid}_config.json")
    os.makedirs(frames_dir, exist_ok=True)
    # 全删重抽，保证产物一致
    for old in glob.glob(os.path.join(frames_dir, "f_*.jpg")):
        os.remove(old)
    for name in ["frames.json", "meta.json", "motion.json"]:
        p = os.path.join(frames_dir, name)
        if os.path.exists(p):
            os.remove(p)
    result = extract_adaptive_frames(mp4, frames_dir, adaptive_cfg)
    if not result:
        return False
    frames_info = result["frames"]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["thumbnails"] = {"first": f"frames/{frames_info[0]['file']}",
                         "last": f"frames/{frames_info[-1]['file']}"}
    cfg.setdefault("frames", {})["count"] = len(frames_info)
    cfg["motion"] = result["motion"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return True

def main():
    adaptive_cfg = load_adaptive_config()
    shot_dirs = sorted(d for d in glob.glob(os.path.join(BASE, "S*")) if os.path.isdir(d))
    print(f"共 {len(shot_dirs)} 个镜头文件夹")
    bad = []
    for d in shot_dirs:
        ok, reason = check_shot(d)
        if not ok:
            bad.append((d, reason))
            print(f"[BAD] {os.path.basename(d)}: {reason}")
    print(f"校验完成: 异常 {len(bad)} / {len(shot_dirs)}")
    repaired, failed = 0, 0
    for d, _ in bad:
        if "无 mp4" in _:
            continue
        if repair(d, adaptive_cfg):
            repaired += 1
            print(f"[REPAIRED] {os.path.basename(d)}")
        else:
            failed += 1
            print(f"[FAIL] {os.path.basename(d)}")
    print(f"修复完成: 成功 {repaired}, 失败 {failed}")

    # 抽查 S001
    s1 = os.path.join(BASE, "S001", "frames", "frames.json")
    if os.path.exists(s1):
        with open(s1, encoding="utf-8") as f:
            frames = json.load(f)
        tiers = {}
        for fr in frames:
            tiers[fr["tier"]] = tiers.get(fr["tier"], 0) + 1
        print(f"抽查 S001: {len(frames)} 帧, tier 分布={tiers}")

if __name__ == "__main__":
    main()
