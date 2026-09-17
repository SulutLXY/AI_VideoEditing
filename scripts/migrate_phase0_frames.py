# -*- coding: utf-8 -*-
"""存量镜头修正 pass：清旧帧产物 -> 按新逻辑重抽 -> 修 config 的 thumbnails/frames/motion。"""
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

from src.adaptive_frame_extractor import extract_adaptive_frames

BASE = os.path.join(PROJECT_ROOT, "workspace", "output", "phase0_rough_clips")


def load_adaptive_config():
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f).get("phase0", {}).get("adaptive_frames", {})

def main():
    adaptive_cfg = load_adaptive_config()
    shot_dirs = sorted(
        d for d in glob.glob(os.path.join(BASE, "S*"))
        if os.path.isdir(d)
    )
    print(f"共 {len(shot_dirs)} 个镜头文件夹")

    ok, fail = 0, 0
    for shot_dir in shot_dirs:
        sid = os.path.basename(shot_dir)
        mp4 = os.path.join(shot_dir, f"{sid}.mp4")
        frames_dir = os.path.join(shot_dir, "frames")
        cfg_path = os.path.join(shot_dir, f"{sid}_config.json")

        # 1) 删除冗余缩略图（根目录 + frames/ 下）
        for junk in [f"{sid}_first.jpg", f"{sid}_last.jpg",
                     os.path.join("frames", "first.jpg"),
                     os.path.join("frames", "last.jpg")]:
            p = os.path.join(shot_dir, junk)
            if os.path.exists(p):
                os.remove(p)

        # 2) 清 frames/ 旧帧产物（保留 audio.wav / audio_profile.json）
        for old in glob.glob(os.path.join(frames_dir, "f_*.jpg")):
            os.remove(old)
        for name in ["frames.json", "meta.json", "motion.json"]:
            p = os.path.join(frames_dir, name)
            if os.path.exists(p):
                os.remove(p)

        # 3) 新逻辑重抽帧
        result = extract_adaptive_frames(mp4, frames_dir, adaptive_cfg)
        if not result:
            print(f"[FAIL] {sid}: 抽帧失败")
            fail += 1
            continue

        frames_info = result["frames"]
        n = len(frames_info)
        first_file = frames_info[0]["file"]
        last_file = frames_info[-1]["file"]

        # 4) 修 config：thumbnails 指向 f_xxxx 序列首尾，更新 frames.count 和 motion
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["thumbnails"] = {"first": f"frames/{first_file}", "last": f"frames/{last_file}"}
        cfg.setdefault("frames", {})["count"] = n
        cfg["motion"] = result["motion"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        ok += 1
        print(f"[OK] {sid}: {n} 帧, 首={first_file}, 尾={last_file}")

    print(f"\n完成: 成功 {ok}, 失败 {fail}")

    # 抽查 S001 帧数与 tier 分布
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
