"""
自适应参考帧抽取器（Phase 0 产物）

基于 Farneback 光流 + 灰度直方图相似度，对单个粗剪片段做局部变化速度分析，
按片段内局部变化速度分三档自适应抽帧（非整镜头统一定档）：

- 静止档（raw 相似度 >= static_sim）：画面几乎不变，仅在显著变化时抽 1 帧，
  另每 anchor_interval 秒保底锚点 1 帧（防长固定镜头零帧）
- 中速档：每 medium_interval 秒抽 1 帧
- 快变化档（raw < fast_sim_abs 或归一化相似度 < fast_sim）：每 fast_interval 秒抽 1 帧，
  单段连续快变化封顶 max_fast_frames 帧（成本控制）

滞回：新档位需持续 min_tier_hold 秒才允许晋升换档，防止瞬时掉档抖动。
静止档用绝对阈值（固定镜头在任何片段里都判静止）；快变化档用"绝对 + 片段内相对"
双条件，既抓整段高速镜头，也抓中速片段内的局部快变化。
相似度归一化按片段自身分布做中位数基准鲁棒归一化（median→0.5、±4*MAD 展开），
0.70/0.90 只是默认档位阈值，可在 phase0.adaptive_frames 配置覆盖。

输出（480p jpg，落盘到 <片段>_frames/）：
- f_%04d.jpg   帧图片（编号与 meta.json times 一一对应）
- frames.json  每帧 {index, t, file, sim, sim_norm, tier}
- meta.json    {source, duration, fps, times, frame_count, width, height}（兼容现有 loader）

已存在 frames.json 时直接返回（幂等，支持中断续跑）。
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import logger

# 档位名
TIER_STATIC = "static"
TIER_MEDIUM = "medium"
TIER_FAST = "fast"

DEFAULTS: Dict[str, Any] = {
    "analysis_width": 320,      # 运动分析宽度（Farneback/直方图在此尺寸计算）
    "frame_width": 854,         # 落盘帧宽度（16:9 下即 480p；竖屏按高 480 等比）
    "frame_height": 480,
    "static_sim": 0.90,         # 静止档阈值（raw 相似度，绝对判定）
    "fast_sim": 0.70,           # 快变化档阈值（归一化相似度，片段内相对判定）
    "fast_sim_abs": 0.70,       # 快变化档阈值（raw 相似度，整段高速判定）
    "static_change_sim": 0.86,  # 静止档"显著变化"阈值：与最近抽取点的相似度低于此值抽帧
    "anchor_interval": 2.0,     # 静止档保底锚点间隔（秒）
    "medium_interval": 0.5,     # 中速档抽帧间隔（秒）
    "fast_interval": 0.1,       # 快变化档抽帧间隔（秒）
    "max_fast_frames": 10,      # 单段连续快变化抽帧上限
    "min_tier_hold": 0.3,       # 档位滞回：换档后最少维持秒数
    "min_select_gap": 0.1,      # 任意两次抽帧的最小间隔（秒）
    "slow_flow": 0.8,           # 主体速度档阈值（analysis_width 下平均光流幅值 px/帧）
    "fast_flow": 2.5,
    "quality": 90,
}


def normalize_sims(sims: List[float]) -> List[float]:
    """片段内相似度鲁棒归一化：median→0.5，按 4*MAD 线性展开并截断到 [0,1]。

    全片几乎无变化（MAD≈0）时退化为全 0.5（中速档），避免除零和档位翻转。
    """
    if not sims:
        return []
    arr = np.asarray(sims, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(4.0 * mad, 1e-3)
    norm = 0.5 + (arr - med) / scale * 0.5
    return [float(min(1.0, max(0.0, v))) for v in norm]


def assign_tiers(
    times: List[float],
    sims: List[float],
    sims_norm: List[float],
    cfg: Dict,
) -> List[str]:
    """给每帧定档（含滞回）。

    静止档用 raw 相似度绝对判定（固定镜头在任何片段里都判静止）；
    快变化档用"绝对 + 片段内相对"双条件（raw < fast_sim_abs 或
    归一化相似度 < fast_sim），抓整段高速与片段内局部快变化。

    times/sims/sims_norm 下标对齐（第 i 项为第 i+1 帧与前帧的比较值），
    返回长度与输入相同、与 times 一一对应的档位序列。
    """
    static_sim = cfg.get("static_sim", 0.90)
    fast_sim = cfg.get("fast_sim", 0.70)
    fast_sim_abs = cfg.get("fast_sim_abs", 0.70)
    hold = cfg.get("min_tier_hold", 0.3)

    tiers: List[str] = []
    if not times:
        return tiers

    def _want(s: float, sn: float) -> str:
        if s >= static_sim:
            return TIER_STATIC
        if sn < fast_sim or s < fast_sim_abs:
            return TIER_FAST
        return TIER_MEDIUM

    current = _want(sims[0], sims_norm[0])
    candidate: Optional[Tuple[str, float]] = None  # (候选档, 候选起始 t)
    for t, s, sn in zip(times, sims, sims_norm):
        want = _want(s, sn)
        if want == current:
            candidate = None
            tiers.append(current)
            continue
        # 新档位必须先持续 hold 秒才能晋升（滞回），防瞬时掉档抖动
        if candidate is None or candidate[0] != want:
            candidate = (want, t)
            tiers.append(current)
            continue
        if t - candidate[1] >= hold:
            current = want
            candidate = None
        tiers.append(current)
    return tiers


def motion_summary(
    flows: List[float],
    tiers: List[str],
    cfg: Dict,
) -> Dict:
    """由光流幅值与档位序列汇总运动档案（写入片段 config.json 的 motion 字段）。"""
    arr = np.asarray(flows, dtype=np.float64) if flows else np.asarray([0.0])
    avg_flow = float(arr.mean())
    p90_flow = float(np.percentile(arr, 90))
    slow_f = cfg.get("slow_flow", 0.8)
    fast_f = cfg.get("fast_flow", 2.5)
    if avg_flow < slow_f:
        subject_speed = "slow"
    elif avg_flow < fast_f:
        subject_speed = "medium"
    else:
        subject_speed = "fast"
    counts = {TIER_STATIC: 0, TIER_MEDIUM: 0, TIER_FAST: 0}
    for t in tiers:
        counts[t] = counts.get(t, 0) + 1
    dominant = max(counts, key=counts.get) if counts else TIER_MEDIUM
    return {
        "avg_flow": round(avg_flow, 3),
        "p90_flow": round(p90_flow, 3),
        "flow_unit": f"px/frame@{cfg.get('analysis_width', 320)}p(Farneback)",
        "subject_speed": subject_speed,
        "dominant_tier": dominant,
        "tier_frame_counts": counts,
    }


def _target_size(w: int, h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """等比缩放到盒内（与 VisionEngine._downscale 同规则）"""
    if w <= 0 or h <= 0:
        return box_w, box_h
    scale = min(box_w / w, box_h / h)
    if scale >= 1.0:
        return w, h
    return max(1, int(w * scale)), max(1, int(h * scale))


def extract_adaptive_frames(
    video_path: str,
    out_dir: str,
    cfg: Optional[Dict] = None,
) -> Optional[Dict]:
    """对单个片段做运动分析 + 三档自适应抽帧落盘。

    返回 {"meta": meta, "frames": frames_json, "motion": motion}；
    失败返回 None（调用方回退现场抽帧）。已存在 frames.json 时幂等返回。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    os.makedirs(out_dir, exist_ok=True)
    frames_json_path = os.path.join(out_dir, "frames.json")
    if os.path.exists(frames_json_path):
        try:
            with open(frames_json_path, encoding="utf-8") as f:
                frames_info = json.load(f)
            meta_path = os.path.join(out_dir, "meta.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            motion = {}
            cfg_path = os.path.join(out_dir, "motion.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as f:
                    motion = json.load(f)
            return {"meta": meta, "frames": frames_info, "motion": motion}
        except Exception:
            pass  # 档案损坏则重建

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"[AdaptiveFrames] 无法打开视频: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    a_w = cfg["analysis_width"]
    a_h = max(1, round(src_h * a_w / src_w))
    f_w, f_h = _target_size(src_w, src_h, cfg["frame_width"], cfg["frame_height"])

    # ---------- 第一遍：顺序解码，算 sim/flow，帧以 jpeg 字节暂存内存 ----------
    records: List[Dict] = []  # {t, sim, flow, jpg}
    prev_gray = None
    prev_hist = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = msec / 1000.0 if msec and msec > 0 else len(records) / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (a_w, a_h), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        sim = 1.0
        flow_mag = 0.0
        if prev_gray is not None:
            c = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            sim = float(min(1.0, max(0.0, (c + 1.0) / 2.0)))
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(mag.mean())
        prev_gray = small
        prev_hist = hist

        disp = frame
        if (src_w, src_h) != (f_w, f_h):
            disp = cv2.resize(frame, (f_w, f_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["quality"])])
        if not ok:
            continue
        records.append({
            "t": round(float(t), 3),
            "sim": round(sim, 4),
            "flow": round(flow_mag, 4),
            "jpg": buf.tobytes(),
            "hist": hist,  # 分析尺寸直方图，静止档显著变化检测用（内存态，不落盘）
        })
    cap.release()

    if not records:
        logger.warning(f"[AdaptiveFrames] 未解码到任何帧: {video_path}")
        return None

    # ---------- 归一化 + 定档 + 选帧 ----------
    sims = [r["sim"] for r in records]
    sims_norm = normalize_sims(sims)
    # 第 i 项 sim 描述第 i 帧相对前一帧的变化，档位序列与帧对齐（首帧给中速档）
    pair_times = [r["t"] for r in records[1:]]
    pair_tiers = (
        assign_tiers(pair_times, sims[1:], sims_norm[1:], cfg)
        if len(records) > 1 else []
    )
    tiers = [pair_tiers[0] if pair_tiers else TIER_STATIC] + pair_tiers

    selected: List[Tuple[int, str]] = []  # (record 下标, 档位)
    last_sel_t = -1e9
    anchor_t = records[0]["t"]
    last_sel_hist = records[0]["hist"]  # 静止档"显著变化"= 与最近抽取点的累积变化
    fast_run = 0
    prev_tier = TIER_MEDIUM
    static_change = cfg["static_change_sim"]
    for i, r in enumerate(records):
        t, tier = r["t"], tiers[i]
        if tier != prev_tier:
            if prev_tier == TIER_FAST:
                fast_run = 0
            prev_tier = tier
        if i == 0:
            selected.append((i, tier))
            last_sel_t = t
            continue
        gap = t - last_sel_t
        if gap < cfg["min_select_gap"]:
            continue
        take = False
        if tier == TIER_STATIC:
            if t - anchor_t >= cfg["anchor_interval"]:
                take = True
            else:
                # 与最近抽取点比较（非相邻帧）：累积变化超阈值才算显著变化，
                # 避免相邻帧小抖动把静止档打成连拍
                c = cv2.compareHist(last_sel_hist, r["hist"], cv2.HISTCMP_CORREL)
                vs_last = float(min(1.0, max(0.0, (c + 1.0) / 2.0)))
                if vs_last < static_change:
                    take = True
        elif tier == TIER_MEDIUM:
            if gap >= cfg["medium_interval"]:
                take = True
        else:  # FAST
            if gap >= cfg["fast_interval"] and fast_run < cfg["max_fast_frames"]:
                take = True
        if take:
            selected.append((i, tier))
            last_sel_t = t
            anchor_t = t
            last_sel_hist = r["hist"]
            if tier == TIER_FAST:
                fast_run += 1

    # ---------- 落盘 ----------
    times: List[float] = []
    frames_info: List[Dict] = []
    for k, (i, tier) in enumerate(selected):
        name = f"f_{k:04d}.jpg"
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(records[i]["jpg"])
        times.append(records[i]["t"])
        frames_info.append({
            "index": i,
            "t": records[i]["t"],
            "file": name,
            "sim": records[i]["sim"],
            "sim_norm": round(sims_norm[i], 4),
            "tier": tier,
        })

    meta = {
        "source": os.path.abspath(video_path),
        "duration": round(records[-1]["t"], 3),
        "fps": round(fps, 3),
        "times": times,
        "frame_count": len(times),
        "width": f_w,
        "height": f_h,
        "extraction": "adaptive_v1",
    }
    motion = motion_summary([r["flow"] for r in records], tiers, cfg)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(frames_json_path, "w", encoding="utf-8") as f:
        json.dump(frames_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "motion.json"), "w", encoding="utf-8") as f:
        json.dump(motion, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[AdaptiveFrames] {os.path.basename(video_path)}: "
        f"{len(times)} 帧 (静止{motion['tier_frame_counts'].get(TIER_STATIC, 0)}/"
        f"中速{motion['tier_frame_counts'].get(TIER_MEDIUM, 0)}/"
        f"快速{motion['tier_frame_counts'].get(TIER_FAST, 0)}), "
        f"主体速度={motion['subject_speed']}, avg_flow={motion['avg_flow']}"
    )
    return {"meta": meta, "frames": frames_info, "motion": motion}
