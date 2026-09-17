"""
自适应参考帧抽取器（Phase 0 产物）

基于 Farneback 光流 + 灰度直方图相似度 + 频域纹理尺度，对单个粗剪片段做
局部变化速度分析，按片段内局部变化速度分五档自适应抽帧（非整镜头统一定档）。

定档管线（v4：统一速度尺度、受控事件抽帧、配置指纹缓存）：
1. 每帧算光流幅值（Farneback, 320p）与低频能量占比（FFT 半径<1/4 频谱能量，
   仅作纹理尺度启发式，不能保证对应真实景别）
2. 光流均值平滑 k=5（消重复帧逐帧交替）；低频长窗平滑 k=45（景别是慢变属性）
3. 景别折算：corrected = flow × clip(scale_a + scale_b×低频占比, 0.2, 2.0)
   ——特写位移虚高打折（S087 特写普通镜 raw 3.0 → 折算 1.0），
     全景/中景接近 1:1（S001 追逐 raw 3.7 → 折算 2.9）
4. 折算速度再均值平滑 k=15 定档（持续运动水平，窗口与滞回同尺度）：
   静止 <0.3 / 慢镜头 <0.7 / 普通镜 <2.0 / 快镜头 <5.5 / 特快 ≥5.5
5. 滞回：换档需持续 min_tier_hold 秒（候选档计时不因短暂回退重置）

抽帧触发（四层）：
- 换档瞬间立即取样（升档帧必须落帧，防滞回+间隔双丢变速过程）
- 内容事件：折算速度或相似度越阈值的起始沿取样，带冷却；记录 capture_reason
- 静止档：保底锚点 / 受控事件 / 与最近抽取点累积变化超阈值
- 各档按间隔抽帧（静止变化时+2s锚点 / 慢1.0s / 普通0.5s / 快0.3s / 特快0.1s），
  快/特快连续段设帧数上限

输出（480p jpg，落盘到调用方指定的帧目录，如 <镜头文件夹>/frames/）：
- f_%04d.jpg   帧图片（编号与 meta.json times 一一对应）
- frames.json  每帧 {index, t, file, sim, sim_norm, tier}
- meta.json    {source, duration, fps, times, frame_count, width, height}（兼容现有 loader）
- motion.json  {avg_flow, p90_flow, perceived_speed, low_freq, subject_speed, ...}

仅复用版本、配置、源文件指纹一致且引用文件完整的缓存；保证首尾覆盖。
"""
import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils import logger

# 档位名（五档，按运动速度升序）
TIER_STATIC = "static"
TIER_SLOW = "slow"
TIER_NORMAL = "normal"
TIER_FAST = "fast"
TIER_VERY_FAST = "very_fast"
TIER_ORDER = [TIER_STATIC, TIER_SLOW, TIER_NORMAL, TIER_FAST, TIER_VERY_FAST]
EXTRACTION_VERSION = "adaptive_v4"

DEFAULTS: Dict[str, Any] = {
    "analysis_width": 320,      # 运动分析宽度（Farneback/直方图在此尺寸计算）
    "frame_width": 854,         # 落盘帧宽度（16:9 下即 480p；竖屏按高 480 等比）
    "frame_height": 480,
    # ---- 分档阈值：折算感知速度主轴（px/帧 @analysis_width），相似度门槛 ----
    "static_sim": 0.92,         # 静止档相似度门槛（raw）
    "slow_sim": 0.75,           # 慢镜头档相似度门槛（raw）
    "normal_sim": 0.55,         # 普通镜档相似度门槛（raw）
    "fast_sim": 0.35,           # 快镜头档相似度下限（仅作档位门槛，不再触发特快）
    "static_flow": 0.3,         # 静止档折算速度上限
    "slow_flow": 0.7,           # 慢镜头档折算速度上限
    "normal_flow": 2.0,         # 普通镜档折算速度上限（特写普通镜折算值 ~1.0，留足余量）
    "fast_flow": 5.5,           # 快镜头档折算速度上限；≥此值判特快
    # ---- 景别折算：f = clip(scale_a + scale_b×低频占比, scale_f_min, scale_f_max) ----
    # 锚点标定：S087 特写(低频0.918,raw3.0)折扣→0.27 得普通档；
    #           S001 全景/中景(低频0.880,raw3.7)→0.78 得快档
    "scale_enabled": True,
    "scale_a": 12.57,
    "scale_b": -13.4,
    "scale_f_min": 0.2,
    "scale_f_max": 2.0,
    # ---- 平滑窗口（帧数）----
    "flow_smooth": 5,           # 光流均值平滑（消重复帧逐帧交替）
    "scale_smooth": 45,         # 低频占比长窗平滑（景别是慢变属性，~1.5s@30fps）
    "tier_smooth": 15,          # 折算速度定档平滑（~0.5s，与滞回同尺度）
    # ---- 快速推拉镜头旁路：景别折算不能抹掉真实的全局缩放峰值 ----
    "scale_motion_smooth": 7,   # 全局缩放强度平滑（约 0.23s@30fps）
    "scale_motion_trigger": 1.0,  # 边缘缩放位移达到此值时保留原始光流（px/frame）
    # ---- 内容爆变捕获（不改档位，命中即取样）----
    "burst_sim": 0.65,          # raw 相似度低于此值视为爆变帧（闪光/撞击/瞬移）
    # ---- 各档抽帧间隔（秒）----
    "anchor_interval": 2.0,     # 静止档保底锚点间隔（秒）
    "slow_interval": 1.0,       # 慢镜头档抽帧间隔（秒）
    "normal_interval": 0.5,     # 普通镜档抽帧间隔（秒；兼容旧键 medium_interval）
    "fast_interval": 0.3,       # 快镜头档抽帧间隔（秒）
    "very_fast_interval": 0.1,  # 特快档抽帧间隔（秒）
    # ---- 帧数上限（按连续同档段计数，换档重置）----
    "max_fast_frames": 30,      # 连续快镜头段抽帧上限
    "max_very_fast_frames": 50,  # 连续特快段抽帧上限
    # ---- 静止档显著变化检测 ----
    "static_change_sim": 0.86,  # 与最近抽取点的相似度低于此值抽帧（累积变化）
    "static_flow_trigger": 1.5,  # 静止档内光流瞬时超此值也抽帧（对象突然移动）
    # ---- 通用 ----
    "min_tier_hold": 0.3,       # 档位滞回：换档后最少维持秒数
    "very_fast_hold": 0.1,      # 特快通常很短，单独缩短进入滞回时间
    "min_select_gap": 0.1,      # 任意两次抽帧的最小间隔（秒）
    "quality": 90,
    "event_cooldown": 0.5,     # 连续内容事件的最短取样间隔
}


def cache_signature(video_path: str, cfg: Dict) -> str:
    """版本、有效参数、源文件变化均使缓存失效。"""
    stat = os.stat(video_path)
    payload = [EXTRACTION_VERSION, os.path.abspath(video_path), stat.st_size,
               stat.st_mtime_ns, cfg]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

# 各档抽帧间隔对应的配置键
_TIER_INTERVAL_KEY = {
    TIER_SLOW: "slow_interval",
    TIER_NORMAL: "normal_interval",
    TIER_FAST: "fast_interval",
    TIER_VERY_FAST: "very_fast_interval",
}
# 各档帧数上限对应的配置键（静止/慢镜头不设上限）
_TIER_CAP_KEY = {
    TIER_FAST: "max_fast_frames",
    TIER_VERY_FAST: "max_very_fast_frames",
}


def normalize_sims(sims: List[float]) -> List[float]:
    """片段内相似度鲁棒归一化：median→0.5，按 4*MAD 线性展开并截断到 [0,1]。

    全片几乎无变化（MAD≈0）时退化为全 0.5（普通档），避免除零和档位翻转。
    """
    if not sims:
        return []
    arr = np.asarray(sims, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(4.0 * mad, 1e-3)
    norm = 0.5 + (arr - med) / scale * 0.5
    return [float(min(1.0, max(0.0, v))) for v in norm]


def _median_smooth(vals: List[float], k: int = 5) -> List[float]:
    """中值平滑（窗口 k，奇数）。Farneback 光流在快速/重复帧画面上会出现
    高低交替的逐帧抖动，单帧抖动撑不过滞回会被全部抑制，导致整段误判静止。
    中值对交替脉冲稳健，且不会像均值那样抹掉真实的短促变速。"""
    n = len(vals)
    if n == 0 or k <= 1 or n < k:
        return list(vals)
    half = k // 2
    arr = np.asarray(vals, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(arr[lo:hi])
    return [float(v) for v in out]


def _mean_smooth(vals: List[float], k: int = 5) -> List[float]:
    """均值平滑（窗口 k）。用于光流消抖与折算速度稳定化；
    对重复帧画面的 0/9 交替抖动，均值比中值更有效（中值对 50% 占空比交替无解）。"""
    n = len(vals)
    if n == 0 or k <= 1:
        return list(vals)
    half = k // 2
    arr = np.asarray(vals, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return [float(v) for v in out]


def low_freq_ratio(gray: np.ndarray, radius: float = 0.25) -> float:
    """低频能量占比：频谱半径 < radius 的能量 / 总能量（0~1）。

    纹理统计会受到虚焦、背景和画风影响，不代表可靠的语义景别识别。
    """
    g = gray.astype(np.float32)
    g -= g.mean()
    f = np.fft.fftshift(np.fft.fft2(g))
    p = np.abs(f) ** 2
    h, w = p.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    return float(p[r < radius].sum() / (p.sum() + 1e-9))


def scale_factor(low_freq: float, cfg: Dict) -> float:
    """景别折算系数：特写（低频占比高）位移虚高打折，全景/远景接近 1:1 或放大。"""
    if not cfg.get("scale_enabled", True):
        return 1.0
    a = cfg.get("scale_a", 12.57)
    b = cfg.get("scale_b", -13.4)
    f_min = cfg.get("scale_f_min", 0.2)
    f_max = cfg.get("scale_f_max", 2.0)
    return min(f_max, max(f_min, a + b * low_freq))


def global_scale_motion(flow: np.ndarray, step: int = 8) -> float:
    """从稠密光流拟合全局相似变换，返回画面边缘的缩放位移（px/frame）。

    平移/跟随镜头的 scale 约为 1，结果接近 0；快速推拉或主体整体缩放时结果显著升高。
    RANSAC 会排除头发、衣摆等局部运动，避免 S087 这类特写普通动作触发旁路。
    """
    h, w = flow.shape[:2]
    if h < step * 3 or w < step * 3:
        return 0.0
    yy, xx = np.mgrid[step:h:step, step:w:step]
    src = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    offsets = flow[yy, xx].reshape(-1, 2).astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(
        src,
        src + offsets,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=500,
        confidence=0.95,
    )
    if matrix is None:
        return 0.0
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return abs(scale - 1.0) * min(w, h) / 2.0


def assign_tiers(
    times: List[float],
    sims: List[float],
    sims_norm: List[float],
    flows: List[float],
    cfg: Dict,
) -> List[str]:
    """给每帧定档（五档，含滞回）。输入 flows 为**景别折算后的感知速度**
    （调用方负责光流平滑 + 低频折算 + 定档平滑，见模块头文档）。

    感知速度为主轴、相似度为档位门槛：
    - 特快档：感知速度 >= fast_flow（5.5），仅持续运动水平触发；
      瞬时的内容爆变（闪光/撞击）不改档位，由抽取层的 burst_sim 捕获事件处理；
    - 静止档：< static_flow（0.3）且 sim >= static_sim（0.92）；
    - 慢镜头档：< slow_flow（0.7）且 sim >= slow_sim（0.75）；
    - 普通镜档：< normal_flow（2.0）且 sim >= normal_sim（0.55）；
    - 快镜头档：< fast_flow（5.5）且 sim >= fast_sim（0.35）；
    - 兜底：剩余情形归快镜头档。

    times/sims/sims_norm/flows 下标对齐（第 i 项为第 i+1 帧与前帧的比较值），
    返回长度与输入相同、与 times 一一对应的档位序列。
    """
    static_sim = cfg.get("static_sim", 0.92)
    slow_sim = cfg.get("slow_sim", 0.75)
    normal_sim = cfg.get("normal_sim", 0.55)
    fast_sim = cfg.get("fast_sim", 0.35)
    static_flow = cfg.get("static_flow", 0.3)
    slow_flow = cfg.get("slow_flow", 0.7)
    normal_flow = cfg.get("normal_flow", 2.0)
    fast_flow = cfg.get("fast_flow", 5.5)
    hold = cfg.get("min_tier_hold", 0.3)
    very_fast_hold = cfg.get("very_fast_hold", 0.1)

    tiers: List[str] = []
    if not times:
        return tiers

    def _want(s: float, sn: float, fl: float) -> str:
        # 特快仅由持续感知速度触发；sn 保留在签名中（frames.json 记录用），不参与定档
        if fl >= fast_flow:
            return TIER_VERY_FAST
        if fl < static_flow and s >= static_sim:
            return TIER_STATIC
        if fl < slow_flow and s >= slow_sim:
            return TIER_SLOW
        if fl < normal_flow and s >= normal_sim:
            return TIER_NORMAL
        if fl < fast_flow and s >= fast_sim:
            return TIER_FAST
        return TIER_FAST  # 兜底（理论不可达，防御性保留）

    current = _want(sims[0], sims_norm[0], flows[0])
    candidate: Optional[str] = None      # 候选档
    candidate_active = 0.0               # 候选档累计持续秒数（回到当前档时暂停，不重置）
    prev_t = times[0]
    for t, s, sn, fl in zip(times, sims, sims_norm, flows):
        dt = t - prev_t
        prev_t = t
        want = _want(s, sn, fl)
        if want == current:
            # 回到当前档：候选保留、计时暂停（连续暴力中的短暂凹陷不打断晋升进程）
            tiers.append(current)
            continue
        if candidate is None or candidate != want:
            candidate = want
            candidate_active = 0.0
            tiers.append(current)
            continue
        candidate_active += dt
        required_hold = very_fast_hold if want == TIER_VERY_FAST else hold
        if candidate_active >= required_hold:
            current = want
            candidate = None
            candidate_active = 0.0
        tiers.append(current)
    return tiers


def motion_summary(
    flows: List[float],
    tiers: List[str],
    cfg: Dict,
    perceived_flows: Optional[List[float]] = None,
    low_freqs: Optional[List[float]] = None,
) -> Dict:
    """由光流幅值与档位序列汇总运动档案（写入片段 config.json 的 motion 字段）。

    avg_flow/p90_flow 为原始光流（事实值）；subject_speed 按**折算感知速度**分档
    （与定档口径一致）；perceived_speed 为折算速度均值；low_freq 为低频占比中值（景别代理）。
    """
    arr = np.asarray(flows, dtype=np.float64) if flows else np.asarray([0.0])
    avg_flow = float(arr.mean())
    p90_flow = float(np.percentile(arr, 90))
    p_arr = (
        np.asarray(perceived_flows, dtype=np.float64)
        if perceived_flows else arr
    )
    perceived = float(p_arr.mean())
    low_freq = (
        round(float(np.median(np.asarray(low_freqs, dtype=np.float64))), 4)
        if low_freqs else None
    )
    static_f = cfg.get("static_flow", 0.3)
    slow_f = cfg.get("slow_flow", 0.7)
    normal_f = cfg.get("normal_flow", 2.0)
    fast_f = cfg.get("fast_flow", 5.5)
    if perceived < static_f:
        subject_speed = "static"
    elif perceived < slow_f:
        subject_speed = "slow"
    elif perceived < normal_f:
        subject_speed = "normal"
    elif perceived < fast_f:
        subject_speed = "fast"
    else:
        subject_speed = "very_fast"
    counts = {tier: 0 for tier in TIER_ORDER}
    for t in tiers:
        counts[t] = counts.get(t, 0) + 1
    dominant = max(counts, key=counts.get) if counts else TIER_NORMAL
    return {
        "avg_flow": round(avg_flow, 3),
        "p90_flow": round(p90_flow, 3),
        "perceived_speed": round(perceived, 3),
        "low_freq": low_freq,
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
    """对单个片段做运动分析 + 五档自适应抽帧落盘。

    首帧即 f_0000.jpg、尾帧即序列最后一帧（首尾帧天然在 f_xxxx 序列中，
    不再单独落盘缩略图）。返回 {"meta", "frames", "motion"}；
    失败返回 None（调用方回退现场抽帧）。已存在同版本 frames.json 时幂等返回；
    旧版产物会自动重建，避免代码升级后继续静默复用旧速度档位。
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    signature = cache_signature(video_path, cfg)
    os.makedirs(out_dir, exist_ok=True)
    frames_json_path = os.path.join(out_dir, "frames.json")
    if os.path.exists(frames_json_path):
        try:
            with open(frames_json_path, encoding="utf-8") as f:
                frames_info = json.load(f)
            meta_path = os.path.join(out_dir, "meta.json")
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if (meta.get("extraction") == EXTRACTION_VERSION
                    and meta.get("signature") == signature
                    and frames_info
                    and all(os.path.isfile(os.path.join(out_dir, x["file"])) for x in frames_info)
                    and os.path.isfile(os.path.join(out_dir, "motion.json"))):
                motion = {}
                cfg_path = os.path.join(out_dir, "motion.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, encoding="utf-8") as f:
                        motion = json.load(f)
                return {"meta": meta, "frames": frames_info, "motion": motion}
            logger.info(
                f"[AdaptiveFrames] 检测到旧版抽帧产物 "
                f"{meta.get('extraction', 'unknown')} 或参数变化，自动重建为 {EXTRACTION_VERSION}"
            )
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
    # 配置中的平滑窗口以 30fps 为基准，速度统一到 320 宽、30fps。
    def window(key, default):
        return max(1, int(round(cfg.get(key, default) * fps / 30))) | 1
    velocity_scale = fps / 30.0 * 320.0 / a_w
    a_h = max(1, round(src_h * a_w / src_w))
    f_w, f_h = _target_size(src_w, src_h, cfg["frame_width"], cfg["frame_height"])

    # ---------- 第一遍：顺序解码，算 sim/flow，帧以 jpeg 字节暂存内存 ----------
    records: List[Dict] = []  # {t, sim, flow, low_freq, scale_motion, jpg}
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
        scale_motion = 0.0
        if prev_gray is not None:
            c = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            sim = float(min(1.0, max(0.0, (c + 1.0) / 2.0)))
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, small, None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = float(mag.mean())
            scale_motion = global_scale_motion(flow)
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
            "low_freq": round(low_freq_ratio(small), 6),
            "scale_motion": round(scale_motion, 4),
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
    # 第 i 项 sim 描述第 i 帧相对前一帧的变化，档位序列与帧对齐（首帧给静止档）
    pair_times = [r["t"] for r in records[1:]]
    pair_flows_raw = [r["flow"] for r in records[1:]]
    pair_low_freqs_raw = [r["low_freq"] for r in records[1:]]
    pair_scale_motion_raw = [r["scale_motion"] for r in records[1:]]
    # 持续运动基线先按景别折算；快速推拉镜头走全局缩放旁路，保留原始峰值。
    flow_smoothed = _mean_smooth([x * velocity_scale for x in pair_flows_raw], window("flow_smooth", 5))
    low_freqs = _mean_smooth(pair_low_freqs_raw, window("scale_smooth", 45))
    corrected = [
        flow * scale_factor(low_freq, cfg)
        for flow, low_freq in zip(flow_smoothed, low_freqs)
    ]
    corrected = _mean_smooth(corrected, window("tier_smooth", 15))
    scale_motions = _mean_smooth(
        [x * velocity_scale for x in pair_scale_motion_raw], window("scale_motion_smooth", 7)
    )
    scale_trigger = float(cfg.get("scale_motion_trigger", 1.0))
    pair_flows = [
        max(perceived, raw) if scale_motion >= scale_trigger else perceived
        for perceived, raw, scale_motion in zip(corrected, flow_smoothed, scale_motions)
    ]
    pair_tiers = (
        assign_tiers(pair_times, sims[1:], sims_norm[1:], pair_flows, cfg)
        if len(records) > 1 else []
    )
    tiers = [pair_tiers[0] if pair_tiers else TIER_STATIC] + pair_tiers

    selected: List[Tuple[int, str]] = []  # (record 下标, 档位)
    last_sel_t = -1e9
    anchor_t = records[0]["t"]
    last_sel_hist = records[0]["hist"]  # 静止档"显著变化"= 与最近抽取点的累积变化
    tier_run = 0  # 连续同档段已抽帧数（换档重置；只有快/特快档设上限）
    prev_tier = TIER_NORMAL
    static_change = cfg["static_change_sim"]
    static_flow_trigger = cfg.get("static_flow_trigger", 1.5)
    burst_sim = cfg.get("burst_sim", 0.65)
    reasons = {0: "first"}
    last_event_t = -1e9
    event_active = False
    for i, r in enumerate(records):
        t, tier = r["t"], tiers[i]
        # 升档（变慢→变快）瞬间立即取样：变速点本身必须落帧，否则按旧档间隔
        # 会跳过整个变速过程（滞回导致的换档延迟 + 间隔门槛双重丢失）
        tier_up = tier != prev_tier and TIER_ORDER.index(tier) > TIER_ORDER.index(prev_tier)
        if tier != prev_tier:
            tier_run = 0
            prev_tier = tier
        if i == 0:
            selected.append((i, tier))
            last_sel_t = t
            continue
        gap = t - last_sel_t
        effective = pair_flows[i - 1] if pair_flows else 0.0
        active = effective >= static_flow_trigger or r["sim"] < burst_sim
        onset = active and not event_active
        event_active = active
        if gap < cfg["min_select_gap"]:
            continue
        take = False
        reason = "interval"
        if tier_up:
            take = True
            reason = "tier_up"
        elif onset and t - last_event_t >= cfg["event_cooldown"]:
            take = True
            reason = "event_onset"
        elif tier == TIER_STATIC:
            if t - anchor_t >= cfg["anchor_interval"]:
                take = True
                reason = "anchor"
            else:
                # 与最近抽取点比较（非相邻帧）：累积变化超阈值才算显著变化，
                # 避免相邻帧小抖动把静止档打成连拍
                c = cv2.compareHist(last_sel_hist, r["hist"], cv2.HISTCMP_CORREL)
                vs_last = float(min(1.0, max(0.0, (c + 1.0) / 2.0)))
                if vs_last < static_change and gap >= cfg["event_cooldown"]:
                    take = True
                    reason = "content_change"
        else:
            interval = cfg.get(_TIER_INTERVAL_KEY[tier], DEFAULTS[_TIER_INTERVAL_KEY[tier]])
            cap_key = _TIER_CAP_KEY.get(tier)
            cap = cfg.get(cap_key, DEFAULTS[cap_key]) if cap_key else None
            if gap >= interval and (cap is None or tier_run < cap):
                take = True
        if take:
            selected.append((i, tier))
            reasons[i] = reason
            if reason == "event_onset":
                last_event_t = t
            last_sel_t = t
            anchor_t = t
            last_sel_hist = r["hist"]
            tier_run += 1

    # 首尾覆盖是独立约束，不能因周期或段落封顶漏掉尾帧。
    if selected[-1][0] != len(records) - 1:
        selected.append((len(records) - 1, tiers[-1]))
        reasons[len(records) - 1] = "last"
    # ---------- 落盘 ----------
    # 旧版本可能比新版本抽出更多帧，先移除旧 f_*.jpg，避免目录残留孤儿文件。
    for old_name in os.listdir(out_dir):
        if old_name.startswith("f_") and old_name.endswith(".jpg"):
            os.remove(os.path.join(out_dir, old_name))
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
            "flow": records[i]["flow"],
            "perceived_speed": round(pair_flows[max(0, i - 1)], 4) if pair_flows else 0.0,
            "low_freq": records[i]["low_freq"],
            "scale_motion": records[i]["scale_motion"],
            "tier": tier,
            "capture_reason": reasons[i],
        })

    meta = {
        "source": os.path.abspath(video_path),
        "duration": round(records[-1]["t"], 3),
        "fps": round(fps, 3),
        "times": times,
        "frame_count": len(times),
        "width": f_w,
        "height": f_h,
        "extraction": EXTRACTION_VERSION,
        "signature": signature,
    }
    perceived_all = [pair_flows[0] if pair_flows else 0.0] + pair_flows
    motion = motion_summary(
        [r["flow"] for r in records],
        tiers,
        cfg,
        perceived_flows=perceived_all,
        low_freqs=[r["low_freq"] for r in records],
    )
    motion["p90_scale_motion"] = round(
        float(np.percentile(pair_scale_motion_raw, 90)) if pair_scale_motion_raw else 0.0,
        3,
    )
    motion["speed_unit"] = "px/frame@320width,30fps"
    motion["sampling_audit"] = {
        "selected_frames": len(selected),
        "events": sum(v == "event_onset" for v in reasons.values()),
        "first_last_covered": selected[0][0] == 0 and selected[-1][0] == len(records) - 1,
    }
    # 自动记录分档与连续速度冲突，供整批统计定位，不要求用户逐帧排查。
    conflicts = sum(tier == TIER_STATIC and speed >= cfg["normal_flow"]
                    for tier, speed in zip(tiers, perceived_all))
    motion["sampling_audit"]["static_high_speed_frames"] = conflicts
    motion["sampling_audit"]["semantic_classification"] = "heuristic"
    if conflicts:
        logger.warning(f"[AdaptiveFrames] {os.path.basename(video_path)}: "
                       f"{conflicts} 帧速度与滞回档位冲突，已使用受控事件采样")

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(frames_json_path, "w", encoding="utf-8") as f:
        json.dump(frames_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "motion.json"), "w", encoding="utf-8") as f:
        json.dump(motion, f, ensure_ascii=False, indent=2)

    counts = motion["tier_frame_counts"]
    logger.info(
        f"[AdaptiveFrames] {os.path.basename(video_path)}: "
        f"{len(times)} 帧 (静止{counts.get(TIER_STATIC, 0)}/"
        f"慢{counts.get(TIER_SLOW, 0)}/普通{counts.get(TIER_NORMAL, 0)}/"
        f"快{counts.get(TIER_FAST, 0)}/特快{counts.get(TIER_VERY_FAST, 0)}), "
        f"主体速度={motion['subject_speed']}, avg_flow={motion['avg_flow']}, "
        f"perceived={motion['perceived_speed']}"
    )
    return {"meta": meta, "frames": frames_info, "motion": motion}
