"""
计算机视觉与视频预扫描工具

职责：
- 视频物理属性提取（分辨率、帧率、码率、时长）
- 场景变化候选点检测
- 视觉质量估算（可选，依赖 OpenCV）
- 关键帧提取（不依赖模型配置，仅做基础操作）

保持与具体 AI 模型无关。
"""
import os
import re
import tempfile
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from src.utils import run_ffmpeg, run_ffprobe, logger, ensure_dir


def cv_pre_scan(video_path: str) -> Dict[str, Any]:
    """CV 预扫描：提取分辨率、帧率、码率、时长、候选切分点"""
    if not os.path.exists(video_path):
        raise ValueError(f"视频文件不存在: {video_path}")
    size = os.path.getsize(video_path)
    if size == 0:
        raise ValueError(f"视频文件大小为 0: {video_path}")

    try:
        probe = run_ffprobe(video_path)
    except Exception as e:
        logger.error(f"ffprobe 读取失败 ({video_path}): {e}")
        raise

    fmt = probe.get("format", {})
    streams = probe.get("streams", [])

    video_stream = None
    for s in streams:
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if not video_stream:
        logger.error(f"ffprobe 返回中无视频流: {video_path}\nstdout: {probe}")
        raise ValueError(f"无法解析视频流: {video_path} (文件大小 {size} bytes, 无 video stream)")

    width = int(video_stream.get("width", 1920))
    height = int(video_stream.get("height", 1080))
    aspect_ratio = f"{width}:{height}"

    r_frame_rate = video_stream.get("r_frame_rate", "24/1")
    num, den = r_frame_rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 24.0

    duration = float(fmt.get("duration", 0))
    bitrate = fmt.get("bit_rate")
    codec = video_stream.get("codec_name")

    # 候选切分点：基于 FFmpeg 场景检测
    candidates = detect_scene_candidates(video_path)

    # 计算简单的视觉质量：基于清晰度（拉普拉斯方差）
    visual_quality = estimate_visual_quality(video_path)

    return {
        "resolution": (width, height),
        "aspect_ratio": aspect_ratio,
        "fps": fps,
        "duration": duration,
        "bitrate": bitrate,
        "codec": codec,
        "visual_quality": visual_quality,
        "scene_change_candidates": candidates,
    }


def detect_scene_candidates(video_path: str, threshold: float = 0.1) -> List[float]:
    """检测高置信度场景变化候选点（兼容旧接口）

    基于 cut_score，只返回 score >= threshold 的时间戳。
    """
    frame_scores = compute_frame_scene_scores(video_path)
    return [t for t, cut_score, _ in frame_scores if cut_score >= threshold]


def _compute_frame_scene_scores_ffmpeg(video_path: str) -> List[Tuple[float, float]]:
    """OpenCV 不可用时，回退到 FFmpeg 直方图场景检测"""
    try:
        result = run_ffmpeg([
            "-i", video_path,
            "-vf", "select='gt(scene,0.0)',metadata=print:file=-",
            "-f", "null", "-",
        ], check=False)
    except Exception as e:
        logger.warning(f"FFmpeg scene score 计算失败 ({video_path}): {e}")
        return []

    if result.returncode != 0:
        return []

    frame_scores = []
    current_time = None
    for line in result.stdout.split("\n"):
        time_match = re.search(r"pts_time:([\d.]+)", line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        score_match = re.search(r"lavfi\.scene_score=([\d.]+(?:\.\d+)?)", line)
        if score_match and current_time is not None:
            score = float(score_match.group(1))
            frame_scores.append((current_time, score))
            current_time = None

    return frame_scores


def _frame_histogram_diff(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """灰度直方图差异，返回 0~1，越大表示两帧颜色分布差异越大"""
    import cv2
    hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
    hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
    cv2.normalize(hist_prev, hist_prev, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist_curr, hist_curr, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    corr = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
    # corr in [-1, 1], 1 = 完全相同
    return max(0.0, 1.0 - corr)


def _frame_edge_change_ratio(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """边缘变化率 ECR，返回 0~1

    对相机运动、主体运动更鲁棒，对镜头硬切更敏感。
    """
    import cv2
    edges_prev = cv2.Canny(prev_gray, 50, 150)
    edges_curr = cv2.Canny(curr_gray, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges_prev_dilated = cv2.dilate(edges_prev, kernel, iterations=2)
    edges_curr_dilated = cv2.dilate(edges_curr, kernel, iterations=2)

    prev_count = int(np.count_nonzero(edges_prev))
    curr_count = int(np.count_nonzero(edges_curr))
    if prev_count == 0 or curr_count == 0:
        return 0.0

    entering = int(np.count_nonzero(edges_curr & ~edges_prev_dilated))
    exiting = int(np.count_nonzero(edges_prev & ~edges_curr_dilated))

    ecr_in = entering / curr_count
    ecr_out = exiting / prev_count
    return max(ecr_in, ecr_out)


def _frame_motion_residual_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """全局运动补偿后的残差 score，返回 0~1

    思路：
    - 对相邻帧提取特征点并匹配；
    - 用 RANSAC 估计全局仿射变换（可覆盖平移/旋转/缩放/倾斜，即相机运动）；
    - 把 prev 按这个全局变换 warp 到 curr 的视角；
    - 计算补偿后的残差。

    真实硬切：特征点匹配失败或无法找到一致的全局变换，残差突增。
    普通相机运动/跟拍：仿射变换能基本对齐，残差很小。
    """
    import cv2
    # 在较小分辨率上估计全局运动，提速
    small_prev = cv2.resize(prev_gray, (320, 180))
    small_curr = cv2.resize(curr_gray, (320, 180))

    # 提取角点特征
    prev_pts = cv2.goodFeaturesToTrack(
        small_prev, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
    )
    if prev_pts is None or len(prev_pts) < 10:
        return _frame_histogram_diff(prev_gray, curr_gray)

    # 光流跟踪
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(small_prev, small_curr, prev_pts, None)
    if curr_pts is None:
        return _frame_histogram_diff(prev_gray, curr_gray)

    valid_prev = []
    valid_curr = []
    for i, st in enumerate(status):
        if st[0] == 1:
            valid_prev.append(prev_pts[i])
            valid_curr.append(curr_pts[i])

    if len(valid_prev) < 10:
        return _frame_histogram_diff(prev_gray, curr_gray)

    valid_prev = np.array(valid_prev, dtype=np.float32).reshape(-1, 2)
    valid_curr = np.array(valid_curr, dtype=np.float32).reshape(-1, 2)

    # 估计全局仿射变换
    affine, inliers = cv2.estimateAffine2D(
        valid_prev, valid_curr,
        method=cv2.RANSAC, ransacReprojThreshold=3.0,
    )
    if affine is None:
        # 找不到一致的全局运动，保守回退到直方图差异
        return _frame_histogram_diff(prev_gray, curr_gray)

    inlier_ratio = float(np.count_nonzero(inliers)) / len(inliers) if inliers is not None else 0.0
    if inlier_ratio < 0.5:
        # 多数特征点不服膺同一个全局变换，可能是切镜也可能是强动作，
        # 保守起见用直方图差异，避免动作戏持续误报 1.0
        return _frame_histogram_diff(prev_gray, curr_gray)

    # 把仿射矩阵缩放到原分辨率
    h, w = prev_gray.shape
    scale_x = w / small_prev.shape[1]
    scale_y = h / small_prev.shape[0]
    affine_scaled = affine.copy()
    affine_scaled[0, 2] *= scale_x
    affine_scaled[1, 2] *= scale_y

    warped_prev = cv2.warpAffine(prev_gray, affine_scaled, (w, h))

    # 像素残差
    residual = cv2.absdiff(curr_gray, warped_prev)
    pixel_score = float(np.mean(residual)) / 255.0

    # 边缘残差
    edges_curr = cv2.Canny(curr_gray, 50, 150)
    edges_warped = cv2.Canny(warped_prev, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_warped_dilated = cv2.dilate(edges_warped, kernel, iterations=1)
    curr_count = max(1, int(np.count_nonzero(edges_curr)))
    entering = int(np.count_nonzero(edges_curr & ~edges_warped_dilated))
    edge_score = entering / curr_count

    return max(pixel_score, edge_score)


def compute_frame_scene_scores(video_path: str) -> List[Tuple[float, float, float]]:
    """逐帧计算相邻帧的 scene score，返回 [(timestamp, cut_score, hist_score), ...]

    当前默认信号：
    - cut_score = hist_score = 灰度直方图差异（1 - HISTCMP_CORREL）

    说明：
    - 直方图差异对颜色/亮度突变敏感，对一般素材干净可用；
    - 已实验 ECR 与光流运动残差，但在密集动作戏中容易把持续运动误判为切镜，
      因此当前默认 scoring 以直方图为主；
    - 保留 (cut_score, hist_score) 三元组结构，方便后续实验其他组合特征。
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV 未安装，回退到 FFmpeg 直方图场景检测")
        return _compute_frame_scene_scores_ffmpeg(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"OpenCV 无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_scores: List[Tuple[float, float, float]] = []
    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 下采样到 640x360，兼顾速度与稳定性
        gray = cv2.resize(gray, (640, 360))

        if prev_gray is not None:
            hist_diff = _frame_histogram_diff(prev_gray, gray)
            # 当前以直方图作为主导信号；运动残差保留在函数内但暂不混入 cut_score，
            # 避免动作戏中运动补偿失败导致的持续误报。
            cut_score = hist_diff
            frame_scores.append((timestamp, cut_score, hist_diff))

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return frame_scores


def detect_scene_peaks(
    frame_scores: List[Tuple[float, float, float]],
    threshold: float = 0.3,
    merge_window: float = 0.3,
) -> List[float]:
    """从逐帧 scene score 中检测峰值切分候选点

    - 只保留 cut_score >= threshold 的点；
    - 在 merge_window 秒内的高分点会聚类，保留分数最高的那个作为峰值；
    - 返回按时间排序的峰值时间戳列表。
    """
    if not frame_scores:
        return []

    # 筛选超过阈值的点
    high_scores = [(t, cut_score) for t, cut_score, _ in frame_scores if cut_score >= threshold]
    if not high_scores:
        return []

    # 按时间排序
    high_scores.sort(key=lambda x: x[0])

    # 聚类：合并邻近峰值
    clusters = []
    for t, s in high_scores:
        if not clusters or t - clusters[-1][-1][0] > merge_window:
            clusters.append([(t, s)])
        else:
            clusters[-1].append((t, s))

    # 每个聚类取最高分点
    peaks = [max(cluster, key=lambda x: x[1])[0] for cluster in clusters]
    return sorted(peaks)


def classify_cut_candidate(
    timestamp: float,
    frame_scores: List[Tuple[float, float]],
    context: float = 0.2,
    calm_threshold: float = 0.15,
) -> str:
    """预判断候选切点类型

    - hard_cut: 候选点前 context 秒内和后 context 秒内都相对平和
                （最大 scene score < calm_threshold），说明是孤立硬切；
    - ambiguous: 候选点附近还有其他高分帧，可能是运动/特效/软过渡，
                 需要 VLM 进一步判断。
    """
    left_scores = [s for t, s in frame_scores if timestamp - context <= t < timestamp]
    right_scores = [s for t, s in frame_scores if timestamp < t <= timestamp + context]

    left_max = max(left_scores) if left_scores else 0.0
    right_max = max(right_scores) if right_scores else 0.0

    if left_max < calm_threshold and right_max < calm_threshold:
        return "hard_cut"
    return "ambiguous"


def estimate_visual_quality(video_path: str, sample_sec: float = 5.0) -> Optional[float]:
    """抽取一帧并估算清晰度（拉普拉斯方差），返回 1-5 评分"""
    try:
        import cv2
    except ImportError:
        logger.debug("OpenCV 未安装，跳过视觉质量估算")
        return None

    probe = run_ffprobe(video_path)
    duration = float(probe.get("format", {}).get("duration", 0))
    if duration <= 0:
        return None

    sample_time = min(sample_sec, duration / 2)
    temp_frame = os.path.join(tempfile.gettempdir(), "llm_autocut_quality_frame.jpg")
    try:
        run_ffmpeg([
            "-ss", str(sample_time),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            temp_frame,
        ])
        img = cv2.imread(temp_frame)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 映射到 1-5
        score = min(5.0, max(1.0, lap / 100.0))
        return round(score, 2)
    except Exception as e:
        logger.warning(f"视觉质量估算失败: {e}")
        return None
    finally:
        if os.path.exists(temp_frame):
            try:
                os.remove(temp_frame)
            except Exception:
                pass


def extract_keyframes(
    video_path: str,
    shot_id: str,
    tc_in: str,
    tc_out: str,
    fps: float,
    output_dir: str,
    strategy: str = "adaptive",
    interval: float = 2.0,
) -> List[str]:
    """为镜头按策略提取关键帧"""
    from src.utils import tc_to_sec, sec_to_tc

    start = tc_to_sec(tc_in, fps)
    end = tc_to_sec(tc_out, fps)
    duration = end - start

    if strategy == "adaptive":
        positions = [0.1, 0.5, 0.9]
    else:
        positions = []
        t = 0.0
        while t < duration:
            positions.append(t / duration if duration > 0 else 0.0)
            t += interval
        if not positions:
            positions = [0.5]

    ensure_dir(output_dir)
    keyframes = []
    for idx, pos in enumerate(positions):
        time_sec = start + duration * pos
        output_path = os.path.join(output_dir, f"{shot_id}_kf{idx}.jpg")
        try:
            run_ffmpeg([
                "-ss", str(time_sec),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                output_path,
            ])
            if os.path.exists(output_path):
                keyframes.append(output_path)
        except Exception as e:
            logger.warning(f"提取关键帧失败 {shot_id} pos={pos}: {e}")

    return keyframes


def split_video(video_path: str, output_path: str, start_sec: float, end_sec: float, copy: bool = True) -> str:
    """按时间范围切分视频，默认不重新编码"""
    duration = end_sec - start_sec
    cmd = ["-ss", str(start_sec), "-t", str(duration), "-i", video_path]
    if copy:
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac"]
    cmd.append(output_path)

    run_ffmpeg(cmd)
    return output_path
