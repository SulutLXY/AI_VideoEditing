"""
Phase 0: 纯 CV 粗剪（v0.2 定版）

职责：
- 对 RAW 原始素材做镜头级粗切分。
- 不调用 VLM/LLM，不调用 ASR，不建立关系图。
- 基于 OpenCV 逐帧灰度直方图差异 + 孤立峰值分析判断硬切点。
- 对持续高活动区域（运动/叠化/淡入淡出/高速镜头）做软转场保护，不切分。
- 输出切好的片段文件和粗配置文件。

积分规则（v0.2 定版）：
- 逐帧计算相邻帧灰度直方图差异，得到 cut_score（0~1）。
- 孤立峰值判断（硬切）：
  - 某帧为 ±0.1 秒窗口内的局部最大值；
  - 且比 ±0.1~±0.5 秒上下文均值高出 prominence >= 0.25；
  - 且该点不在持续高活动区域内部。
- 持续高活动判断（运动/软转场）：
  - 连续高分帧（score >= 全局 25 分位数 + 0.05）持续 >= 0.6 秒；
  - 视为运动/叠化/淡入淡出/高速镜头，记录但不切分。
- 邻近硬切点（<= 0.3 秒）合并，保留分数最高者。
- 超过 3.5 秒的片段用 0.10 低阈值做二次检测，避免漏拆弱切镜。
- 片段时长低于 min_shot_duration 时合并。
"""
import os
from typing import List, Dict, Any, Tuple

from src.cv_utils import (
    cv_pre_scan,
    compute_frame_scene_scores,
    split_video,
)
from src.models import Shot, Provenance, Relationship, Relationships
from src.utils import logger, ensure_dir, sec_to_tc, tc_to_sec, save_json


class RoughCutAnalyzer:
    """Phase 0 粗剪分析器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.paths = config.get("paths", {})
        self.processing = config.get("processing", {})
        self.split_scoring = config.get("split_scoring", {})

        self.output_dir = self.paths.get("output", "./output")
        self.rough_clips_dir = os.path.join(self.output_dir, "phase0_rough_clips")
        ensure_dir(self.rough_clips_dir)

        # 从配置读取 Phase 0 参数，未配置时使用当前定版默认值
        phase0_cfg = config.get("phase0", {})
        self.hard_cut_window = phase0_cfg.get("hard_cut_window", 0.1)
        self.hard_cut_merge_window = phase0_cfg.get("hard_cut_merge_window", 0.3)
        self.prominence_min = phase0_cfg.get("prominence_min", 0.25)
        self.long_segment_threshold = phase0_cfg.get("long_segment_threshold", 3.5)
        self.long_segment_prominence = phase0_cfg.get("long_segment_prominence", 0.10)
        self.baseline_delta = phase0_cfg.get("baseline_delta", 0.05)
        self.soft_transition_gap_tol = phase0_cfg.get("soft_transition_gap_tol", 0.15)
        self.soft_transition_min_duration = phase0_cfg.get("soft_transition_min_duration", 0.6)
        self.min_shot_duration = self.processing.get("min_shot_duration", 1.0)

        self._shot_counter = 0

    def _next_shot_id(self) -> str:
        self._shot_counter += 1
        return f"S{self._shot_counter:03d}"

    def run(self, video_paths: List[str]) -> List[Shot]:
        """执行 Phase 0 粗剪"""
        logger.info("=" * 60)
        logger.info("Phase 0: 纯 CV 粗剪")
        logger.info("=" * 60)

        all_shots: List[Shot] = []
        for video_path in video_paths:
            shots = self._process_video(video_path)
            all_shots.extend(shots)

        # 建立同素材片段之间的前后关系
        self._link_shot_relationships(all_shots)

        # 汇总素材级信息
        assets = self._build_assets_summary(all_shots)

        # 保存粗配置文件
        rough_config = {
            "project": self.config.get("project", {}),
            "total_shots": len(all_shots),
            "assets": assets,
            "shots": [shot.to_dict() for shot in all_shots],
        }
        config_path = os.path.join(self.output_dir, "phase0_rough_config.json")
        save_json(rough_config, config_path)

        logger.info(f"Phase 0 完成: 共生成 {len(all_shots)} 个粗剪片段")
        logger.info(f"  粗配置文件: {config_path}")
        logger.info(f"  片段目录: {self.rough_clips_dir}")
        return all_shots

    def _process_video(self, video_path: str) -> List[Shot]:
        """处理单个 RAW 视频"""
        filename = os.path.basename(video_path)
        logger.info(f"[Phase 0] 粗剪: {filename}")

        cv_meta = cv_pre_scan(video_path)
        fps = cv_meta.get("fps", 24.0)
        duration = cv_meta.get("duration", 0.0)
        logger.info(f"  CV 预扫描: 时长={duration:.2f}s, fps={fps}, 分辨率={cv_meta.get('resolution')}")

        if duration <= 0:
            logger.warning(f"  视频时长为 0，跳过: {filename}")
            return []

        # 1. 逐帧 scene score: (timestamp, cut_score, hist_score)
        frame_scores = compute_frame_scene_scores(video_path)
        if not frame_scores:
            logger.warning(f"  未检测到 scene score，按单个片段输出: {filename}")
            return [self._build_whole_shot(video_path, cv_meta, duration, [])]

        logger.info(f"  帧级 scene score: {len(frame_scores)} 个点（OpenCV 灰度直方图差异）")

        # 2. 帧级孤立峰值分析：区分硬切 / 运动 / 软转场
        hard_cuts, soft_transitions = self._analyze_samples(frame_scores)
        logger.info(f"  硬切点: {hard_cuts}")
        if soft_transitions:
            logger.info(f"  软转场标记: {len(soft_transitions)} 个")

        # 4. 构建切分时间线并合并过短片段
        cut_times = self._build_cut_times(hard_cuts, duration)
        logger.info(f"  最终切分点: {cut_times}, 生成 {len(cut_times) - 1} 个片段")

        # 5. 切分物理片段并生成 Shot
        shots = self._split_and_build_shots(video_path, cv_meta, cut_times, soft_transitions)
        return shots

    def _analyze_samples(
        self,
        frame_scores: List[Tuple[float, float, float]],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """基于帧级 scene score 的孤立峰值分析，返回硬切点和软转场区域

        frame_scores 元素为 (timestamp, cut_score, hist_score)：
        - cut_score 用于硬切检测（组合了直方图 + 光流残差 + ECR）
        - hist_score 用于软转场/运动区域检测
        """
        hard_cuts: List[float] = []
        soft_transitions: List[Dict[str, Any]] = []

        if not frame_scores:
            return hard_cuts, soft_transitions

        # 全局基线：基于 cut_score 的 25 分位数，避免局部平台被自身抬高
        sorted_cut_scores = sorted(cut_score for _, cut_score, _ in frame_scores)
        global_baseline = sorted_cut_scores[int(len(sorted_cut_scores) * 0.25)] if sorted_cut_scores else 0.0
        high_threshold = global_baseline + self.baseline_delta

        # 1. 先识别持续高活动区域：运动 / 叠化 / 淡入淡出 / 高速镜头
        # 用 cut_score，因为它同时包含直方图和运动结构信息，能捕捉动作戏的持续高活动
        high_points = [(t, cut_score) for t, cut_score, _ in frame_scores if cut_score >= high_threshold]
        soft_regions = self._merge_points_into_regions(
            high_points,
            gap_tol=self.soft_transition_gap_tol,
            min_duration=self.soft_transition_min_duration,
        )

        # 2. 在软转场区域外，用 cut_score 找孤立峰值作为硬切候选
        # 硬切判断：局部最大（±0.1s），prominence 用 ±0.5s 上下文基线，
        # 更能区分孤立硬切和持续运动 plateau。
        candidates: List[Tuple[float, float, float]] = []
        for t, cut_score, _ in frame_scores:
            # 跳过软转场区域内部（留出 0.05s 边界余量，避免在斜坡起点误判）
            if self._inside_any_region(t, soft_regions, inner_margin=0.05):
                continue

            # 局部最大值窗口：±0.1 秒
            local_window = [(t2, s2) for t2, s2, _ in frame_scores if abs(t2 - t) <= 0.1]
            if len(local_window) < 3:
                continue

            max_point = max(local_window, key=lambda x: x[1])
            if max_point[0] != t:
                continue

            # 上下文基线：±0.5 秒窗口，排除 ±0.1s 近邻，避免峰值自身拉高基线
            context_window = [
                (t2, s2) for t2, s2, _ in frame_scores
                if 0.1 < abs(t2 - t) <= 0.5
            ]
            if len(context_window) < 3:
                # 上下文不足，用 ±0.1s 内其他点兜底
                context_window = [(t2, s2) for t2, s2 in local_window if abs(t2 - t) > 1e-6]

            if not context_window:
                continue
            local_baseline = sum(s2 for _, s2 in context_window) / len(context_window)
            prominence = cut_score - local_baseline

            if prominence >= self.prominence_min:
                candidates.append((t, cut_score, prominence))
                logger.info(f"  硬切候选: {t:.2f}s (prominence={prominence:.3f})")

        # 3. 合并邻近硬切候选点，每个簇保留分数最高者
        hard_cuts = self._merge_hard_candidates(candidates, merge_window=self.hard_cut_merge_window)

        # 4. 长片段二次检测：对第一段 pass 后仍超过阈值的片段，
        #    用更低的 prominence 阈值寻找弱但可能是真实切镜的峰值。
        duration = frame_scores[-1][0] if frame_scores else 0.0
        hard_cuts = self._refine_long_segments(
            frame_scores,
            hard_cuts,
            duration,
            soft_regions,
        )

        # 5. 格式化软转场输出
        for start, end, _ in soft_regions:
            soft_transitions.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "type": "soft_transition",
                "confidence": "medium",
                "note": f"在 {start:.2f}s - {end:.2f}s 检测到持续画面变动（运动/叠化/淡入淡出/高速镜头），Phase 0 不切分",
            })
            logger.info(f"  软转场/运动: {start:.2f}s - {end:.2f}s")

        return sorted(hard_cuts), soft_transitions

    def _refine_long_segments(
        self,
        frame_scores: List[Tuple[float, float, float]],
        hard_cuts: List[float],
        duration: float,
        soft_regions: List[Tuple[float, float, List[float]]],
    ) -> List[float]:
        """对过长片段做第二轮低阈值峰值检测，避免漏掉弱真实切镜"""
        if not frame_scores or not hard_cuts:
            return hard_cuts

        cut_times = sorted(set([0.0] + hard_cuts + [duration]))
        extra_cuts: List[Tuple[float, float]] = []

        for i in range(len(cut_times) - 1):
            seg_start, seg_end = cut_times[i], cut_times[i + 1]
            if seg_end - seg_start <= self.long_segment_threshold:
                continue

            # 只在该片段内找峰值
            segment_scores = [
                (t, s) for t, s, _ in frame_scores
                if seg_start - 0.05 <= t <= seg_end + 0.05
            ]
            if not segment_scores:
                continue

            best_t = None
            best_prom = 0.0
            for t, s in segment_scores:
                # 跳过软转场区域
                if self._inside_any_region(t, soft_regions, inner_margin=0.05):
                    continue

                # 跳过与现有硬切点过近的位置，避免边界重复
                if any(abs(t - hc) <= self.hard_cut_merge_window for hc in hard_cuts):
                    continue

                # 局部最大 ±0.1s
                local_window = [
                    (t2, s2) for t2, s2 in segment_scores if abs(t2 - t) <= 0.1
                ]
                if len(local_window) < 3:
                    continue
                if max(local_window, key=lambda x: x[1])[0] != t:
                    continue

                # 上下文基线 ±0.1s~±0.5s
                context_window = [
                    (t2, s2) for t2, s2 in segment_scores
                    if 0.1 < abs(t2 - t) <= 0.5
                ]
                if len(context_window) < 3:
                    continue

                baseline = sum(s2 for _, s2 in context_window) / len(context_window)
                prominence = s - baseline

                if prominence >= self.long_segment_prominence and prominence > best_prom:
                    best_t = t
                    best_prom = prominence

            if best_t is not None:
                extra_cuts.append((best_t, best_prom))
                logger.info(f"  长片段二次检测切点: {best_t:.2f}s (prominence={best_prom:.3f})")

        # 合并原始切点和额外切点，再过滤一次邻近合并
        all_candidates = [(t, 0.0, 0.0) for t in hard_cuts] + [(t, 0.0, 0.0) for t, _ in extra_cuts]
        merged = self._merge_hard_candidates(all_candidates, merge_window=self.hard_cut_merge_window)
        return sorted(merged)

    @staticmethod
    def _merge_points_into_regions(
        points: List[Tuple[float, float]],
        gap_tol: float,
        min_duration: float,
    ) -> List[Tuple[float, float, List[float]]]:
        """把相邻高分点合并为连续区域，过滤掉过短的噪声"""
        if not points:
            return []

        points = sorted(points, key=lambda x: x[0])
        regions = []
        start = points[0][0]
        end = points[0][0]
        scores = [points[0][1]]

        for t, s in points[1:]:
            # 加极小 epsilon，避免浮点误差导致 0.1 秒间隔被判定为超出容差
            if t - end <= gap_tol + 1e-6:
                end = t
                scores.append(s)
            else:
                if end - start >= min_duration - 1e-6:
                    regions.append((start, end, scores))
                start = t
                end = t
                scores = [s]

        if end - start >= min_duration - 1e-6:
            regions.append((start, end, scores))

        return regions

    @staticmethod
    def _inside_any_region(
        t: float,
        regions: List[Tuple[float, float, List[float]]],
        inner_margin: float = 0.0,
    ) -> bool:
        """判断时间 t 是否落在某个区域内部（留出边界余量）"""
        for start, end, _ in regions:
            if start + inner_margin <= t <= end - inner_margin:
                return True
        return False

    @staticmethod
    def _merge_hard_candidates(
        candidates: List[Tuple[float, float, float]],
        merge_window: float,
    ) -> List[float]:
        """合并邻近硬切候选点，每个簇保留 scene score 最高的时间戳"""
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda x: x[0])
        merged = []
        cluster = [candidates[0]]

        for c in candidates[1:]:
            if c[0] - cluster[-1][0] <= merge_window:
                cluster.append(c)
            else:
                best = max(cluster, key=lambda x: x[1])
                merged.append(best[0])
                cluster = [c]

        best = max(cluster, key=lambda x: x[1])
        merged.append(best[0])

        return sorted(set(merged))

    def _build_cut_times(self, hard_cuts: List[float], duration: float) -> List[float]:
        """构建完整切分点时间线，并合并过短片段"""
        cut_times = sorted(set([0.0] + hard_cuts + [duration]))

        # 合并过短片段
        merged = [cut_times[0]]
        for i in range(1, len(cut_times) - 1):
            if cut_times[i] - merged[-1] < self.min_shot_duration:
                continue
            merged.append(cut_times[i])
        merged.append(cut_times[-1])

        # 最后一段也要满足最小时长，否则并入前一段
        if len(merged) >= 3 and merged[-1] - merged[-2] < self.min_shot_duration:
            merged.pop(-2)

        return merged

    def _split_and_build_shots(
        self,
        video_path: str,
        cv_meta: Dict[str, Any],
        cut_times: List[float],
        soft_transitions: List[Dict[str, Any]],
    ) -> List[Shot]:
        """按切点时间线切分视频并生成 Shot，同时注入落在片段内的软转场标记"""
        filename = os.path.basename(video_path)
        fps = cv_meta.get("fps", 24.0)
        shots = []

        for i in range(len(cut_times) - 1):
            start = cut_times[i]
            end = cut_times[i + 1]
            shot_id = self._next_shot_id()

            # FFmpeg 切分
            ext = os.path.splitext(video_path)[1] or ".mp4"
            clip_path = os.path.join(self.rough_clips_dir, f"{shot_id}{ext}")
            try:
                split_video(video_path, clip_path, start, end, copy=True)
            except Exception as e:
                logger.warning(f"FFmpeg 粗剪失败 {shot_id}: {e}")
                clip_path = ""

            # 收集落在当前片段内的软转场标记
            shot_soft_transitions = [
                st for st in soft_transitions
                if st["start"] >= start - 0.05 and st["end"] <= end + 0.05
            ]

            shot = Shot(
                shot_id=shot_id,
                state="RAW",
                source_file=filename,
                source_path=os.path.abspath(video_path),
                tc_in=sec_to_tc(start, fps),
                tc_out=sec_to_tc(end, fps),
                duration_sec=round(end - start, 3),
                fps=fps,
                resolution=cv_meta.get("resolution", (1920, 1080)),
                aspect_ratio=cv_meta.get("aspect_ratio", "16:9"),
                bitrate=cv_meta.get("bitrate"),
                codec=cv_meta.get("codec"),
                visual_quality=cv_meta.get("visual_quality"),
                cv_metadata=cv_meta,
                soft_transitions=shot_soft_transitions,
                needs_review=True,
                provenance=Provenance(
                    state="RAW",
                    generated_by="phase0_rough_cut",
                    split_decision={
                        "start": start,
                        "end": end,
                        "score": 0.0,
                        "reason": f"Phase 0 CV 粗剪: {start:.2f}s - {end:.2f}s",
                    },
                ),
            )
            shots.append(shot)

        return shots

    @staticmethod
    def _link_shot_relationships(shots: List[Shot]) -> None:
        """为同一段原始素材拆出的粗剪片段建立前后关系"""
        # 按 source_file 分组，组内按 tc_in 时间排序
        groups: Dict[str, List[Shot]] = {}
        for shot in shots:
            groups.setdefault(shot.source_file, []).append(shot)

        for source_file, group in groups.items():
            # 按时间码对应的秒数排序（tc_in 格式 HH:MM:SS:FF，可用 duration_sec 近似）
            group.sort(key=lambda s: s.duration_sec if s.tc_in == "00:00:00:00" else 0)
            # 更稳妥：解析 tc_in 为秒数
            from src.utils import tc_to_sec
            group.sort(key=lambda s: tc_to_sec(s.tc_in, s.fps))

            for i in range(len(group)):
                current = group[i]
                if i > 0:
                    prev = group[i - 1]
                    current.relationships.prev = Relationship(
                        shot_id=prev.shot_id,
                        relationship_type="同素材时间连续",
                        coherence_score=0.95,
                    )
                if i < len(group) - 1:
                    nxt = group[i + 1]
                    current.relationships.next = Relationship(
                        shot_id=nxt.shot_id,
                        relationship_type="同素材时间连续",
                        coherence_score=0.95,
                    )

    @staticmethod
    def _build_assets_summary(shots: List[Shot]) -> Dict[str, Any]:
        """汇总每个原始素材的基础信息和其子片段"""
        assets: Dict[str, Any] = {}
        for shot in shots:
            source_file = shot.source_file
            if source_file not in assets:
                assets[source_file] = {
                    "source_path": shot.source_path,
                    "resolution": shot.resolution,
                    "fps": shot.fps,
                    "aspect_ratio": shot.aspect_ratio,
                    "bitrate": shot.bitrate,
                    "codec": shot.codec,
                    "visual_quality": shot.visual_quality,
                    "duration_sec": 0.0,
                    "shot_ids": [],
                }
            assets[source_file]["shot_ids"].append(shot.shot_id)
            assets[source_file]["duration_sec"] += shot.duration_sec
        return assets

    def _build_whole_shot(
        self,
        video_path: str,
        cv_meta: Dict[str, Any],
        duration: float,
        soft_transitions: List[Dict[str, Any]],
    ) -> Shot:
        """未检测到切点时，把整个视频作为一个片段"""
        shot_id = self._next_shot_id()
        filename = os.path.basename(video_path)
        fps = cv_meta.get("fps", 24.0)
        ext = os.path.splitext(video_path)[1] or ".mp4"

        clip_path = os.path.join(self.rough_clips_dir, f"{shot_id}{ext}")
        try:
            split_video(video_path, clip_path, 0.0, duration, copy=True)
        except Exception as e:
            logger.warning(f"FFmpeg 整段拷贝失败 {shot_id}: {e}")
            clip_path = ""

        return Shot(
            shot_id=shot_id,
            state="RAW",
            source_file=filename,
            source_path=os.path.abspath(video_path),
            tc_in="00:00:00:00",
            tc_out=sec_to_tc(duration, fps),
            duration_sec=duration,
            fps=fps,
            resolution=cv_meta.get("resolution", (1920, 1080)),
            aspect_ratio=cv_meta.get("aspect_ratio", "16:9"),
            bitrate=cv_meta.get("bitrate"),
            codec=cv_meta.get("codec"),
            visual_quality=cv_meta.get("visual_quality"),
            cv_metadata=cv_meta,
            soft_transitions=soft_transitions,
            needs_review=True,
            provenance=Provenance(
                state="RAW",
                generated_by="phase0_rough_cut",
                split_decision={
                    "start": 0.0,
                    "end": duration,
                    "score": 0.0,
                    "reason": "Phase 0 未检测到切点，按整段输出",
                },
            ),
        )
