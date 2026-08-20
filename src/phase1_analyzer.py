"""
Phase 1: 素材多模态语义分析

职责：
- 对 Phase 0 粗剪后的片段调用 VLM 做镜头内容细节分析。
- 对 PROCESSED 素材直接做 VLM 分析（不切分）。
- 对 ANALYZED 素材只做配置转译。
- 输出完整 Shot 列表与 phase1_analysis.json。
- 不建立关系图（关系图在后续按需由上层业务生成）。
"""
import os
from typing import List, Dict, Any, Tuple, Optional

from src.models import Shot, Relationship, Relationships
from src.services.vlm_service import VLMService
from src.services.asr_service import ASRService
from src.services.face_service import FaceService
from src.processors.phase1_analysis_processor import Phase1AnalysisProcessor
from src.processors.analyzed_processor import AnalyzedProcessor
from src.phase0_rough_cut import RoughCutAnalyzer
from src.utils import (
    logger, ensure_dir, get_video_files, resolve_material_state,
    save_json, load_json,
)


class Phase1Analyzer:
    """阶段1分析器：只做语义分析，不做切分，不建关系图"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.paths = config.get("paths", {})
        self.processing = config.get("processing", {})
        self.materials_config = config.get("materials", [])

        self.output_dir = self.paths.get("output", "./output")
        self.temp_dir = self.paths.get("temp", "./temp")
        ensure_dir(self.output_dir)
        ensure_dir(self.temp_dir)
        ensure_dir(os.path.join(self.output_dir, "phase1_keyframes"))
        ensure_dir(os.path.join(self.output_dir, "logs"))

        self.vlm_service = VLMService(config)
        self.asr_service = ASRService(config)
        self.face_service = FaceService(config)

        self.phase1_processor = Phase1AnalysisProcessor(
            vlm_service=self.vlm_service,
            asr_service=self.asr_service,
            face_service=self.face_service,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            keyframe_strategy=self.processing.get("keyframe_strategy", "adaptive"),
            keyframe_interval=self.processing.get("keyframe_interval", 2.0),
        )
        self.analyzed_processor = AnalyzedProcessor(
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
        )

        self._shot_counter = 0

    def _next_shot_id(self) -> str:
        self._shot_counter += 1
        return f"S{self._shot_counter:03d}"

    def run(self) -> List[Shot]:
        """执行 Phase 1 分析"""
        logger.info("=" * 60)
        logger.info("Phase 1: 素材多模态语义分析")
        logger.info("=" * 60)

        video_tasks = self._collect_videos()
        if not video_tasks:
            logger.error("未找到任何视频任务")
            return []

        logger.info(f"共发现 {len(video_tasks)} 个视频任务")
        for path, state, _ in video_tasks:
            logger.info(f"  [{state}] {os.path.basename(path)}")

        all_shots: List[Shot] = []
        for video_path, state, meta_format in video_tasks:
            shots = self._process_by_state(video_path, state, meta_format)
            all_shots.extend(shots)

        logger.info(f"共生成 {len(all_shots)} 个 Shot")

        # 保存结果
        result = {
            "project": self.project,
            "total_shots": len(all_shots),
            "shots": [shot.to_dict() for shot in all_shots],
        }
        save_json(result, os.path.join(self.output_dir, "phase1_analysis.json"))

        # 为每个 Shot 保存独立配置文件
        self._save_shot_configs(all_shots)

        # 本地模型用完后卸载，释放显存给 Phase 2/3 的 LLM
        self._unload_local_models()

        logger.info("Phase 1 完成")
        return all_shots

    def _unload_local_models(self):
        """如果使用了本地模型，分析结束后卸载以释放显存"""
        try:
            if hasattr(self.vlm_service, "local_service") and self.vlm_service.local_service is not None:
                self.vlm_service.local_service._get_engine().unload()
        except Exception as e:
            logger.debug(f"卸载本地 VLM 失败: {e}")

        try:
            if hasattr(self.asr_service, "local_service") and self.asr_service.local_service is not None:
                self.asr_service.local_service.unload()
        except Exception as e:
            logger.debug(f"卸载本地 ASR 失败: {e}")

        try:
            if self.face_service is not None:
                self.face_service.unload()
        except Exception as e:
            logger.debug(f"卸载本地 Face 失败: {e}")

    def _process_by_state(
        self,
        video_path: str,
        state: str,
        meta_format: Optional[str],
    ) -> List[Shot]:
        """按素材状态分发处理"""
        if state == "RAW":
            return self._process_raw(video_path)
        elif state == "PROCESSED":
            return self.phase1_processor.process(
                video_path=video_path,
                next_shot_id_func=self._next_shot_id,
                state="PROCESSED",
            )
        elif state == "ANALYZED":
            return self.analyzed_processor.process(video_path, meta_format, self._next_shot_id)
        else:
            logger.error(f"未知素材状态: {state}")
            return []

    def _process_raw(self, video_path: str) -> List[Shot]:
        """处理 RAW 素材：优先读取 Phase 0 粗剪结果，否则先粗剪再分析"""
        rough_config_path = os.path.join(self.output_dir, "phase0_rough_config.json")

        if os.path.exists(rough_config_path):
            logger.info(f"[RAW] 使用 Phase 0 粗剪结果: {rough_config_path}")
            rough_data = load_json(rough_config_path)
            rough_shots = [Shot.from_dict(s) for s in rough_data.get("shots", [])]
            # 只分析与当前 RAW 视频来源一致的片段
            source_file = os.path.basename(video_path)
            relevant = [s for s in rough_shots if s.source_file == source_file]
            if not relevant:
                logger.warning(f"[RAW] Phase 0 结果中未找到 {source_file} 的片段，按整段分析")
                relevant = [self._build_fallback_rough_shot(video_path)]
        else:
            logger.warning(f"[RAW] 未找到 Phase 0 粗剪结果，先执行 Phase 0 粗剪: {video_path}")
            analyzer = RoughCutAnalyzer(self.config)
            relevant = analyzer._process_video(video_path)
            if not relevant:
                logger.warning(f"[RAW] Phase 0 未切分，按整段分析: {video_path}")
                relevant = [self._build_fallback_rough_shot(video_path)]

        shots = []
        id_map: Dict[str, str] = {}  # Phase 0 shot_id -> Phase 1 shot_id
        for rough_shot in relevant:
            clip_path = self._resolve_clip_path(rough_shot)
            if not clip_path or not os.path.exists(clip_path):
                logger.warning(f"[RAW] 粗剪片段不存在，跳过: {clip_path}")
                continue

            new_shot_id = self._next_shot_id()
            id_map[rough_shot.shot_id] = new_shot_id

            analyzed = self.phase1_processor.process(
                video_path=clip_path,
                next_shot_id_func=self._next_shot_id,
                shot_id=new_shot_id,
                state="RAW",
                cv_metadata=rough_shot.cv_metadata,
            )
            shots.extend(analyzed)

        # 按顺序重建同素材内的前后关系
        shots.sort(key=lambda s: s.tc_in)
        for i in range(len(shots)):
            current = shots[i]
            if i > 0:
                prev = shots[i - 1]
                current.relationships.prev = Relationship(
                    shot_id=prev.shot_id,
                    relationship_type="同素材时间连续",
                    coherence_score=0.95,
                )
            if i < len(shots) - 1:
                nxt = shots[i + 1]
                current.relationships.next = Relationship(
                    shot_id=nxt.shot_id,
                    relationship_type="同素材时间连续",
                    coherence_score=0.95,
                )

        return shots

    def _build_fallback_rough_shot(self, video_path: str) -> Shot:
        """未找到 Phase 0 结果时，构建一个代表整段的 fallback Shot"""
        from src.cv_utils import cv_pre_scan
        from src.models import Provenance

        cv_meta = cv_pre_scan(video_path)
        duration = cv_meta.get("duration", 0.0)
        fps = cv_meta.get("fps", 24.0)

        return Shot(
            shot_id="S000",
            state="RAW",
            source_file=os.path.basename(video_path),
            source_path=os.path.abspath(video_path),
            tc_in="00:00:00:00",
            tc_out="00:00:00:00" if duration == 0 else None,
            duration_sec=duration,
            fps=fps,
            resolution=cv_meta.get("resolution", (1920, 1080)),
            aspect_ratio=cv_meta.get("aspect_ratio", "16:9"),
            bitrate=cv_meta.get("bitrate"),
            codec=cv_meta.get("codec"),
            cv_metadata=cv_meta,
            needs_review=True,
            provenance=Provenance(
                state="RAW",
                generated_by="phase1_fallback",
                split_decision={"reason": "未找到 Phase 0 结果，按整段分析"},
            ),
        )

    def _resolve_clip_path(self, rough_shot: Shot) -> str:
        """从粗 Shot 中解析片段文件路径"""
        # 优先使用 cv_metadata.shot_config.clip_path
        cfg = (rough_shot.cv_metadata or {}).get("shot_config", {})
        clip_path = cfg.get("clip_path") or cfg.get("split_clip_path") or ""
        if clip_path and os.path.exists(clip_path):
            return clip_path

        # 其次尝试按 shot_id 在 phase0_rough_clips 中查找
        clip_path = os.path.join(self.output_dir, "phase0_rough_clips", f"{rough_shot.shot_id}.mp4")
        if os.path.exists(clip_path):
            return clip_path

        # 回退到 source_path（整段分析）
        return rough_shot.source_path

    def _collect_videos(self) -> List[Tuple[str, str, Optional[str]]]:
        """收集所有视频任务及其状态"""
        tasks = []
        overrides = self.config.get("materials_overrides", [])
        seen = set()

        material_dirs = []
        for item in self.materials_config:
            if item.get("path"):
                material_dirs.append(item["path"])

        raw_dir = self.paths.get('raw_materials')
        if raw_dir and raw_dir not in material_dirs:
            material_dirs.append(raw_dir)

        if not material_dirs:
            logger.warning("未配置任何素材目录")
            return tasks

        for directory in material_dirs:
            if not os.path.exists(directory):
                logger.warning(f"素材目录不存在: {directory}")
                continue

            video_files = get_video_files(directory)
            for video_path in video_files:
                abs_path = os.path.abspath(video_path)
                if abs_path in seen:
                    continue
                seen.add(abs_path)

                state, item = resolve_material_state(abs_path, self.materials_config, overrides)
                meta_format = item.get("meta_format") if state == "ANALYZED" else None
                tasks.append((abs_path, state, meta_format))

        return tasks

    def _save_shot_configs(self, shots: List[Shot]):
        """为每个 Shot 生成独立的 JSON 配置文件"""
        import json

        clips_dir = os.path.join(self.output_dir, "phase1_split_clips")
        ensure_dir(clips_dir)

        for shot in shots:
            base_config = (shot.cv_metadata or {}).get("shot_config", {}) if shot.cv_metadata else {}

            config = dict(base_config)
            config.update({
                "shot_id": shot.shot_id,
                "clip_path": f"{shot.shot_id}.mp4",
                "source_file": shot.source_file,
                "source_path": shot.source_path,
                "tc_in": shot.tc_in,
                "tc_out": shot.tc_out,
                "duration_sec": shot.duration_sec,
                "relationships": shot.relationships.to_dict() if shot.relationships else None,
                "keyframes": shot.keyframes,
            })

            config_path = os.path.join(clips_dir, f"{shot.shot_id}_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"已为 {len(shots)} 个 Shot 生成独立配置文件，目录: {clips_dir}")
