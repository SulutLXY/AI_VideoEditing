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
import json
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
        # Phase 0 产物映射：片段绝对路径 -> phase0_rough_clips/Sxxx_frames/
        # （自适应抽帧 + 音频档案，由 _resolve_phase0_assets 填充）
        self._frames_dirs: Dict[str, str] = {}

    def _next_shot_id(self) -> str:
        self._shot_counter += 1
        return f"S{self._shot_counter:03d}"

    def _resolve_phase0_assets(self, video_tasks: List[Tuple[str, str, Optional[str]]]):
        """阶段A替代：定位 Phase 0 每片段产物（自适应抽帧 + 音频档案）。

        Phase 0 已随片段落盘 phase0_rough_clips/Sxxx_frames/
        （frames.json/meta.json/audio.wav/audio_profile.json），这里只做映射不做重算；
        找不到的片段由 VLM 分析时回退现场抽帧、由 ASR 服务回退现场转录。
        """
        rough_config_path = os.path.join(self.output_dir, "phase0_rough_config.json")
        rough_shots: List[Shot] = []
        if os.path.exists(rough_config_path):
            try:
                rough_data = load_json(rough_config_path)
                rough_shots = [Shot.from_dict(s) for s in rough_data.get("shots", [])]
            except Exception as e:
                logger.warning(f"[Phase 1] 读取 Phase 0 配置失败: {e}")

        clips_dir = os.path.join(self.output_dir, "phase0_rough_clips")

        def _map_clip(cp: str):
            ap = os.path.abspath(cp)
            stem = os.path.splitext(os.path.basename(ap))[0]
            frames_dir = os.path.join(clips_dir, f"{stem}_frames")
            if os.path.exists(os.path.join(frames_dir, "frames.json")):
                self._frames_dirs[ap] = frames_dir

        for path, state, _ in video_tasks:
            if state == "ANALYZED":
                continue
            if state == "RAW":
                source_file = os.path.basename(path)
                relevant = [s for s in rough_shots if s.source_file == source_file]
                if not relevant:
                    _map_clip(path)
                else:
                    for rs in relevant:
                        cp = self._resolve_clip_path(rs)
                        if cp and os.path.exists(cp):
                            _map_clip(cp)
            else:  # PROCESSED 整段：若 Phase 0 恰好有同名产物则复用
                _map_clip(path)

        if self._frames_dirs:
            logger.info(
                f"[Phase 1] Phase 0 产物就绪: {len(self._frames_dirs)} 个片段"
                f"（帧 + 音频档案）"
            )

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

        # 阶段A替代：定位 Phase 0 每片段产物（自适应抽帧 + 音频档案随片段落盘），
        # 不再由 Phase 1 预抽帧/音频分析；缺失的片段由 VLM 分析时回退现场抽帧/转录
        self._resolve_phase0_assets(video_tasks)

        # 阶段A2.5：参考图预分析（一次性，幂等），生成关键词档案供身份确认调用
        self._profile_references()

        all_shots: List[Shot] = []
        for video_path, state, meta_format in video_tasks:
            shots = self._process_by_state(video_path, state, meta_format)
            all_shots.extend(shots)
            # 每分析完一个素材立即写出其镜头配置（含视频拷贝），
            # 资产库可实时看到进度，中途中断也保留已完成部分
            if shots:
                self._save_shot_configs(shots)
                logger.info(
                    f"[Phase 1] 已写出 {len(shots)} 个镜头配置: "
                    f"{[s.shot_id for s in shots]}"
                )

        logger.info(f"共生成 {len(all_shots)} 个 Shot")

        # 保存结果
        result = {
            "project": self.project,
            "total_shots": len(all_shots),
            "shots": [shot.to_dict() for shot in all_shots],
        }
        save_json(result, os.path.join(self.output_dir, "phase1_analysis.json"))

        # 本地模型用完后卸载，释放显存给 Phase 2/3 的 LLM
        self._unload_local_models()

        logger.info("Phase 1 完成")
        return all_shots

    def _profile_references(self):
        """阶段A2.5：参考图预分析（本地 VLM 一次性提取关键词档案）。

        引擎此处按需加载；档案写 refs/ref_profiles.json，图片未变化时幂等跳过。
        """
        local = getattr(self.vlm_service, "local_service", None)
        if local is None or not getattr(local, "enabled", False):
            return
        if not getattr(local, "use_reference_images", False):
            return
        refs_dir = self.config.get("paths", {}).get("reference_images")
        if not refs_dir or not os.path.isdir(refs_dir):
            return
        try:
            from src.local_models.ref_profiler import ensure_ref_profiles
            engine = local._get_engine()
            engine.load()  # _get_engine 只构造不加载，阶段B的 process_video 才触发 load
            if getattr(engine, "model", None) is None:
                logger.warning("[Phase 1] 本地 VLM 模型加载失败，跳过参考图预分析")
                return
            ensure_ref_profiles(refs_dir, engine)
        except Exception as e:
            logger.warning(f"[Phase 1] 参考图预分析失败（身份确认将仅用用户备注）: {e}")

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
            shot_id = self._next_shot_id()
            existing = self._load_existing_shot(shot_id)
            if existing is not None:
                logger.info(f"[Phase 1] 断点续跑，跳过已分析镜头: {shot_id}")
                return [existing]
            return self.phase1_processor.process(
                video_path=video_path,
                next_shot_id_func=self._next_shot_id,
                shot_id=shot_id,
                state="PROCESSED",
                frames_dir=self._frames_dirs.get(os.path.abspath(video_path)),
                on_shot_done=self._on_shot_done,
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

            # 断点续跑：该镜头已有完整配置（json + mp4）则直接复用，跳过 VLM 分析
            existing = self._load_existing_shot(new_shot_id)
            if existing is not None:
                logger.info(f"[Phase 1] 断点续跑，跳过已分析镜头: {new_shot_id}")
                shots.append(existing)
                continue

            analyzed = self.phase1_processor.process(
                video_path=clip_path,
                next_shot_id_func=self._next_shot_id,
                shot_id=new_shot_id,
                state="RAW",
                cv_meta=rough_shot.cv_metadata,
                source_file=rough_shot.source_file,
                source_path=rough_shot.source_path,
                tc_in=rough_shot.tc_in,
                tc_out=rough_shot.tc_out,
                frames_dir=self._frames_dirs.get(os.path.abspath(clip_path)),
                on_shot_done=self._on_shot_done,
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
        """收集所有视频任务及其状态

        如果已存在 Phase 0 粗剪结果，则只收集粗剪配置中实际包含的 source_file，
        避免 E/F/G 等剧情终点之后的素材进入 Phase 1 分析。
        同时按 source_file 去重，防止 workspace/materials/raw 与 raw_materials
        指向同批素材副本时重复分析。
        """
        tasks = []
        overrides = self.config.get("materials_overrides", [])
        seen_path = set()
        seen_source = set()

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

        # 若存在 Phase 0 配置，仅保留配置里出现的 source_file
        rough_config_path = os.path.join(self.output_dir, "phase0_rough_config.json")
        allowed_sources: Optional[set] = None
        if os.path.exists(rough_config_path):
            try:
                rough_cfg = load_json(rough_config_path)
                allowed_sources = set(rough_cfg.get("assets", {}).keys())
                logger.info(f"[Phase 1] 根据 Phase 0 配置限定素材范围: {len(allowed_sources)} 个源文件")
            except Exception as e:
                logger.warning(f"[Phase 1] 读取 Phase 0 配置失败，将扫描全部素材: {e}")

        for directory in material_dirs:
            if not os.path.exists(directory):
                logger.warning(f"素材目录不存在: {directory}")
                continue

            video_files = get_video_files(directory)
            for video_path in video_files:
                abs_path = os.path.abspath(video_path)
                if abs_path in seen_path:
                    continue
                seen_path.add(abs_path)

                source_name = os.path.basename(abs_path)
                if source_name in seen_source:
                    logger.info(f"[Phase 1] 跳过重复素材: {source_name}")
                    continue

                if allowed_sources is not None and source_name not in allowed_sources:
                    logger.info(f"[Phase 1] 跳过 Phase 0 配置外的素材: {source_name}")
                    continue

                seen_source.add(source_name)
                state, item = resolve_material_state(abs_path, self.materials_config, overrides)
                meta_format = item.get("meta_format") if state == "ANALYZED" else None
                tasks.append((abs_path, state, meta_format))

        return tasks

    def _on_shot_done(self, shot: Shot):
        """单镜头分析完成回调：立即落盘该镜头配置（CPU 同步写，json 与 mp4 同步出现）。

        保存粒度是镜头而非源视频——一个源视频可能有几十个镜头，
        等源视频全部跑完再写就无法中途判断输出质量或断点续跑。
        """
        try:
            self._save_shot_configs([shot])
        except Exception as e:
            logger.warning(f"[Phase 1] 写出镜头配置失败 {shot.shot_id}: {e}")

    def _load_existing_shot(self, shot_id: str) -> Optional[Shot]:
        """断点续跑：配置与片段都在则直接重建 Shot，跳过 VLM 分析。

        shot_id 由确定性计数器分配（源视频与片段顺序不变则 ID 对齐），
        配置里含完整 shot 序列化（见 _save_shot_configs）。
        """
        clips_dir = os.path.join(self.output_dir, "phase1_split_clips")
        clip_path = os.path.join(clips_dir, f"{shot_id}.mp4")
        config_path = os.path.join(clips_dir, f"{shot_id}_config.json")
        if not (os.path.exists(clip_path) and os.path.exists(config_path)):
            return None
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
            shot_data = data.get("shot")
            if not shot_data:
                return None
            return Shot.from_dict(shot_data)
        except Exception as e:
            logger.warning(f"[Phase 1] 读取已有镜头配置失败 {shot_id}，将重新分析: {e}")
            return None

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
                # 完整 Shot 序列化：断点续跑时直接重建对象，无需重新分析
                "shot": shot.to_dict(),
            })

            config_path = os.path.join(clips_dir, f"{shot.shot_id}_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"已为 {len(shots)} 个 Shot 生成独立配置文件，目录: {clips_dir}")
