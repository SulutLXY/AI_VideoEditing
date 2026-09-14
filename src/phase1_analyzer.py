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
import re
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
        # 阶段A预抽帧结果：片段绝对路径 -> 帧目录（stage A 填充，stage B 消费）
        self._frames_dirs: Dict[str, str] = {}

    def _next_shot_id(self) -> str:
        self._shot_counter += 1
        return f"S{self._shot_counter:03d}"

    def _preextract_all_frames(self, video_tasks: List[Tuple[str, str, Optional[str]]]):
        """阶段A：纯 CV 预抽帧（480p jpg 落盘），与 LLM 分析解耦。

        全部片段先抽好帧，用户可从日志/phase1_frames 目录实时看到解析进度；
        分析中断后重跑时帧直接复用（幂等），LLM 阶段只读图不再解码视频。
        """
        local = getattr(self.vlm_service, "local_service", None)
        if local is None or not getattr(local, "enabled", False):
            return

        from src.local_models.vision_engine import VisionEngine

        frames_root = os.path.join(self.output_dir, "phase1_frames")
        ensure_dir(frames_root)

        # 汇总所有待分析的视频（RAW 解析到片段级，PROCESSED 整段）
        clip_paths: List[str] = []
        rough_shots: List[Shot] = []
        rough_config_path = os.path.join(self.output_dir, "phase0_rough_config.json")
        if os.path.exists(rough_config_path):
            try:
                rough_data = load_json(rough_config_path)
                rough_shots = [Shot.from_dict(s) for s in rough_data.get("shots", [])]
            except Exception as e:
                logger.warning(f"[Phase 1] 读取 Phase 0 配置失败，预抽帧按整段处理: {e}")

        for path, state, _ in video_tasks:
            if state == "ANALYZED":
                continue
            if state == "RAW":
                source_file = os.path.basename(path)
                relevant = [s for s in rough_shots if s.source_file == source_file]
                if not relevant:
                    clip_paths.append(path)
                else:
                    for rs in relevant:
                        cp = self._resolve_clip_path(rs)
                        if cp and os.path.exists(cp):
                            clip_paths.append(cp)
            else:  # PROCESSED 整段分析
                clip_paths.append(path)

        # 去重（同一片段不重复抽帧）；目录名冲突时加路径哈希后缀
        seen = set()
        used_stems: Dict[str, str] = {}
        total, ok = 0, 0
        for cp in clip_paths:
            ap = os.path.abspath(cp)
            if ap in seen:
                continue
            seen.add(ap)
            total += 1

            stem = os.path.splitext(os.path.basename(ap))[0]
            owner = used_stems.get(stem)
            if owner is None:
                used_stems[stem] = ap
            elif owner != ap:
                import hashlib
                stem = f"{stem}_{hashlib.md5(ap.encode('utf-8')).hexdigest()[:6]}"

            out_dir = os.path.join(frames_root, stem)
            meta = VisionEngine.preextract_frames(
                ap, out_dir,
                count=local.keyframe_count,
                interval=local.frame_interval,
                max_frames=local.max_frames,
            )
            if meta:
                ok += 1
                self._frames_dirs[ap] = out_dir
                has_audio = self._extract_audio_wav(ap, out_dir)
                logger.info(
                    f"[Phase 1] 帧已就绪 {os.path.basename(ap)}: "
                    f"{meta['frame_count']} 帧 / {meta['duration']}s"
                    + ("（含音频）" if has_audio else "（无音频轨）")
                )
            else:
                logger.warning(f"[Phase 1] 预抽帧失败（分析时将回退现场抽帧）: {ap}")

        logger.info(f"[Phase 1] 预抽帧完成: {ok}/{total} 个片段帧就绪 -> {frames_root}")

    @staticmethod
    def _extract_audio_wav(video_path: str, out_dir: str) -> bool:
        """抽取 16k 单声道 wav 到帧目录（幂等）。返回是否存在音频轨。"""
        import subprocess
        wav_path = os.path.join(out_dir, "audio.wav")
        if os.path.exists(wav_path):
            return True
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not os.path.exists(wav_path):
                return False
            # 空音频（0 字节级）视为无音频轨
            return os.path.getsize(wav_path) > 4096
        except Exception as e:
            logger.warning(f"[Phase 1] 抽取音频失败 {os.path.basename(video_path)}: {e}")
            return False

    @staticmethod
    def _detect_volume(wav_path: str) -> Dict[str, Optional[float]]:
        """ffmpeg volumedetect 检测平均/最大音量（dB）"""
        import subprocess
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-i", wav_path,
                 "-af", "volumedetect", "-f", "null", "-"],
                capture_output=True, text=True,
            )
            out = (r.stderr or "") + (r.stdout or "")
            mean = mx = None
            m = re.search(r"mean_volume: ([-\d.]+) dB", out)
            if m:
                mean = float(m.group(1))
            m = re.search(r"max_volume: ([-\d.]+) dB", out)
            if m:
                mx = float(m.group(1))
            return {"mean_volume_db": mean, "max_volume_db": mx}
        except Exception:
            return {"mean_volume_db": None, "max_volume_db": None}

    def _analyze_all_audio(self):
        """阶段A2：串行音频分析（SenseVoice，CPU 跑，不与 GPU 上的 VLM 抢显存）。

        逐片段生成 audio_profile.json：
          { has_speech, event, language, emotion,
            mean_volume_db, max_volume_db,
            text, transcript: [{start,end,text}] }
        已存在的跳过（幂等）。产物供 VLM prompt 注入与 Phase 4 混音参考。
        """
        local_asr = getattr(self.asr_service, "local_service", None)
        if local_asr is None or not getattr(local_asr, "enabled", False):
            logger.info("[Phase 1] 本地 ASR 未启用，跳过阶段A2音频分析")
            return

        engine = None
        total, ok = 0, 0
        for clip_path, frames_dir in self._frames_dirs.items():
            profile_path = os.path.join(frames_dir, "audio_profile.json")
            if os.path.exists(profile_path):
                continue
            wav_path = os.path.join(frames_dir, "audio.wav")
            if not os.path.exists(wav_path):
                continue

            total += 1
            try:
                if engine is None:
                    engine = local_asr._get_engine()
                result = engine.process_wav(wav_path)
                volume = self._detect_volume(wav_path)
                mean_db = volume.get("mean_volume_db")
                has_speech = bool(result.get("has_speech"))
                # 无对白镜头靠音量区分：纯静音 vs 有 BGM/环境音
                if has_speech:
                    sound_env = "speech"
                elif mean_db is not None and mean_db > -45.0:
                    sound_env = "ambient_or_bgm"
                else:
                    sound_env = "silent"
                profile = {
                    "has_speech": has_speech,
                    "event": result.get("event"),
                    "language": result.get("language"),
                    "emotion": result.get("emotion"),
                    "sound_env": sound_env,
                    **volume,
                    "text": result.get("text", ""),
                    "transcript": result.get("transcriptions", []),
                }
                with open(profile_path, "w", encoding="utf-8") as f:
                    json.dump(profile, f, ensure_ascii=False, indent=2)
                ok += 1

                # 一行摘要：语言/事件/情绪/有无台词
                desc_bits = []
                if profile["language"]:
                    desc_bits.append(profile["language"])
                desc_bits.append(profile["event"] or ("Speech" if profile["has_speech"] else {
                    "ambient_or_bgm": "BGM/环境音",
                    "silent": "静音",
                }.get(sound_env, "无语音")))
                if profile["emotion"]:
                    desc_bits.append(profile["emotion"])
                desc_bits.append("有台词" if profile["text"] else "无台词")
                logger.info(
                    f"[Phase 1] 音频分析 {os.path.basename(clip_path)}: "
                    f"{'/'.join(desc_bits)}"
                    + (f" \"{profile['text'][:30]}\"" if profile["text"] else "")
                )
            except Exception as e:
                logger.warning(f"[Phase 1] 音频分析失败 {os.path.basename(clip_path)}: {e}")

        if total:
            logger.info(f"[Phase 1] 音频分析完成: {ok}/{total} 个片段")

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

        # 阶段A：纯 CV 预抽帧（本地模型模式下），全部就绪后再进入 LLM 分析
        self._preextract_all_frames(video_tasks)
        # 阶段A2：串行音频分析（SenseVoice，CPU），先生成音频档案供 VLM 参考
        self._analyze_all_audio()
        # 音频分析完毕即释放 ASR 显存，把 GPU 完整留给阶段B的 VLM（8GB 卡显存紧张）
        local_asr = getattr(self.asr_service, "local_service", None)
        if local_asr is not None:
            try:
                local_asr.unload()
            except Exception as e:
                logger.debug(f"阶段A2后卸载 ASR 失败: {e}")

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
            return self.phase1_processor.process(
                video_path=video_path,
                next_shot_id_func=self._next_shot_id,
                state="PROCESSED",
                frames_dir=self._frames_dirs.get(os.path.abspath(video_path)),
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
                cv_meta=rough_shot.cv_metadata,
                source_file=rough_shot.source_file,
                source_path=rough_shot.source_path,
                tc_in=rough_shot.tc_in,
                tc_out=rough_shot.tc_out,
                frames_dir=self._frames_dirs.get(os.path.abspath(clip_path)),
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
