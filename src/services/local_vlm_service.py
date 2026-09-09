"""
本地 VLM 服务封装

把 video-intelligence-extractor 的 VisionEngine 适配成项目 VLMService 的接口，
供 Phase 1 在 `models.vlm.provider == "local"` 时调用。
"""
import os
import json
from typing import List, Dict, Tuple, Any, Optional

from src.models import Segment
from src.utils import logger


class LocalVLMService:
    """基于 Qwen2.5-VL 的本地视觉理解服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local_cfg = config.get("models", {}).get("local", {})
        vision_cfg = local_cfg.get("vision", {})

        self.enabled = bool(local_cfg.get("enabled", False))
        self.device = local_cfg.get("device", "cuda")
        self.cache_dir = local_cfg.get("cache_dir")
        self.model_id = vision_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.model_path = vision_cfg.get("model_path")
        self.load_in_4bit = vision_cfg.get("load_in_4bit", True)
        self.max_new_tokens = vision_cfg.get("max_new_tokens", 512)
        self.keyframe_count = vision_cfg.get("keyframe_count", 3)

        # 多图联合分析参数（v0.5）：>1s 镜头按 interval 抽帧，注入剧本/参考图
        self.frame_interval = float(vision_cfg.get("frame_interval", 0.5))
        self.max_frames = int(vision_cfg.get("max_frames", 16))
        self.use_script_context = bool(vision_cfg.get("use_script_context", True))
        self.use_reference_images = bool(vision_cfg.get("use_reference_images", True))
        self.script_max_chars = int(vision_cfg.get("script_max_chars", 1500))
        self.max_ref_images = int(vision_cfg.get("max_ref_images", 8))

        self._engine = None
        self._script_cache: Optional[str] = None
        self._ref_images_cache: Optional[List] = None

    def _get_engine(self):
        if self._engine is None:
            from src.local_models.vision_engine import VisionEngine
            self._engine = VisionEngine(
                model_id=self.model_id,
                model_path=self.model_path,
                device=self.device,
                load_in_4bit=self.load_in_4bit,
                max_new_tokens=self.max_new_tokens,
                cache_dir=self.cache_dir,
            )
        return self._engine

    def _load_script_context(self) -> Optional[str]:
        """加载剧本大纲作为分析上下文（项目类型/风格 + 剧本正文，截断到 script_max_chars）"""
        if self._script_cache is not None:
            return self._script_cache or None

        path = self.config.get("paths", {}).get("script_outline", "./script.md")
        context = ""
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read().strip()
                project = self.config.get("project", {})
                header_parts = []
                if project.get("genre"):
                    header_parts.append(f"类型：{project['genre']}")
                if project.get("style"):
                    header_parts.append(f"风格：{project['style']}")
                header = "，".join(header_parts)
                context = (f"（{header}）\n" if header else "") + text
                if len(context) > self.script_max_chars:
                    context = context[: self.script_max_chars] + "……（剧本节选）"
                logger.info(f"[LocalVLM] 已加载剧本上下文: {path} ({len(context)} 字符)")
            except Exception as e:
                logger.warning(f"[LocalVLM] 加载剧本上下文失败: {e}")
        else:
            logger.info(f"[LocalVLM] 未找到剧本文件 {path}，跳过剧情背景注入")
        self._script_cache = context
        return context or None

    def _load_ref_images(self) -> Optional[List]:
        """加载角色参考图 [(角色名, PIL.Image, 说明), ...]

        优先读取 refs.json 清单（WebUI 管理，含角色名与说明文字）；
        无清单时回退目录扫描（两种命名约定，说明为空）：
        - refs/角色名/*.jpg（子目录）
        - refs/角色名_角度.jpg（下划线前缀，扁平存放）
        """
        if self._ref_images_cache is not None:
            return self._ref_images_cache or None

        refs: List = []
        candidates = []
        local_cfg = self.config.get("models", {}).get("local", {})
        face_refs = local_cfg.get("face", {}).get("refs_dir")
        if face_refs:
            candidates.append(face_refs)
        path_refs = self.config.get("paths", {}).get("reference_images")
        if path_refs and path_refs not in candidates:
            candidates.append(path_refs)

        try:
            from PIL import Image
        except Exception as e:
            logger.warning(f"[LocalVLM] PIL 不可用，跳过参考图注入: {e}")
            self._ref_images_cache = []
            return None

        exts = (".jpg", ".jpeg", ".png", ".webp")
        manifest_name = "refs.json"
        for base in candidates:
            if not base or not os.path.isdir(base):
                continue

            # 1) 清单模式（WebUI 管理，含说明文字）
            manifest_path = os.path.join(base, manifest_name)
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        data = json.load(f)
                    for e in data.get("images", []):
                        if len(refs) >= self.max_ref_images:
                            break
                        file_name = str(e.get("file", "")).strip()
                        full = os.path.join(base, file_name)
                        if not file_name or not os.path.exists(full):
                            logger.warning(f"[LocalVLM] 清单中的参考图缺失，跳过: {file_name}")
                            continue
                        try:
                            refs.append((
                                str(e.get("name", "")).strip(),
                                Image.open(full).convert("RGB"),
                                str(e.get("description", "")).strip(),
                            ))
                        except Exception as ex:
                            logger.warning(f"[LocalVLM] 参考图读取失败 {full}: {ex}")
                    if refs:
                        logger.info(
                            f"[LocalVLM] 已从清单加载 {len(refs)} 张角色参考图: "
                            f"{[n for n, _, _ in refs]}"
                        )
                        break
                except Exception as ex:
                    logger.warning(f"[LocalVLM] 读取参考图清单失败，回退目录扫描: {ex}")

            # 2) 目录扫描模式（无清单，说明为空）
            try:
                for entry in sorted(os.listdir(base)):
                    full = os.path.join(base, entry)
                    if os.path.isdir(full):
                        # 子目录约定：目录名 = 角色名
                        name = entry
                        imgs = [
                            os.path.join(full, f) for f in sorted(os.listdir(full))
                            if f.lower().endswith(exts)
                        ][:2]
                    elif entry.lower().endswith(exts) and entry != manifest_name:
                        # 扁平约定：文件名前缀（第一个下划线前）= 角色名
                        stem = os.path.splitext(entry)[0]
                        name = stem.split("_")[0]
                        imgs = [full]
                    else:
                        continue
                    for img_path in imgs:
                        try:
                            refs.append((name, Image.open(img_path).convert("RGB"), ""))
                        except Exception as e:
                            logger.warning(f"[LocalVLM] 参考图读取失败 {img_path}: {e}")
                        if len(refs) >= self.max_ref_images:
                            break
                    if len(refs) >= self.max_ref_images:
                        break
            except Exception as e:
                logger.warning(f"[LocalVLM] 扫描参考图目录失败 {base}: {e}")
            if refs:
                break

        if refs:
            logger.info(f"[LocalVLM] 已加载 {len(refs)} 张角色参考图: {[n for n, _, _ in refs]}")
        self._ref_images_cache = refs
        return refs or None

    def _get_engine(self):
        if self._engine is None:
            from src.local_models.vision_engine import VisionEngine
            self._engine = VisionEngine(
                model_id=self.model_id,
                model_path=self.model_path,
                device=self.device,
                load_in_4bit=self.load_in_4bit,
                max_new_tokens=self.max_new_tokens,
                cache_dir=self.cache_dir,
            )
        return self._engine

    def sample_frames(self, video_path: str, duration: float, temp_dir: str) -> List[Tuple[float, str]]:
        """本地服务不依赖外部 base64 帧，返回空列表即可；关键帧由 engine 内部抽取"""
        return []

    def analyze_whole_video(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> Dict[str, Any]:
        """对单个视频片段（或已切分 Shot）做完整内容分析"""
        if not self.enabled:
            logger.warning("[LocalVLM] 本地模型未启用，跳过本地视觉分析")
            return {}

        logger.info(f"[LocalVLM] 本地视觉分析: {os.path.basename(video_path)}")
        try:
            engine = self._get_engine()
            result = engine.process_video(
                video_path,
                keyframe_count=self.keyframe_count,
                frame_interval=self.frame_interval,
                max_frames=self.max_frames,
                script_context=self._load_script_context() if self.use_script_context else None,
                ref_images=self._load_ref_images() if self.use_reference_images else None,
            )
            return self._map_to_shot_fields(result)
        except Exception as e:
            logger.error(f"[LocalVLM] 本地视觉分析失败: {e}")
            return {}

    def analyze_temporal_segments(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> List[Segment]:
        """本地模型目前按整段分析，返回单个覆盖全长的 segment"""
        description = self.analyze_whole_video(video_path, frames, duration)
        return [Segment(
            start=0.0,
            end=duration,
            description=description.get("action", ""),
            location=description.get("location", ""),
            time_of_day=description.get("time_of_day", ""),
            characters=description.get("characters", []),
            action=description.get("action", ""),
            emotion=description.get("emotion", ""),
            dialogue=description.get("dialogue", ""),
            camera_position=description.get("camera_position", ""),
            camera_movement=description.get("camera_movement", ""),
            shot_size=description.get("shot_size", ""),
            framing=description.get("framing", ""),
            lighting=description.get("lighting", ""),
            color_tone=description.get("color_tone", ""),
            style=description.get("style", ""),
            atmosphere=description.get("atmosphere", ""),
            culture=description.get("culture", ""),
            key_objects=description.get("key_objects", []),
            tags=description.get("tags", []),
            direction=description.get("direction", ""),
            performance=description.get("performance", ""),
            action_details=description.get("action_details", ""),
            continuity_score=float(description.get("continuity_score", 0.0) or 0.0),
            continuity_notes=description.get("continuity_notes", ""),
            coherence_score=0.0,
            is_long_take=False,
        )]

    def validate_cut_candidates(
        self,
        candidates: List[Tuple[float, List[Tuple[float, str]]]],
    ) -> Dict[float, bool]:
        """本地模型暂不支持切点验证，返回空，由 Phase 0 CV 独立处理"""
        logger.debug("[LocalVLM] 本地模型不支持 VLM 切点验证，跳过")
        return {}

    @staticmethod
    def _map_to_shot_fields(result: Dict[str, Any]) -> Dict[str, Any]:
        """把 VisionEngine 输出字段映射到 Shot 模型字段"""
        return {
            "location": result.get("location", ""),
            "time_of_day": result.get("time_of_day", ""),
            "characters": result.get("characters", []),  # 由 Face 服务补充/校正
            "action": result.get("action", ""),
            "action_details": result.get("action_details", ""),
            "emotion": result.get("emotion", ""),
            "dialogue": "",
            "camera_position": result.get("camera_position", ""),
            "camera_movement": result.get("camera_movement", ""),
            "shot_size": result.get("shot_size", ""),
            "framing": result.get("framing", ""),
            "lighting": result.get("lighting", ""),
            "color_tone": result.get("color_tone", ""),
            "style": result.get("style", ""),
            "atmosphere": result.get("atmosphere", ""),
            "culture": result.get("culture", ""),
            "key_objects": result.get("key_objects", []),
            "tags": result.get("tags", []),
            "direction": result.get("direction", ""),
            "performance": result.get("performance", ""),
            "continuity_score": float(result.get("continuity_score", 0.0) or 0.0),
            "continuity_notes": result.get("continuity_notes", ""),
            "notes": result.get("notes", ""),
            "internal_segments": result.get("key_frames", []),
        }
