"""
本地人脸识别服务封装

基于 video-intelligence-extractor 的 FaceEngine（InsightFace），
从 reference_images 目录建库，识别 Shot 中出现的人物身份。
"""
import os
from typing import List, Dict, Any, Optional

from src.utils import logger


class FaceService:
    """基于 InsightFace 的人脸识别服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local_cfg = config.get("models", {}).get("local", {})
        face_cfg = local_cfg.get("face", {})
        paths = config.get("paths", {})

        self.enabled = bool(local_cfg.get("enabled", False)) and bool(face_cfg.get("enabled", True))
        self.device = local_cfg.get("device", "cuda")
        self.cache_dir = local_cfg.get("cache_dir")
        self.model_name = face_cfg.get("model_name", "buffalo_l")
        self.det_thresh = face_cfg.get("det_thresh", 0.5)
        self.match_thresh = face_cfg.get("match_thresh", 0.65)
        self.refs_dir = face_cfg.get("refs_dir") or paths.get("reference_images", "./refs")

        self._engine = None
        self._gallery_built = False

    def _get_engine(self):
        if self._engine is None:
            from src.local_models.face_engine import FaceEngine
            model_root = os.path.join(self.cache_dir, "insightface", "models") if self.cache_dir else None
            self._engine = FaceEngine(
                model_name=self.model_name,
                det_thresh=self.det_thresh,
                match_thresh=self.match_thresh,
                device=self.device,
                model_root=model_root,
            )
        return self._engine

    def _ensure_gallery(self):
        if not self._gallery_built and os.path.isdir(self.refs_dir):
            try:
                engine = self._get_engine()
                registered = engine.build_gallery(self.refs_dir)
                logger.info(f"[FaceService] 人脸库注册完成: {registered}")
                self._gallery_built = True
            except Exception as e:
                logger.warning(f"[FaceService] 人脸库构建失败: {e}")
                self._gallery_built = True  # 避免反复重试

    def identify_characters(
        self,
        video_path: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> List[str]:
        """
        识别视频中出现的人物身份。
        如果提供 start_sec/end_sec，只返回与该时间段有重叠的识别结果。
        """
        if not self.enabled:
            return []

        self._ensure_gallery()
        if not os.path.isdir(self.refs_dir):
            logger.debug(f"[FaceService] 参考图目录不存在，跳过人脸识别: {self.refs_dir}")
            return []

        try:
            engine = self._get_engine()
            persons = engine.process_video(video_path, sample_fps=5)
        except Exception as e:
            logger.error(f"[FaceService] 人脸识别失败 {os.path.basename(video_path)}: {e}")
            return []

        names = set()
        for person in persons:
            identity = person.get("identity", "")
            if identity.startswith("unknown"):
                continue
            if start_sec is not None and end_sec is not None:
                # 过滤与 shot 时间段有重叠的 appearance
                has_overlap = any(
                    app.get("end", 0) >= start_sec and app.get("start", 0) <= end_sec
                    for app in person.get("appearances", [])
                )
                if not has_overlap:
                    continue
            names.add(identity)

        return sorted(names)

    def unload(self):
        if self._engine is not None:
            self._engine.unload()
            self._engine = None
