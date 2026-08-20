"""
Face Engine: 人脸检测 + 人脸库建库 + 身份识别
基于 InsightFace 底层接口（det_10g + w600k_r50）
"""
import os
import sys
import cv2
import zipfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import insightface
from insightface.model_zoo import get_model


# ---- 模型自动下载（国内网络适配）----
# 优先从 ModelScope 下载，避免 GitHub 国内连不通

MODELSCOPE_MODEL_ID = "LumilioPhotos/buffalo_l"
GITHUB_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]


def _download_from_modelscope(root_dir: str, model_name: str = "buffalo_l") -> bool:
    """从 ModelScope 下载并解压模型，成功返回 True"""
    try:
        from modelscope import snapshot_download
        print(f"[FaceEngine] Downloading from ModelScope: {MODELSCOPE_MODEL_ID}")
        downloaded_dir = snapshot_download(MODELSCOPE_MODEL_ID)
        
        # ModelScope 下载的结构是: downloaded_dir/onnx/detection.fp32.onnx, recognition.fp32.onnx
        onnx_dir = os.path.join(downloaded_dir, "onnx")
        if not os.path.exists(onnx_dir):
            print(f"[FaceEngine] ModelScope download path unexpected: {downloaded_dir}")
            return False
        
        # 创建目标目录并复制/重命名文件
        model_dir = os.path.join(root_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        det_src = os.path.join(onnx_dir, "detection.fp32.onnx")
        rec_src = os.path.join(onnx_dir, "recognition.fp32.onnx")
        det_dst = os.path.join(model_dir, "det_10g.onnx")
        rec_dst = os.path.join(model_dir, "w600k_r50.onnx")
        
        if os.path.exists(det_src):
            import shutil
            shutil.copy2(det_src, det_dst)
            print(f"[FaceEngine] Copied detection model -> {det_dst}")
        if os.path.exists(rec_src):
            import shutil
            shutil.copy2(rec_src, rec_dst)
            print(f"[FaceEngine] Copied recognition model -> {rec_dst}")
        
        # 创建占位文件（insightface 会检查这些）
        for placeholder in ["genderage.onnx", "2d106det.onnx", "1k3d68.onnx"]:
            placeholder_path = os.path.join(model_dir, placeholder)
            if not os.path.exists(placeholder_path):
                open(placeholder_path, 'w').close()
        
        return os.path.exists(det_dst) and os.path.exists(rec_dst)
    except Exception as e:
        print(f"[FaceEngine] ModelScope download failed: {e}")
        return False


def _ensure_model_local(root_dir: str, model_name: str = "buffalo_l") -> str:
    """确保模型已下载到本地，返回模型目录路径"""
    model_dir = os.path.join(root_dir, model_name)
    det_file = os.path.join(model_dir, "det_10g.onnx")
    rec_file = os.path.join(model_dir, "w600k_r50.onnx")

    if os.path.exists(det_file) and os.path.exists(rec_file):
        return model_dir

    print(f"[FaceEngine] Model not found locally, attempting download...")
    os.makedirs(root_dir, exist_ok=True)

    # 方法1: ModelScope（国内源，首选）
    if _download_from_modelscope(root_dir, model_name):
        print(f"[FaceEngine] Model ready at {model_dir}")
        return model_dir

    # 方法2: 直接 GitHub（国内大概率失败，备用）
    zip_path = os.path.join(root_dir, f"{model_name}.zip")
    downloaded = False
    for url in GITHUB_URLS:
        try:
            print(f"[FaceEngine] Trying GitHub: {url[:60]}...")
            import urllib.request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, zip_path)
            downloaded = True
            break
        except Exception as e:
            print(f"[FaceEngine] GitHub failed: {e}")
            continue

    if downloaded:
        print(f"[FaceEngine] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(root_dir)
        print(f"[FaceEngine] Extracted to {model_dir}")
        return model_dir

    # 全部失败
    print("\n" + "="*60)
    print("[FaceEngine] ERROR: Cannot download model automatically.")
    print("Please run the following command manually:")
    print(f"  modelscope download --model {MODELSCOPE_MODEL_ID}")
    print(f"Then copy onnx/detection.fp32.onnx -> {det_file}")
    print(f"     and onnx/recognition.fp32.onnx -> {rec_file}")
    print("="*60 + "\n")
    sys.exit(1)


class FaceEngine:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_thresh: float = 0.5,
        match_thresh: float = 0.65,
        device: str = "cuda",
        model_root: Optional[str] = None,
    ):
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.match_thresh = match_thresh
        self.device = device
        self.model_root = model_root or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "insightface", "models"
        )
        self.det_model = None      # 检测模型
        self.rec_model = None      # 识别模型
        self.gallery: Dict[str, np.ndarray] = {}  # name -> 512-D embedding
        self._loaded = False

    def load(self):
        """加载检测+识别模型（显存占用 ~1GB）"""
        if self._loaded:
            return

        model_dir = _ensure_model_local(self.model_root, self.model_name)

        det_path = os.path.join(model_dir, "det_10g.onnx")
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")

        print(f"[FaceEngine] Loading detection model...")
        self.det_model = get_model(det_path)
        ctx_id = 0 if self.device == "cuda" else -1
        self.det_model.prepare(ctx_id=ctx_id, input_size=(640, 640))

        print(f"[FaceEngine] Loading recognition model...")
        self.rec_model = get_model(rec_path)
        self.rec_model.prepare(ctx_id=ctx_id)

        self._loaded = True
        print(f"[FaceEngine] Models loaded from {model_dir}")

    def unload(self):
        """卸载模型，释放显存"""
        if self.det_model is not None:
            del self.det_model
            self.det_model = None
        if self.rec_model is not None:
            del self.rec_model
            self.rec_model = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        print("[FaceEngine] unloaded")

    # ------------------------------------------------------------------
    # 建库
    # ------------------------------------------------------------------
    def build_gallery(self, references_dir: str) -> List[str]:
        """
        从 references_dir 建人脸库
        目录结构: references/张三/front.jpg side.jpg back.jpg
        返回成功入库的人物名称列表
        """
        self.load()
        ref_path = Path(references_dir)
        if not ref_path.exists():
            print(f"[FaceEngine] Warning: references dir not found: {references_dir}")
            return []

        registered = []
        for person_dir in sorted(ref_path.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            embeddings = []
            for img_file in person_dir.iterdir():
                if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                try:
                    pil_img = Image.open(str(img_file))
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[FaceEngine] Cannot load {img_file.name}: {e}")
                    continue
                if img is None:
                    continue
                faces = self._detect_faces(img)
                if not faces:
                    print(f"[FaceEngine] No face in {img_file.name}, skipped")
                    continue
                # 取最大人脸（按 bbox 面积排序）
                faces = sorted(faces, key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]), reverse=True)
                emb = self._get_embedding(img, faces[0])
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)  # 归一化
                self.gallery[name] = avg_emb
                registered.append(name)
                print(f"[FaceEngine] Registered '{name}' with {len(embeddings)} reference(s)")
            else:
                print(f"[FaceEngine] Failed to register '{name}', no valid face found")

        return registered

    # ------------------------------------------------------------------
    # 核心：检测 + 特征提取
    # ------------------------------------------------------------------
    def _detect_faces(self, img: np.ndarray) -> List[Dict]:
        """
        检测人脸，返回包含 bbox 和关键点的 face 对象列表
        """
        try:
            bboxes, kpss = self.det_model.detect(img)
            if bboxes is None or len(bboxes) == 0:
                return []
            results = []
            for i, bbox in enumerate(bboxes):
                score = float(bbox[4])
                if score < self.det_thresh:
                    continue
                # 构建 face 字典（兼容 insightface 接口）
                face = {
                    'bbox': bbox[:4],
                    'kps': kpss[i] if kpss is not None and i < len(kpss) else None,
                    'det_score': score,
                }
                results.append(face)
            return results
        except Exception as e:
            print(f"[FaceEngine] detect warning: {e}")
            return []

    def _get_embedding(self, img: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        """
        从人脸提取 512-D embedding
        face: {'bbox': [...], 'kps': [...], 'det_score': ...}
        """
        try:
            # ArcFaceONNX.get() 需要 (img, face) 参数
            feat = self.rec_model.get(img, face)
            if feat is None:
                return None
            # 归一化
            feat = feat / np.linalg.norm(feat)
            return feat
        except Exception as e:
            # 备用：手动预处理 + forward
            try:
                x1, y1, x2, y2 = map(int, face['bbox'])
                face_img = img[y1:y2, x1:x2]
                if face_img.size == 0:
                    return None
                # resize to 112x112
                face_img = cv2.resize(face_img, (112, 112))
                # normalize
                face_img = face_img.astype(np.float32) / 255.0
                # standard ArcFace preprocessing
                face_img = (face_img - 0.5) / 0.5
                # HWC -> CHW
                face_img = np.transpose(face_img, (2, 0, 1))
                # add batch dim
                face_img = np.expand_dims(face_img, axis=0)
                feat = self.rec_model.forward(face_img)
                if isinstance(feat, tuple):
                    feat = feat[0]
                feat = feat.flatten()
                feat = feat / np.linalg.norm(feat)
                return feat
            except Exception as e2:
                print(f"[FaceEngine] embedding warning: {e} / fallback: {e2}")
                return None

    # ------------------------------------------------------------------
    # 视频处理
    # ------------------------------------------------------------------
    def process_video(self, video_path: str, sample_fps: int = 5) -> List[Dict]:
        """
        处理单个视频，返回人物列表（含出现时间段）
        """
        self.load()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[FaceEngine] Cannot open video: {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        frame_interval = max(1, int(video_fps / sample_fps))

        # 原始追踪数据: frame_idx -> list of (identity, conf)
        frame_results: Dict[int, List[Tuple[str, float]]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                faces = self._detect_faces(frame)
                matches = []
                for face in faces:
                    emb = self._get_embedding(frame, face)
                    if emb is None:
                        continue
                    best_name, best_sim = self._match(emb)
                    matches.append((best_name, float(best_sim)))
                frame_results[frame_idx] = matches
            frame_idx += 1

        cap.release()

        # 合并连续出现的时间段
        persons = self._merge_appearances(frame_results, video_fps, duration)
        return persons

    def _match(self, emb: np.ndarray) -> Tuple[str, float]:
        """与库比对，返回 (name, cosine_similarity)"""
        if not self.gallery:
            return ("unknown", 0.0)
        best_name = "unknown"
        best_sim = -1.0
        for name, ref_emb in self.gallery.items():
            sim = float(np.dot(emb, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim < self.match_thresh:
            return ("unknown", best_sim)
        return (best_name, best_sim)

    def _merge_appearances(self, frame_results: Dict[int, List[Tuple[str, float]]],
                           fps: float, duration: float) -> List[Dict]:
        """合并连续帧，生成带时间段的输出"""
        # 先按人名聚合所有出现的帧
        raw_tracks: Dict[str, List[Tuple[int, float]]] = {}
        for fidx, matches in frame_results.items():
            for name, conf in matches:
                raw_tracks.setdefault(name, []).append((fidx, conf))

        persons = []
        unknown_counter = 1

        for name, frames in raw_tracks.items():
            if name == "unknown":
                display_name = f"unknown_{unknown_counter}"
                unknown_counter += 1
            else:
                display_name = name

            # 按帧序号排序，切分连续段（允许 3 帧间隔的断裂合并）
            frames = sorted(frames, key=lambda x: x[0])
            gaps = 3  # 允许 gap
            segments = []
            cur_seg = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i][0] - frames[i-1][0] <= gaps:
                    cur_seg.append(frames[i])
                else:
                    segments.append(cur_seg)
                    cur_seg = [frames[i]]
            segments.append(cur_seg)

            appearances = []
            for seg in segments:
                start_f, end_f = seg[0][0], seg[-1][0]
                avg_conf = sum(c for _, c in seg) / len(seg)
                appearances.append({
                    "start": round(start_f / fps, 2),
                    "end": min(round(end_f / fps, 2), duration),
                    "frame_count": len(seg),
                    "confidence": round(avg_conf, 3)
                })

            # 最终取平均置信度
            avg_conf = sum(c for _, c in frames) / len(frames)
            persons.append({
                "identity": display_name,
                "confidence": round(avg_conf, 3),
                "appearances": appearances
            })

        # 如果有多个 unknown，合并为一个
        unknowns = [p for p in persons if p["identity"].startswith("unknown")]
        if len(unknowns) > 1:
            knowns = [p for p in persons if not p["identity"].startswith("unknown")]
            merged_apps = []
            for u in unknowns:
                merged_apps.extend(u["appearances"])
            # 按时间排序合并
            merged_apps = sorted(merged_apps, key=lambda x: x["start"])
            persons = knowns + [{
                "identity": "unknown",
                "confidence": round(sum(a["confidence"] for a in merged_apps)/len(merged_apps), 3) if merged_apps else 0,
                "appearances": merged_apps
            }]

        return persons
