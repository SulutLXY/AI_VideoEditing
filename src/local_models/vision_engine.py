"""
Vision Engine: 本地视频内容理解
基于 Qwen2.5-VL-3B-Instruct (4-bit 量化)

职责：
- 对单个视频片段（Shot 级别）抽取关键帧并调用本地 VLM 分析。
- 输出与项目 Shot 模型对齐的完整字段。
"""
import os
import re
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# 设置 HuggingFace 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 注意：cv2 / torch / PIL / transformers 只在本地模型启用时才会被加载，
# 因此允许在顶层 import；若当前环境未安装，只要不导入本模块就不会报错。
import cv2
import torch
from PIL import Image


class VisionEngine:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        model_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "vision"
        )
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """加载模型（4-bit 量化后 ~3-4GB 显存）"""
        if self._loaded:
            return

        if not torch.cuda.is_available():
            print(
                "[VisionEngine] WARNING: CUDA not available, "
                "skipping vision model (install CUDA PyTorch for full features)"
            )
            self._loaded = True
            return

        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            from modelscope import snapshot_download

            model_path = self._resolve_model_path()
            print(f"[VisionEngine] Loading model from: {model_path}")

            quant_config = None
            if self.load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if not self.load_in_4bit else torch.float32,
                device_map="auto" if self.load_in_4bit else self.device,
                quantization_config=quant_config,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self._loaded = True
            print("[VisionEngine] Qwen2.5-VL loaded")
        except Exception as e:
            print(f"[VisionEngine] WARNING: Model load failed ({e}), using fallback mode")
            self._loaded = True

    def unload(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()
        print("[VisionEngine] unloaded")

    def _resolve_model_path(self) -> str:
        """解析模型路径，优先级：model_path > cache_dir/model_id > 自动下载"""
        if self.model_path and os.path.exists(os.path.join(self.model_path, "config.json")):
            return self.model_path

        safe_name = self.model_id.replace("/", "--")
        cached = os.path.join(self.cache_dir, safe_name)
        if os.path.exists(os.path.join(cached, "config.json")):
            return cached

        modelscope_id = self.model_id.replace("Qwen/", "qwen/").replace("Microsoft/", "microsoft/")
        print(f"[VisionEngine] Model not found locally, downloading from ModelScope: {modelscope_id}")
        print(f"[VisionEngine] This will download ~6GB, please wait...")
        from modelscope import snapshot_download
        downloaded = snapshot_download(modelscope_id, cache_dir=self.cache_dir)
        return downloaded

    @staticmethod
    def _frame_indices(
        cap: "cv2.VideoCapture",
        count: int,
        interval: float,
        max_frames: int,
    ) -> List[int]:
        """计算抽帧索引（首/中/尾 or 每 interval 秒一帧，超上限均匀重采）"""
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0

        if duration <= 1.0 or count <= 1:
            # 短镜头：原首/中/尾逻辑
            if count >= 3:
                indices = [0, total_frames // 2, total_frames - 1]
            else:
                step = max(1, total_frames // count)
                indices = [min(i * step, total_frames - 1) for i in range(count)]
        else:
            # 长镜头：每 interval 秒一帧
            step_frames = max(1, int(round(interval * fps)))
            indices = list(range(0, total_frames, step_frames))

        # 容器元数据可能虚报帧数（如 -c copy 切分的片段），先探测实际可读的最后一帧
        last_ok = indices[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_ok)
        ret, _ = cap.read()
        while not ret and last_ok > 0:
            last_ok = max(0, last_ok - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_ok)
            ret, _ = cap.read()

        # 超上限：在可读范围内均匀重采到 max_frames 帧
        if len(indices) > max_frames:
            indices = sorted(set(
                int(round(i * last_ok / (max_frames - 1)))
                for i in range(max_frames)
            ))

        # 索引钳制到可读范围内并去重（避免多帧钳到同一位置）
        seen_idx = set()
        return [min(i, last_ok) for i in indices if not (min(i, last_ok) in seen_idx or seen_idx.add(min(i, last_ok)))]

    @staticmethod
    def _read_frames_at_indices(
        cap: "cv2.VideoCapture",
        indices: List[int],
        fps: float,
    ) -> List[Tuple[float, "Image.Image"]]:
        """按索引读帧并附上解码器实际时间戳"""
        frames = []
        last_ts = -1.0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # 用解码器实际位置取时间戳，避免帧数元数据虚报导致的时间漂移
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = msec / 1000.0 if msec and msec > 0 else idx / fps
            if timestamp <= last_ts:
                continue
            last_ts = timestamp
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((round(timestamp, 2), img))
        return frames

    # 进模型前的降采样规格：视频帧 480p、用户参考图最大 720p
    FRAME_BOX = (854, 480)
    REF_BOX = (1280, 720)

    @classmethod
    def preextract_frames(
        cls,
        video_path: str,
        out_dir: str,
        count: int = 3,
        interval: float = 0.5,
        max_frames: int = 16,
    ) -> Optional[Dict]:
        """阶段A：预抽帧落盘（480p jpg + meta.json），供后续 LLM 分析直接读图。

        已存在 meta.json 时跳过（幂等，支持中断续跑）。
        返回 meta dict；失败返回 None。
        """
        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        indices = cls._frame_indices(cap, count, interval, max_frames)
        frames = cls._read_frames_at_indices(cap, indices, fps)
        cap.release()
        if not frames:
            return None

        os.makedirs(out_dir, exist_ok=True)
        times = []
        last_img = None
        for i, (ts, img) in enumerate(frames):
            img = cls._downscale(img, cls.FRAME_BOX)
            img.save(os.path.join(out_dir, f"f_{i:04d}.jpg"), quality=90)
            times.append(ts)
            last_img = img

        meta = {
            "source": os.path.abspath(video_path),
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "times": times,
            "frame_count": len(times),
            "width": last_img.width,
            "height": last_img.height,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    @staticmethod
    def load_frames_from_dir(frames_dir: str) -> List[Tuple[float, "Image.Image"]]:
        """从预抽帧目录读回帧序列（jpg + meta.json）"""
        meta_path = os.path.join(frames_dir, "meta.json")
        if not os.path.exists(meta_path):
            return []
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        frames = []
        for i, ts in enumerate(meta.get("times", [])):
            p = os.path.join(frames_dir, f"f_{i:04d}.jpg")
            if os.path.exists(p):
                frames.append((float(ts), Image.open(p).convert("RGB")))
        return frames

    @staticmethod
    def extract_keyframes(
        video_path: str,
        count: int = 3,
        interval: float = 0.5,
        max_frames: int = 16,
    ) -> List[Tuple[float, Image.Image]]:
        """提取关键帧 (timestamp_sec, PIL.Image)，原始分辨率（缩放在进模型前做）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        indices = VisionEngine._frame_indices(cap, count, interval, max_frames)
        frames = VisionEngine._read_frames_at_indices(cap, indices, fps)
        cap.release()
        return frames

    def _build_prompt(self) -> str:
        return (
            "你是一位专业的影视镜头内容分析师。请仔细观察这张视频截图，并以 JSON 格式输出以下字段。"
            "不要输出任何其他文字，只输出纯 JSON。\n\n"
            "{\n"
            '  "shot_size": "特写/近景/中景/全景/大全景/无法判断",\n'
            '  "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦/无法判断",\n'
            '  "camera_position": "机位描述，如柜台正面/门口右侧/特写机位/无法判断",\n'
            '  "direction": "人物朝向或运动方向，如从左向右/面向镜头/背对镜头/静止",\n'
            '  "action": "画面主体主要动作（一句话）",\n'
            '  "action_details": "动作细节：具体肢体动作、手势、走位、互动对象等",\n'
            '  "emotion": "整体情绪，如焦虑/紧张/温情/平静/兴奋",\n'
            '  "performance": "表演评估：自然度、情绪强度、是否入戏、有无表演痕迹",\n'
            '  "location": "场景地点",\n'
            '  "time_of_day": "白天/傍晚/夜晚/室内灯光/无法判断",\n'
            '  "framing": "构图描述，如居中构图/三分法/前景遮挡/对称构图/过肩镜头",\n'
            '  "lighting": "光效描述",\n'
            '  "color_tone": "色调描述",\n'
            '  "style": "风格标签",\n'
            '  "atmosphere": "氛围描述",\n'
            '  "culture": "文化/时代背景",\n'
            '  "key_objects": ["关键道具1", "关键道具2"],\n'
            '  "tags": ["标签1", "标签2"],\n'
            '  "continuity_score": 0.85,\n'
            '  "continuity_notes": "镜头内部连续性说明：是否一镜到底、有无跳切/穿帮/方向跳变",\n'
            '  "notes": "其他值得注意的信息"\n'
            "}"
        )

    def _build_multi_prompt(
        self,
        frame_times: List[float],
        script_context: Optional[str] = None,
        ref_entries: Optional[List[Tuple[str, str]]] = None,
        audio_profile: Optional[Dict] = None,
    ) -> str:
        """多图联合分析 prompt：帧序列（带时间戳）+ 角色参考图（带说明）+ 剧情背景 + 音频分析"""
        ref_entries = ref_entries or []
        parts = []
        header = (
            "你是一位专业的影视镜头内容分析师。接下来会给你若干张图片："
        )
        if ref_entries:
            header += f"前 {len(ref_entries)} 张是角色参考图，"
        header += f"其余 {len(frame_times)} 张是同一视频片段按时间顺序排列的帧"
        if len(frame_times) >= 2:
            header += f"（相邻帧间隔约 {frame_times[1] - frame_times[0]:.2f} 秒）"
        header += (
            "。请综合所有帧的时序变化进行分析：注意动作演变、镜头运动方向、情绪转折"
            + ("，以及画面人物与角色参考图的对应关系" if ref_entries else "")
            + "。并以 JSON 格式输出以下字段。不要输出任何其他文字，只输出纯 JSON。\n"
        )
        parts.append(header)

        if ref_entries:
            ref_lines = []
            for i, (name, desc) in enumerate(ref_entries):
                line = f"图R{i + 1}: 角色「{name}」的参考形象"
                if desc:
                    line += f"。说明：{desc}"
                ref_lines.append(line)
            parts.append(
                f"【角色参考图】\n" + "\n".join(ref_lines) + "\n"
                "若画面中人物与某张参考图是同一人（含换装/易容/不同角度），"
                "请在 characters 字段中使用该角色名。\n"
            )

        if script_context:
            parts.append(
                f"【剧情背景】\n{script_context}\n"
                "请结合剧情背景判断画面动作与情绪的剧情含义，而不是只描述表面画面。\n"
            )

        if audio_profile:
            parts.append(self._build_audio_section(audio_profile))

        frame_desc = "\n".join(
            f"图{i + 1} (t={t:.2f}s)" for i, t in enumerate(frame_times)
        )
        parts.append(f"【视频帧】\n{frame_desc}\n")

        parts.append(
            "{\n"
            '  "shot_size": "特写/近景/中景/全景/大全景/无法判断",\n'
            '  "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦/无法判断",\n'
            '  "camera_position": "机位描述",\n'
            '  "direction": "人物朝向或运动方向",\n'
            '  "action": "画面主体主要动作（一句话，结合时序）",\n'
            '  "action_details": "动作细节与演变：从第一帧到最后一帧的动作变化过程",\n'
            '  "emotion": "整体情绪（注意帧间情绪转折）",\n'
            '  "performance": "表演评估：自然度、情绪强度、是否入戏",\n'
            '  "location": "场景地点",\n'
            '  "time_of_day": "白天/傍晚/夜晚/室内灯光/无法判断",\n'
            '  "characters": ["画面中出现的角色名（与参考图对应，无匹配则描述外貌）"],\n'
            '  "framing": "构图描述",\n'
            '  "lighting": "光效描述",\n'
            '  "color_tone": "色调描述",\n'
            '  "style": "风格标签",\n'
            '  "atmosphere": "氛围描述",\n'
            '  "culture": "文化/时代背景",\n'
            '  "key_objects": ["关键道具1", "关键道具2"],\n'
            '  "tags": ["标签1", "标签2"],\n'
            '  "continuity_score": 0.85,\n'
            '  "continuity_notes": "镜头内部连续性说明：是否一镜到底、有无跳切/穿帮/方向跳变",\n'
            '  "notes": "其他值得注意的信息"\n'
            "}"
        )
        return "\n".join(parts)

    @staticmethod
    def _build_audio_section(profile: Dict) -> str:
        """把阶段A2的音频分析档案转成 prompt 段落"""
        event = profile.get("event")
        has_speech = bool(profile.get("has_speech"))
        language = profile.get("language")
        emotion = profile.get("emotion")
        text = (profile.get("text") or "").strip()
        transcript = profile.get("transcript") or []

        if not has_speech and not text:
            if event and event != "Speech":
                env_desc = {"Music": "背景音乐，无人物对白",
                            "BGM": "背景音乐，无人物对白",
                            "Noise": "环境噪声，无人物对白"}.get(event, f"音频事件[{event}]，无人物对白")
            else:
                # SenseVoice 对纯 BGM/环境音常返回空，用音量兜底区分
                mean_db = profile.get("mean_volume_db")
                if mean_db is not None and mean_db > -45.0:
                    env_desc = "有背景音乐/环境音（无人物对白）"
                else:
                    env_desc = "几乎无声（无对白无显著环境音）"
            return (
                f"【音频分析】\n该片段{env_desc}。"
                "画面分析时请以视觉信息为准，emotion/atmosphere 从画面判断。\n"
            )

        bits = []
        if language:
            bits.append(f"语言: {language}")
        if emotion:
            bits.append(f"情绪: {emotion}")
        if event:
            bits.append(f"事件: {event}")
        lines = [f"该片段含人物语音。{'，'.join(bits)}。"]

        # 带时间戳的台词清单（最多 5 条，避免 prompt 过长）
        lines_with_ts = [
            f"  ({t.get('start', 0):.1f}s-{t.get('end', 0):.1f}s) {t.get('text', '')}"
            for t in transcript if t.get("text")
        ][:5]
        if lines_with_ts:
            lines.append("台词清单：\n" + "\n".join(lines_with_ts))
        elif text:
            lines.append(f"台词内容：{text}")

        lines.append(
            "请对照画面核对：说话人是否为画面中人物、口型/动作与台词是否吻合、"
            "情绪标签与画面情绪是否一致；若冲突以画面为准并在 notes 中说明。"
        )
        return "【音频分析】\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _load_audio_profile(frames_dir: Optional[str]) -> Optional[Dict]:
        """从预抽帧目录读取音频分析档案（阶段A2产物）"""
        if not frames_dir:
            return None
        p = os.path.join(frames_dir, "audio_profile.json")
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _build_content_prompt(
        self,
        frame_times: List[float],
        script_context: Optional[str] = None,
        audio_profile: Optional[Dict] = None,
    ) -> str:
        """第一步·内容分析 prompt：纯视频帧，专注动作与画面细节。

        人物只要求泛化描述（性别/衣着/体型），身份确认由第二步专门负责，
        避免 3B 小模型在同一调用里混淆任务。
        """
        parts = []
        header = (
            "你是一位专业的影视镜头内容分析师。接下来给你的是同一视频片段"
            f"按时间顺序排列的 {len(frame_times)} 帧"
        )
        if len(frame_times) >= 2:
            header += f"（相邻帧间隔约 {frame_times[1] - frame_times[0]:.2f} 秒）"
        header += (
            "。请综合所有帧的时序变化进行分析：注意动作演变、镜头运动方向、情绪转折、"
            "画面细节（服装、道具、表情、环境）。并以 JSON 格式输出以下字段。"
            "不要输出任何其他文字，只输出纯 JSON。\n"
        )
        parts.append(header)

        if script_context:
            parts.append(
                f"【剧情背景】\n{script_context}\n"
                "请结合剧情背景判断画面动作与情绪的剧情含义，而不是只描述表面画面。\n"
            )

        if audio_profile:
            parts.append(self._build_audio_section(audio_profile))

        frame_desc = "\n".join(
            f"图{i + 1} (t={t:.2f}s)" for i, t in enumerate(frame_times)
        )
        parts.append(f"【视频帧】\n{frame_desc}\n")

        parts.append(
            "{\n"
            '  "shot_size": "特写/近景/中景/全景/大全景/无法判断",\n'
            '  "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦/无法判断",\n'
            '  "camera_position": "机位描述",\n'
            '  "direction": "人物朝向或运动方向",\n'
            '  "action": "画面主体主要动作（一句话，结合时序）",\n'
            '  "action_details": "动作细节与演变：从第一帧到最后一帧的动作变化过程",\n'
            '  "emotion": "整体情绪（注意帧间情绪转折）",\n'
            '  "performance": "表演评估：自然度、情绪强度、是否入戏",\n'
            '  "location": "场景地点",\n'
            '  "time_of_day": "白天/傍晚/夜晚/室内灯光/无法判断",\n'
            "  \"characters\": [\"画面中人物的泛化描述（如：年轻男性，深蓝劲装，高马尾；不要猜测具体名字）\"],\n"
            '  "framing": "构图描述",\n'
            '  "lighting": "光效描述",\n'
            '  "color_tone": "色调描述",\n'
            '  "style": "风格标签",\n'
            '  "atmosphere": "氛围描述",\n'
            '  "culture": "文化/时代背景",\n'
            '  "key_objects": ["关键道具1", "关键道具2"],\n'
            '  "tags": ["标签1", "标签2"],\n'
            '  "continuity_score": 0.85,\n'
            '  "continuity_notes": "镜头内部连续性说明：是否一镜到底、有无跳切/穿帮/方向跳变",\n'
            '  "notes": "其他值得注意的信息"\n'
            "}"
        )
        return "\n".join(parts)

    def _build_identity_prompt(
        self,
        ref_entries: List[Tuple[str, str]],
    ) -> str:
        """第二步·身份确认 prompt：视频抽样帧 + 候选参考图，只输出身份匹配结果。"""
        ref_lines = []
        for i, (name, desc) in enumerate(ref_entries):
            line = f"图R{i + 1}: 「{name}」"
            if desc:
                line += f"。特征：{desc}"
            ref_lines.append(line)

        return (
            "你是一位影视角色对照专家。接下来给你若干张图片："
            f"前 3 张是同一视频片段的抽样帧（首/中/尾），"
            f"后 {len(ref_entries)} 张是候选角色参考图。"
            "请逐一对照：视频帧中出现的人物/动物分别与哪张参考图是同一对象"
            "（允许换装、不同角度、不同动作，依据脸型、发型、服装、体型、毛色等特征判断）。"
            "没有对应参考图的对象列入 unknown 并给出泛化描述。"
            "不要输出任何其他文字，只输出纯 JSON：\n\n"
            "{\n"
            '  "identities": [\n'
            '    {"ref": "参考图角色名", "present": true, "basis": "判断依据（一句话）"}\n'
            "  ],\n"
            '  "unknown": ["未匹配参考图的人物/动物泛化描述"]\n'
            "}"
        )

    def _parse_response(self, text: str) -> Dict:
        """解析模型返回，优先按 JSON，失败则正则兜底"""
        text = text.strip()
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]

        try:
            data = json.loads(json_text.strip())
            if isinstance(data, dict):
                return self._normalize_result(data)
        except Exception:
            pass

        # 兜底正则
        result = {
            "shot_size": "无法判断",
            "camera_movement": "无法判断",
            "action": "",
            "location": "",
            "emotion": "",
            "performance": "",
            "action_details": "",
            "direction": "",
            "framing": "",
            "lighting": "",
            "color_tone": "",
            "style": "",
            "atmosphere": "",
            "culture": "",
            "time_of_day": "",
            "continuity_score": 0.0,
            "continuity_notes": "",
            "notes": "",
            "key_objects": [],
            "tags": [],
        }
        patterns = {
            "shot_size": r"(?:shot_size|景别)[:：]\s*(.+)",
            "camera_movement": r"(?:camera_movement|镜头)[:：]\s*(.+)",
            "action": r"(?:action|动作)[:：]\s*(.+)",
            "location": r"(?:location|场景)[:：]\s*(.+)",
            "emotion": r"(?:emotion|情绪)[:：]\s*(.+)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.MULTILINE)
            if m:
                result[key] = m.group(1).strip()
        return result

    @staticmethod
    def _normalize_result(data: Dict) -> Dict:
        """确保输出字段统一"""
        defaults = {
            "shot_size": "无法判断",
            "camera_movement": "无法判断",
            "camera_position": "",
            "direction": "",
            "action": "",
            "action_details": "",
            "emotion": "",
            "performance": "",
            "location": "",
            "time_of_day": "",
            "framing": "",
            "lighting": "",
            "color_tone": "",
            "style": "",
            "atmosphere": "",
            "culture": "",
            "characters": [],
            "key_objects": [],
            "tags": [],
            "continuity_score": 0.0,
            "continuity_notes": "",
            "notes": "",
        }
        out = {**defaults}
        for k, v in data.items():
            if k in out:
                out[k] = v
        # 3B 模型常把列表字段输出成字符串（如 characters: "角色男装"），
        # 统一纠正为列表，避免下游 list(str) 被拆成单字
        for k in ("characters", "key_objects", "tags"):
            v = out.get(k)
            if isinstance(v, str):
                v = v.strip()
                out[k] = [v] if v else []
            elif v is None:
                out[k] = []
        try:
            out["continuity_score"] = float(out["continuity_score"] or 0.0)
        except Exception:
            out["continuity_score"] = 0.0
        return out

    # 内容分析滑窗参数：每窗最多 5 帧、步进 3（重叠 2）、末尾剩余 ≤6 帧吸收进当前窗
    WINDOW_SIZE = 5
    WINDOW_STEP = 3
    WINDOW_TAIL = 6
    # 身份确认调用可选分辨率（原图永远保留，按需降采样）
    RESOLUTION_BOXES = {"480p": (854, 480), "720p": (1280, 720), "1080p": (1920, 1080)}

    @staticmethod
    def _frame_windows(
        frames: List[Tuple[float, "Image.Image"]],
        size: int = 5,
        step: int = 3,
        tail: int = 6,
    ) -> List[List[Tuple[float, "Image.Image"]]]:
        """把帧序列切成重叠滑窗。

        先取前 5 帧，步进 3 再取连续 5 帧；若取完后剩余帧数 ≤6，
        则把剩余帧（含最后一帧）全部纳入当前窗口，保证末尾帧必被分析。
        """
        n = len(frames)
        if n <= size:
            return [frames]
        windows = []
        start = 0
        while start < n:
            end = start + size
            if 0 < n - end <= tail:
                end = n
            windows.append(frames[start:end])
            if end >= n:
                break
            start += step
        return windows

    def process_video(
        self,
        video_path: str,
        keyframe_count: int = 3,
        frame_interval: float = 0.5,
        max_frames: int = 16,
        script_context: Optional[str] = None,
        ref_images: Optional[List[Tuple[str, "Image.Image"]]] = None,
        frames_dir: Optional[str] = None,
        identity_resolution: str = "480p",
    ) -> Dict:
        """处理单个视频片段，返回与 Shot 模型对齐的视觉分析结果

        两步走（避免 3B 小模型在单调用里顾此失彼）：
        - 第一步 内容分析：纯视频帧滑窗（≤5 帧/窗，步进 3，末尾吸收），
          专注动作/画面细节，人物只给泛化描述；多窗结果按时序合并
        - 第二步 身份确认：抽样帧 + 角色参考图单独一次调用，
          对照确认身份后替换 characters；失败则保留内容调用的泛化描述
        """
        self.load()

        fallback = {
            "shot_size": "无法判断",
            "camera_movement": "无法判断",
            "camera_position": "",
            "direction": "",
            "action": "",
            "action_details": "",
            "emotion": "",
            "performance": "",
            "location": "",
            "time_of_day": "",
            "characters": [],
            "framing": "",
            "lighting": "",
            "color_tone": "",
            "style": "",
            "atmosphere": "",
            "culture": "",
            "key_objects": [],
            "tags": [],
            "continuity_score": 0.0,
            "continuity_notes": "模型未加载或视频无法读取",
            "notes": "",
            "key_frames": [],
        }

        if self.model is None:
            print(f"[VisionEngine] Model not available, skipping {os.path.basename(video_path)}")
            return fallback

        keyframes: List[Tuple[float, "Image.Image"]] = []
        audio_profile: Optional[Dict] = None
        if frames_dir:
            keyframes = self.load_frames_from_dir(frames_dir)
            if keyframes:
                logger.info(
                    f"[VisionEngine] 使用预抽帧: {os.path.basename(frames_dir)} "
                    f"({len(keyframes)} 帧)"
                )
                audio_profile = self._load_audio_profile(frames_dir)
                if audio_profile:
                    logger.info(
                        f"[VisionEngine] 注入音频分析: "
                        f"{audio_profile.get('event') or 'Speech'}/"
                        f"{audio_profile.get('emotion') or '-'}/"
                        f"{'有台词' if audio_profile.get('text') else '无台词'}"
                    )
        if not keyframes:
            keyframes = [
                (t, self._downscale(img, self.FRAME_BOX))
                for t, img in self.extract_keyframes(
                    video_path,
                    count=keyframe_count,
                    interval=frame_interval,
                    max_frames=max_frames,
                )
            ]
        if not keyframes:
            return fallback

        ref_list = ref_images or []
        # 兼容 (name, img) 与 (name, img, desc) 两种条目；参考图单独限 720p
        ref_entries = [
            (r[0], r[2] if len(r) > 2 else "") for r in ref_list
        ]
        ref_imgs = [self._downscale(r[1], self.REF_BOX) for r in ref_list]
        frame_times = [t for t, _ in keyframes]

        # ===================== 第一步：内容分析（纯帧滑窗） =====================
        windows = self._frame_windows(keyframes)
        window_results: List[Dict] = []
        for wi, window in enumerate(windows):
            w_times = [t for t, _ in window]
            try:
                prompt_text = self._build_content_prompt(
                    w_times,
                    script_context=script_context,
                    audio_profile=audio_profile,
                )
                r = self._infer_multi([img for _, img in window], prompt_text)
                window_results.append(r)
                logger.info(f"[VisionEngine] 内容窗 {wi + 1}/{len(windows)} 完成 ({len(window)} 帧)")
            except Exception as e:
                logger.warning(f"[VisionEngine] 内容窗 {wi + 1} 失败，跳过该窗: {e}")
                print(f"[VisionEngine] 内容窗 {wi + 1} 失败，跳过该窗: {e}")

        if window_results:
            final = self._merge_window_results(window_results)
            final["_inference_mode"] = "windowed_content"
        else:
            logger.warning("[VisionEngine] 所有内容窗均失败，回退逐帧投票")
            prompt_text = self._build_prompt()
            all_results = [self._infer_single(img, prompt_text) for _, img in keyframes]
            final = self._vote_aggregate(all_results)
            final["_inference_mode"] = "per_frame_vote"

        # ===================== 第二步：身份确认（帧 + 参考图） =====================
        if ref_list:
            try:
                id_frames = self._identity_sample_frames(
                    video_path, keyframes, identity_resolution
                )
                id_prompt = self._build_identity_prompt(ref_entries)
                ident = self._infer_multi(id_frames + ref_imgs, id_prompt)
                confirmed = [i["ref"] for i in ident.get("identities", []) if i.get("present")]
                unknown = [u for u in ident.get("unknown", []) if u]
                final["characters"] = confirmed + unknown
                final["_identity_confirmed"] = confirmed
                logger.info(
                    f"[VisionEngine] 身份确认: 命中 {confirmed or '无'}，"
                    f"未匹配 {unknown or '无'}"
                )
            except Exception as e:
                logger.warning(f"[VisionEngine] 身份确认失败，保留内容分析的泛化描述: {e}")
                print(f"[VisionEngine] 身份确认失败，保留内容分析的泛化描述: {e}")

        final["key_frames"] = [
            {"timestamp": t, "description": final.get("action", "")}
            for t in frame_times
        ]
        return final

    def analyze_reference_image(self, img: "Image.Image") -> Dict:
        """参考图预分析：单图短输出，提取类型/性别/身份/特征/名字"""
        from src.local_models.ref_profiler import PROFILE_PROMPT
        img = self._downscale(img, (640, 640))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROFILE_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
            )
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return self._parse_ref_profile(response)

    def _parse_ref_profile(self, text: str) -> Dict:
        """解析参考图预分析的返回（entity_type/gender/identity/appearance/entity_name）。

        注意不能走 _parse_response：那会把结果规整成镜头分析字段，丢掉档案关键词。
        """
        text = text.strip()
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]
        try:
            data = json.loads(json_text.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # 正则兜底
        out: Dict[str, Any] = {}
        pats = {
            "entity_type": r"(?:entity_type|类型)[:：]\s*(.+)",
            "gender": r"(?:gender|性别)[:：]\s*(.+)",
            "entity_name": r"(?:entity_name|名字)[:：]\s*(.+)",
        }
        for k, pat in pats.items():
            m = re.search(pat, text)
            if m:
                out[k] = m.group(1).strip()
        app = re.search(r"(?:appearance|特征)[:：]\s*\[([^\]]*)\]", text)
        if app:
            out["appearance"] = [
                s.strip().strip("\"'") for s in app.group(1).split(",") if s.strip()
            ]
        return out

    def _identity_sample_frames(
        self,
        video_path: str,
        keyframes: List[Tuple[float, "Image.Image"]],
        resolution: str,
    ) -> List["Image.Image"]:
        """身份确认的抽样帧（首/中/尾，最多 3 张）。

        默认 480p 直接复用内容帧；选 720p/1080p 时回原视频按时间点重读，
        原图永不改动。
        """
        n = len(keyframes)
        idxs = sorted({0, n // 2, n - 1})
        box = self.RESOLUTION_BOXES.get(resolution, self.FRAME_BOX)

        if resolution == "480p":
            return [self._downscale(keyframes[i][1], box) for i in idxs]

        # 高分辨率：回原视频按时间戳读原图画质
        cap = cv2.VideoCapture(video_path)
        imgs = []
        try:
            for i in idxs:
                t = keyframes[i][0]
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    imgs.append(self._downscale(img, box))
        finally:
            cap.release()
        if not imgs:  # 读取失败兜底用内容帧
            imgs = [self._downscale(keyframes[i][1], box) for i in idxs]
        return imgs

    @staticmethod
    def _merge_window_results(results: List[Dict]) -> Dict:
        """按时序合并多个内容窗的分析结果。

        - action_details/continuity_notes：按窗顺序拼接（时序动作链）
        - 其余标量字段：非空投票，平票取第一个（emotion 平票取最后一个）
        - characters/key_objects/tags：并集
        - continuity_score：均值
        """
        from collections import Counter

        def vote(key: str, prefer_last: bool = False) -> str:
            vals = [str(r.get(key) or "").strip() for r in results]
            vals = [v for v in vals if v and v != "无法判断"] or \
                   [str(r.get(key) or "").strip() for r in results if str(r.get(key) or "").strip()]
            if not vals:
                return ""
            c = Counter(vals)
            top = c.most_common()
            if len(top) == 1 or top[0][1] > top[1][1]:
                return top[0][0]
            return vals[-1] if prefer_last else vals[0]

        def union(key: str) -> List[str]:
            seen: List[str] = []
            for r in results:
                v = r.get(key) or []
                if isinstance(v, str):
                    v = [v]
                for item in v:
                    item = str(item).strip()
                    if item and item not in seen:
                        seen.append(item)
            return seen

        def join_details(key: str) -> str:
            parts: List[str] = []
            for r in results:
                v = str(r.get(key) or "").strip()
                if v and v not in parts:
                    parts.append(v)
            return "；".join(parts)

        scores = [float(r.get("continuity_score") or 0.0) for r in results]
        merged = {
            "shot_size": vote("shot_size"),
            "camera_movement": vote("camera_movement"),
            "camera_position": vote("camera_position"),
            "direction": vote("direction"),
            "action": vote("action"),
            "action_details": join_details("action_details") or vote("action_details"),
            "emotion": vote("emotion", prefer_last=True),
            "performance": vote("performance"),
            "location": vote("location"),
            "time_of_day": vote("time_of_day"),
            "characters": union("characters"),
            "framing": vote("framing"),
            "lighting": vote("lighting"),
            "color_tone": vote("color_tone"),
            "style": vote("style"),
            "atmosphere": vote("atmosphere"),
            "culture": vote("culture"),
            "key_objects": union("key_objects"),
            "tags": union("tags"),
            "continuity_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "continuity_notes": join_details("continuity_notes"),
            "notes": vote("notes"),
        }
        return merged

    @staticmethod
    def _downscale(img: "Image.Image", box: Tuple[int, int]) -> "Image.Image":
        """把图片等比缩进 box（宽, 高）上限内再进 processor。

        视频帧用 FRAME_BOX(854x480)、参考图用 REF_BOX(1280x720)。
        曾漏掉缩放把接近 1080p 的原图直接喂给模型（每图约 1280 视觉 token），
        缩到 480p 后每图仅约 135 token，20 图的多图推理 prefill 和
        KV cache 压力降到约 1/9。
        """
        if img.width <= box[0] and img.height <= box[1]:
            return img
        img = img.copy()
        img.thumbnail(box, Image.LANCZOS)
        return img

    def _infer_multi(self, images: List["Image.Image"], prompt_text: str) -> Dict:
        """多图联合推理（参考图 + 帧序列一次请求；缩放已在 process_video 按档位做好）"""
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": prompt_text}]
                ),
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        logger.info(
            f"[VisionEngine] 多图推理完成: {len(images)} 图, "
            f"输入 {inputs.input_ids.shape[1]} token, "
            f"生成 {generated_ids.shape[1] - inputs.input_ids.shape[1]} token, "
            f"耗时 {time.time() - t0:.1f}s"
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_response(response)

    def _infer_single(self, image: Image.Image, prompt_text: str) -> Dict:
        """单张图片推理（兜底逐帧投票路径，只处理视频帧）"""
        image = self._downscale(image, self.FRAME_BOX)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return self._parse_response(response)

    @staticmethod
    def _vote_aggregate(results: List[Dict]) -> Dict:
        """对多帧结果投票聚合"""
        if not results:
            return {}

        def most_common(key: str, exclude: Tuple[str, ...] = ("", "无法判断")) -> str:
            vals = [r.get(key, "") for r in results if r.get(key, "") not in exclude]
            if not vals:
                return ""
            return Counter(vals).most_common(1)[0][0]

        def longest(key: str) -> str:
            vals = [str(r.get(key, "")).strip() for r in results if r.get(key, "")]
            return max(vals, key=len) if vals else ""

        def union_list(key: str) -> List[str]:
            seen = set()
            out = []
            for r in results:
                for item in r.get(key, []) or []:
                    s = str(item).strip()
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
            return out

        def avg_score(key: str) -> float:
            vals = []
            for r in results:
                try:
                    vals.append(float(r.get(key, 0.0)))
                except Exception:
                    pass
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        return {
            "shot_size": most_common("shot_size"),
            "camera_movement": most_common("camera_movement"),
            "camera_position": most_common("camera_position"),
            "direction": most_common("direction"),
            "action": most_common("action"),
            "action_details": longest("action_details"),
            "emotion": most_common("emotion"),
            "performance": longest("performance"),
            "location": most_common("location"),
            "time_of_day": most_common("time_of_day"),
            "framing": most_common("framing"),
            "lighting": most_common("lighting"),
            "color_tone": most_common("color_tone"),
            "style": most_common("style"),
            "atmosphere": longest("atmosphere"),
            "culture": most_common("culture"),
            "characters": union_list("characters"),
            "key_objects": union_list("key_objects"),
            "tags": union_list("tags"),
            "continuity_score": avg_score("continuity_score"),
            "continuity_notes": longest("continuity_notes"),
            "notes": longest("notes"),
        }
