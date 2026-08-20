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
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
    def extract_keyframes(video_path: str, count: int = 3) -> List[Tuple[float, Image.Image]]:
        """提取关键帧 (timestamp_sec, PIL.Image)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0

        if count >= 3:
            indices = [0, total_frames // 2, total_frames - 1]
        else:
            step = max(1, total_frames // count)
            indices = [min(i * step, total_frames - 1) for i in range(count)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp = idx / fps
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append((round(timestamp, 2), img))
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
        try:
            out["continuity_score"] = float(out["continuity_score"] or 0.0)
        except Exception:
            out["continuity_score"] = 0.0
        return out

    def process_video(self, video_path: str, keyframe_count: int = 3) -> Dict:
        """处理单个视频片段，返回与 Shot 模型对齐的视觉分析结果"""
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

        keyframes = self.extract_keyframes(video_path, keyframe_count)
        if not keyframes:
            return fallback

        prompt_text = self._build_prompt()
        all_results = []
        key_frames_output = []

        for timestamp, image in keyframes:
            result = self._infer_single(image, prompt_text)
            all_results.append(result)
            key_frames_output.append({
                "timestamp": timestamp,
                "description": result.get("action", ""),
            })

        final = self._vote_aggregate(all_results)
        final["key_frames"] = key_frames_output
        return final

    def _infer_single(self, image: Image.Image, prompt_text: str) -> Dict:
        """单张图片推理"""
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
            "key_objects": union_list("key_objects"),
            "tags": union_list("tags"),
            "continuity_score": avg_score("continuity_score"),
            "continuity_notes": longest("continuity_notes"),
            "notes": longest("notes"),
        }
