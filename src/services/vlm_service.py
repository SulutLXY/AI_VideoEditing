"""
VLM（视觉语言模型）服务层

隔离不同 provider 的调用细节，对外提供统一的时序分析接口。
"""
import base64
import json
import os
from typing import List, Dict, Tuple, Optional, Any

from src.models import Segment
from src.utils import logger


class VLMService:
    """视觉语言模型服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("models", {}).get("vlm", {})
        self.provider = self.config.get("provider", "openai")
        self.model_name = self.config.get("model", "gpt-4o")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.3)
        self.frame_sample_rate = self.config.get("frame_sample_rate", 1)
        self.max_frames = self.config.get("max_frames", 60)
        self.local_service = None
        self.client = None
        self._init_provider(config)

    def _init_provider(self, config: Dict[str, Any]):
        """初始化 VLM 客户端或本地模型服务。"""
        if self.provider == "local":
            from src.services.local_vlm_service import LocalVLMService
            self.local_service = LocalVLMService(config)
            return

        openai_compatible_providers = {"openai", "doubao", "qwen", "volcengine", "custom"}
        if self.provider in openai_compatible_providers:
            import openai
            base_url = self.config.get("base_url")
            self.client = openai.OpenAI(
                api_key=self.config.get("api_key"),
                base_url=base_url or None,
            )
            return

        raise NotImplementedError(f"VLM provider {self.provider} 尚未实现")

    def _call(self, messages: List[Dict]) -> str:
        """统一调用 VLM"""
        if self.provider == "doubao":
            content = self._call_doubao_responses(messages)
            logger.debug(f"[VLM] 原始响应:\n{content[:2000]}")
            return content

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        logger.debug(f"[VLM] 原始响应:\n{content[:2000]}")
        return content

    def _call_doubao_responses(self, messages: List[Dict]) -> str:
        """使用豆包/火山方舟 Responses API 格式调用 VLM"""
        # 将 OpenAI chat.completions 的 messages 格式转换为 Responses API 的 input 格式
        user_message = messages[0] if messages else {"role": "user", "content": []}
        content = []
        for item in user_message.get("content", []):
            if item.get("type") == "text":
                content.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                # Doubao Responses API 的 image_url 必须是字符串（URL 或 base64 data URI）
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    image_url = image_url.get("url", "")
                content.append({
                    "type": "input_image",
                    "image_url": image_url,
                })

        input_payload = [{"role": "user", "content": content}]

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=input_payload,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            # Responses API 返回 output_text；若为空则回退到 chat.completions
            output_text = getattr(response, "output_text", "") or ""
            if output_text.strip():
                return output_text
            logger.warning("豆包 Responses API 返回空 output_text，尝试 fallback 到 chat.completions")
        except Exception as e:
            logger.warning(f"豆包 Responses API 调用失败，尝试 fallback 到 chat.completions: {e}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    def _extract_json(self, content: str) -> Dict:
        """从 VLM 返回中提取 JSON"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())

    def build_messages(self, prompt: str, frames: List[Tuple[float, str]]) -> List[Dict]:
        """构建带图片的 messages"""
        images = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
            for _, b64 in frames
        ]
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *images],
            }
        ]

    def analyze_temporal_segments(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> List[Segment]:
        """对视频进行时序分析，返回语义片段列表"""
        if self.local_service is not None:
            return self.local_service.analyze_temporal_segments(video_path, frames, duration)

        prompt = self._build_temporal_prompt(frames, duration)
        messages = self.build_messages(prompt, frames)

        try:
            content = self._call(messages)
            data = self._extract_json(content)
            return self._parse_segments(data)
        except Exception as e:
            logger.error(f"VLM 时序分析失败: {e}")
            return [Segment(start=0.0, end=duration, description="VLM 分析失败，整体作为一个片段")]

    def analyze_whole_video(
        self,
        video_path: str,
        frames: List[Tuple[float, str]],
        duration: float,
    ) -> Dict[str, Any]:
        """分析已处理视频的整体内容，不切分"""
        if self.local_service is not None:
            return self.local_service.analyze_whole_video(video_path, frames, duration)

        prompt = self._build_whole_video_prompt(frames, duration)
        messages = self.build_messages(prompt, frames)

        try:
            content = self._call(messages)
            return self._extract_json(content)
        except Exception as e:
            logger.error(f"PROCESSED 素材分析失败: {e}")
            return {}

    def validate_cut_candidates(
        self,
        candidates: List[Tuple[float, List[Tuple[float, str]]]],
    ) -> Dict[float, bool]:
        """对 CV 检测出的候选切点做 VLM 二分类验证

        candidates: [(timestamp, [(frame_time, base64), ...]), ...]
        每个候选点附带前后各 2 帧，共 4 帧。

        返回: {timestamp: True/False}，True 表示应当切分。
        """
        if not candidates:
            return {}

        if self.local_service is not None:
            return self.local_service.validate_cut_candidates(candidates)

        prompt = self._build_cut_validation_prompt(candidates)

        # 把所有候选帧按顺序平铺
        all_frames = []
        for _, frames in candidates:
            all_frames.extend(frames)

        messages = self.build_messages(prompt, all_frames)

        try:
            content = self._call(messages)
            data = self._extract_json(content)
            results = {}
            for item in data.get("results", []):
                ts = float(item.get("timestamp", -1))
                if ts >= 0:
                    results[ts] = bool(item.get("is_cut", False))
            return results
        except Exception as e:
            logger.error(f"VLM 切点验证失败: {e}")
            return {}

    def _build_cut_validation_prompt(
        self,
        candidates: List[Tuple[float, List[Tuple[float, str]]]],
    ) -> str:
        sections = []
        frame_idx = 0
        for timestamp, frames in candidates:
            times = ", ".join([f"{t:.2f}s" for t, _ in frames])
            sections.append(
                f"候选切点 {timestamp:.2f}s：前后帧时间点 [{times}]，对应输入图片中的 "
                f"frame_{frame_idx} 到 frame_{frame_idx + len(frames) - 1}"
            )
            frame_idx += len(frames)

        sections_text = "\n".join(sections)

        return f"""你是一位专业的电影镜头分析师。下面提供多个候选切点，每个候选点包含前后各 2 帧（共 4 帧）。请判断每个候选点是「真实镜头切换」还是「镜头内的高速运动/特效/模糊/抖动」。

{sections_text}

## 判断标准
- 真实切镜：前后帧在机位、场景、主体、构图上发生明显跳跃，中间没有过渡。
- 非切镜：画面变化是由镜头内运动（人物移动、相机运动、爆炸、快速 pan 等）造成的连续变化。

## 输出格式
请输出纯 JSON：

{{
  "results": [
    {{"timestamp": 1.40, "is_cut": true, "reason": "机位从远景跳切到特写"}},
    {{"timestamp": 4.20, "is_cut": false, "reason": "画面内快速运动导致的连续变化"}}
  ]
}}

只输出 JSON，不要其他文字。"""

    def _build_temporal_prompt(self, frames: List[Tuple[float, str]], duration: float) -> str:
        frame_times = ", ".join([f"{t:.1f}s" for t, _ in frames])
        return f"""你是一位专业的电影镜头分析师。下面是一组按时间顺序排列的视频帧（时间点分别为：{frame_times}），总时长约为 {duration:.1f} 秒。

请仔细观察这些帧的时序变化，按以下内容输出分析结果。

## 输出格式
请输出纯 JSON，包含以下字段：

{{
  "segments": [
    {{
      "start": 0.0,
      "end": 12.5,
      "description": "该片段的整体内容描述",
      "location": "场景地点",
      "time_of_day": "白天/傍晚/夜晚/室内灯光",
      "characters": ["角色名"],
      "action": "主要动作",
      "emotion": "情绪",
      "dialogue": "关键台词（如有）",
      "camera_position": "机位描述，如柜台正面/门口右侧/特写机位",
      "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦",
      "shot_size": "特写/近景/中景/全景/大全景",
      "is_long_take": false,
      "notes": "其他值得注意的信息"
    }},
    ...
  ]
}}

## 分析要求
1. 如果视频是连续长镜头或一镜到底，请只输出一个 segment，并设置 is_long_take=true。
2. 如果视频内部存在明显的机位切换、场景转换、情绪转折或主体变化，请拆分为多个 segment。
3. 每个 segment 的时间范围必须连续且不重叠，覆盖整个视频。
4. 对机位和 camera_position 的描述要准确、稳定，便于后续判断镜头是否切换。
5. 只输出 JSON，不要包含其他文字。"""

    def _build_whole_video_prompt(self, frames: List[Tuple[float, str]], duration: float) -> str:
        frame_times = ", ".join([f"{t:.1f}s" for t, _ in frames])
        return f"""你是一位专业的电影镜头分析师。下面是一组按时间顺序排列的视频帧（时间点分别为：{frame_times}），来自一段视频素材，总时长约为 {duration:.1f} 秒。

请输出这段视频的整体分析结果，JSON 格式：

{{
  "location": "场景地点",
  "time_of_day": "白天/傍晚/夜晚/室内灯光",
  "characters": ["角色名"],
  "action": "主要动作/情节（一句话摘要）",
  "action_details": "动作细节：角色的具体肢体动作、手势、走位、互动对象等",
  "emotion": "整体情绪",
  "dialogue": "关键台词",
  "camera_position": "机位描述",
  "camera_movement": "固定/推/拉/摇/移/跟/手持/变焦",
  "shot_size": "特写/近景/中景/全景/大全景",
  "direction": "人物朝向或运动方向（如：从左向右、从右向左、面向镜头、背对镜头、静止、走向深处）",
  "framing": "构图描述（如：居中构图、三分法、前景遮挡、对称构图、过肩镜头）",
  "performance": "表演评估：自然度、情绪强度、是否入戏、有无明显表演痕迹",
  "continuity_score": 0.85,
  "continuity_notes": "镜头内部连续性说明：是否一镜到底、是否有跳切/穿帮/方向跳变等",
  "lighting": "光效",
  "color_tone": "色调",
  "style": "风格",
  "atmosphere": "氛围",
  "culture": "文化/时代背景",
  "key_objects": ["关键道具"],
  "tags": ["标签"],
  "internal_segments": [
    {{"start": 0.0, "end": 10.0, "description": "段落1"}}
  ],
  "notes": "其他说明"
}}

注意：internal_segments 只用于标记内部段落，不生成独立镜头。只输出 JSON。"""

    def _parse_segments(self, data: Dict) -> List[Segment]:
        segments = []
        for seg in data.get("segments", []):
            segments.append(Segment(
                start=float(seg.get("start", 0)),
                end=float(seg.get("end", 0)),
                description=seg.get("description", ""),
                location=seg.get("location", ""),
                time_of_day=seg.get("time_of_day", ""),
                characters=seg.get("characters", []),
                action=seg.get("action", ""),
                emotion=seg.get("emotion", ""),
                dialogue=seg.get("dialogue", ""),
                camera_position=seg.get("camera_position", ""),
                camera_movement=seg.get("camera_movement", ""),
                shot_size=seg.get("shot_size", ""),
                framing=seg.get("framing", ""),
                lighting=seg.get("lighting", ""),
                color_tone=seg.get("color_tone", ""),
                style=seg.get("style", ""),
                atmosphere=seg.get("atmosphere", ""),
                culture=seg.get("culture", ""),
                tags=seg.get("tags", []),
                key_objects=seg.get("key_objects", []),
                direction=seg.get("direction", ""),
                performance=seg.get("performance", ""),
                action_details=seg.get("action_details", ""),
                continuity_score=float(seg.get("continuity_score", 0.0)),
                continuity_notes=seg.get("continuity_notes", ""),
                coherence_score=float(seg.get("coherence_score", 0.0)),
                is_long_take=bool(seg.get("is_long_take", False)),
            ))
        return segments

    def sample_frames(self, video_path: str, duration: float, temp_dir: str) -> List[Tuple[float, str]]:
        """从视频中抽取帧用于 VLM 分析"""
        import os
        from src.utils import run_ffmpeg, ensure_dir

        if duration <= 0:
            return []

        total_frames = int(duration * self.frame_sample_rate)
        if total_frames <= self.max_frames:
            times = [i / self.frame_sample_rate for i in range(total_frames + 1)]
        else:
            step = duration / (self.max_frames - 1) if self.max_frames > 1 else duration
            times = [min(i * step, duration) for i in range(self.max_frames)]

        if 0.0 not in times:
            times.insert(0, 0.0)
        if duration not in times:
            times.append(duration)
        # 去重并严格限制在 [0, duration) 内，避免 seek 到文件末尾失败
        times = sorted(set(round(t, 3) for t in times if 0 <= t < duration))

        frames_dir = os.path.join(temp_dir, "frames", os.path.basename(video_path))
        ensure_dir(frames_dir)

        logger.info(f"[VLM] 计划抽帧: {len(times)} 帧, 时间点: {times}")

        frames = []
        for i, t in enumerate(times):
            output_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
            try:
                run_ffmpeg([
                    "-ss", str(t),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "3",
                    "-s", "640x360",
                    output_path,
                ])
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    frames.append((t, b64))
            except Exception as e:
                logger.warning(f"抽帧失败 t={t}: {e}")

        logger.info(f"[VLM] 实际抽帧成功: {len(frames)} / {len(times)}")
        return frames
