"""
Phase 3 VLM 终审器

使用本地 Qwen2.5-VL 模型对 preview.mp4 按 sub-beat 维度进行终审评分。
每个 sub-beat 抽取关键帧，调用本地 VLM 判断：
- 关键动作是否被覆盖
- 关键对白是否呈现
- 情绪是否匹配
- 与前后逻辑是否连贯
- 是否存在重复/错误内容

输出 ReviewReport，供 Phase 3 迭代优化使用。
"""
import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import torch
from PIL import Image

from src.local_models.vision_engine import VisionEngine
from src.utils import logger


@dataclass
class SubBeatReview:
    """单个 sub-beat 的 VLM 评分"""
    sub_beat_id: str
    parent_beat_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    key_observations: List[str] = field(default_factory=list)


@dataclass
class ReviewReport:
    """VLM 终审报告"""
    attempt: int
    total_score: float = 0.0
    pass_threshold: float = 0.75
    passed: bool = False
    dimensions: Dict[str, float] = field(default_factory=dict)
    sub_beat_reviews: List[SubBeatReview] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    raw_outputs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLMReviewer:
    """基于本地 Qwen2.5-VL 的 Phase 3 终审器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        local_cfg = config.get("models", {}).get("local", {})
        vision_cfg = local_cfg.get("vision", {})

        self.enabled = bool(local_cfg.get("enabled", False))
        self.device = local_cfg.get("device", "cuda")
        self.cache_dir = vision_cfg.get("cache_dir") or local_cfg.get("cache_dir")
        self.model_id = vision_cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.model_path = vision_cfg.get("model_path")
        self.load_in_4bit = vision_cfg.get("load_in_4bit", True)
        self.max_new_tokens = vision_cfg.get("max_new_tokens", 512)
        self.keyframe_count = config.get("phase3", {}).get("review_keyframe_count", 3)
        self.pass_threshold = float(config.get("phase3", {}).get("pass_threshold", 0.75))

        self._engine: Optional[VisionEngine] = None

    def _get_engine(self) -> Optional[VisionEngine]:
        if self._engine is None and self.enabled:
            self._engine = VisionEngine(
                model_id=self.model_id,
                model_path=self.model_path,
                device=self.device,
                load_in_4bit=self.load_in_4bit,
                max_new_tokens=self.max_new_tokens,
                cache_dir=self.cache_dir,
            )
        return self._engine

    def review(
        self,
        preview_path: str,
        sub_beats: List[Any],
        attempt: int = 1,
    ) -> ReviewReport:
        """对 preview.mp4 执行 VLM 终审"""
        report = ReviewReport(attempt=attempt, pass_threshold=self.pass_threshold)

        if not os.path.exists(preview_path):
            logger.error(f"[VLMReviewer] preview 不存在: {preview_path}")
            report.issues.append({
                "type": "system_error",
                "description": f"preview 不存在: {preview_path}",
            })
            return report

        engine = self._get_engine()
        if engine is None or not self.enabled:
            logger.warning("[VLMReviewer] 本地 VLM 未启用，返回空报告")
            report.issues.append({
                "type": "system_error",
                "description": "本地 VLM 未启用",
            })
            return report

        engine.load()
        if engine.model is None:
            logger.error("[VLMReviewer] 本地 VLM 模型加载失败")
            report.issues.append({
                "type": "system_error",
                "description": "本地 VLM 模型加载失败",
            })
            return report

        duration = self._get_video_duration(preview_path)
        if duration <= 0:
            logger.error(f"[VLMReviewer] 无法读取 preview 时长: {preview_path}")
            report.issues.append({
                "type": "system_error",
                "description": "无法读取 preview 时长",
            })
            return report

        intervals = self._compute_sub_beat_intervals(sub_beats, duration)

        reviews: List[SubBeatReview] = []
        raw_outputs: List[Dict[str, Any]] = []

        # 整体终审：每个 sub-beat 只抽 1 帧，一次性送入 VLM
        # 避免每 sub-beat 单独调用导致 5 轮迭代时间过长
        sb_frames: List[Tuple[Any, Image.Image]] = []
        for sb, (start_sec, end_sec) in zip(sub_beats, intervals):
            frames = self._extract_frames(preview_path, start_sec, end_sec, 1)
            if frames:
                sb_frames.append((sb, frames[0][1]))

        if sb_frames:
            reviews, raw_outputs = self._review_all_subbeats(engine, sb_frames)
        else:
            reviews = []
            raw_outputs = []
            for sb in sub_beats:
                reviews.append(SubBeatReview(
                    sub_beat_id=sb.sub_beat_id,
                    parent_beat_id=sb.parent_beat_id,
                    scores={},
                    reasoning="未能抽取到关键帧",
                ))

        report.sub_beat_reviews = reviews
        report.raw_outputs = raw_outputs
        report.dimensions = self._aggregate_dimensions(reviews)
        report.total_score = self._compute_total_score(report.dimensions)
        report.passed = report.total_score >= self.pass_threshold
        report.issues = self._detect_issues(sub_beats, reviews, intervals)

        logger.info(
            f"[VLMReviewer] attempt {attempt} 终审完成: "
            f"总分={report.total_score:.2f}, 及格={report.passed}"
        )
        return report

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()
        return duration

    @staticmethod
    def _compute_sub_beat_intervals(
        sub_beats: List[Any], total_duration: float
    ) -> List[Tuple[float, float]]:
        """计算每个 sub-beat 在 preview 中的实际时间区间"""
        durations = [float(getattr(sb, "estimated_duration", 0.0) or 0.0) for sb in sub_beats]
        total_est = sum(durations)

        intervals = []
        if total_est > 0 and total_duration > 0:
            cursor = 0.0
            for dur in durations:
                seg_dur = total_duration * (dur / total_est)
                intervals.append((cursor, min(cursor + seg_dur, total_duration)))
                cursor += seg_dur
        else:
            seg_dur = total_duration / len(sub_beats) if sub_beats else 0.0
            for i in range(len(sub_beats)):
                intervals.append((i * seg_dur, min((i + 1) * seg_dur, total_duration)))

        if intervals:
            intervals[-1] = (intervals[-1][0], total_duration)
        return intervals

    @staticmethod
    def _extract_frames(
        video_path: str,
        start_sec: float,
        end_sec: float,
        count: int = 3,
    ) -> List[Tuple[float, Image.Image]]:
        """从指定时间段抽取关键帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        total_frames_in_seg = end_frame - start_frame
        if count == 1:
            # 只取中间帧，代表性最好
            indices = [start_frame + total_frames_in_seg // 2]
        elif count >= 3:
            indices = [
                start_frame,
                start_frame + total_frames_in_seg // 2,
                end_frame - 1,
            ]
        else:
            step = max(1, total_frames_in_seg // count)
            indices = [min(start_frame + i * step, end_frame - 1) for i in range(count)]

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

    def _review_sub_beat(
        self,
        engine: VisionEngine,
        sub_beat: Any,
        frames: List[Tuple[float, Image.Image]],
    ) -> Tuple[SubBeatReview, Dict[str, Any]]:
        """对单个 sub-beat 的关键帧进行 VLM 评分"""
        prompt = self._build_review_prompt(sub_beat)
        images = [img for _, img in frames]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        response = ""
        try:
            text = engine.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = engine.processor(
                text=[text],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(engine.model.device)

            with torch.no_grad():
                generated_ids = engine.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = engine.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            scores, reasoning = self._parse_review_response(response)
        except Exception as e:
            logger.error(f"[VLMReviewer] sub-beat {sub_beat.sub_beat_id} 评分失败: {e}")
            scores = {}
            reasoning = f"模型推理失败: {e}"

        review = SubBeatReview(
            sub_beat_id=sub_beat.sub_beat_id,
            parent_beat_id=sub_beat.parent_beat_id,
            scores=scores,
            reasoning=reasoning,
            key_observations=[],
        )
        raw = {
            "sub_beat_id": sub_beat.sub_beat_id,
            "prompt": prompt,
            "response": response,
            "scores": scores,
            "reasoning": reasoning,
        }
        return review, raw

    def _review_all_subbeats(
        self,
        engine: VisionEngine,
        sb_frames: List[Tuple[Any, Image.Image]],
    ) -> Tuple[List[SubBeatReview], List[Dict[str, Any]]]:
        """分批对所有 sub-beat 进行 VLM 终审

        每批 3 个 sub-beat，每 sub-beat 1 帧，并降低图片分辨率，避免 8GB 显存 OOM。
        """
        if not sb_frames:
            return [], []

        batch_size = 3
        all_reviews: Dict[str, SubBeatReview] = {}
        all_raw_outputs: List[Dict[str, Any]] = []

        # 降低分辨率，减少 VLM 视觉 token 和显存占用
        max_size = 512

        for batch_start in range(0, len(sb_frames), batch_size):
            batch = sb_frames[batch_start:batch_start + batch_size]
            resized = []
            for sb, img in batch:
                w, h = img.size
                if max(w, h) > max_size:
                    ratio = max_size / max(w, h)
                    new_size = (int(w * ratio), int(h * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                resized.append((sb, img))

            prompt = self._build_overall_review_prompt(resized)
            images = [img for _, img in resized]
            sub_beats = [sb for sb, _ in resized]

            content: List[Dict[str, Any]] = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            response = ""
            sb_scores: Dict[str, Dict[str, Any]] = {}
            try:
                text = engine.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = engine.processor(
                    text=[text],
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = inputs.to(engine.model.device)

                with torch.no_grad():
                    generated_ids = engine.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = engine.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                sb_scores = self._parse_overall_review_response(response, sub_beats)
            except Exception as e:
                logger.error(f"[VLMReviewer] 整体终审批次 {batch_start // batch_size + 1} 失败: {e}")
                response = f"模型推理失败: {e}"

            for sb in sub_beats:
                scores = sb_scores.get(sb.sub_beat_id, {})
                reasoning = scores.pop("reasoning", response[:200]) if isinstance(scores, dict) else response[:200]
                all_reviews[sb.sub_beat_id] = SubBeatReview(
                    sub_beat_id=sb.sub_beat_id,
                    parent_beat_id=sb.parent_beat_id,
                    scores=scores if scores else {},
                    reasoning=str(reasoning) if reasoning else "整体终审未返回有效评分",
                )
                all_raw_outputs.append({
                    "sub_beat_id": sb.sub_beat_id,
                    "prompt": prompt,
                    "response": response,
                    "scores": scores,
                })

        reviews = [all_reviews.get(sb.sub_beat_id, SubBeatReview(
            sub_beat_id=sb.sub_beat_id,
            parent_beat_id=sb.parent_beat_id,
            scores={},
            reasoning="该 sub-beat 不在终审批次中",
        )) for sb, _ in sb_frames]

        return reviews, all_raw_outputs

    @staticmethod
    def _build_overall_review_prompt(sb_frames: List[Tuple[Any, Image.Image]]) -> str:
        """构建整体 VLM 终审 prompt"""
        lines: List[str] = []
        lines.append("你是一位专业影视剪辑终审员。下面是一组视频截图，每张截图按顺序对应一个剧情单元。")
        lines.append("请逐一判断每个剧情单元是否符合要求，并只输出一个 JSON 对象，不要输出任何其他文字。\n")

        for i, (sb, _) in enumerate(sb_frames, 1):
            actions = "、".join(sb.key_actions) if sb.key_actions else "无"
            dialogue = sb.key_dialogue or "无"
            emotion = sb.emotion or "无"
            pace = sb.pace or "正常"
            gender = sb.gender_state or "无"
            transition = sb.gender_transition or "无"
            lines.append(
                f"[{i}] {sb.sub_beat_id}: {sb.content}\n"
                f"    关键动作: {actions}\n"
                f"    关键对白: {dialogue}\n"
                f"    情绪: {emotion}  节奏: {pace}  性别状态: {gender}  状态转换: {transition}"
            )

        score_fields = (
            '"action_coverage": 0.7,     // 0.0-1.0，关键动作是否被画面覆盖\n'
            '        "dialogue_coverage": 0.8,    // 0.0-1.0，关键对白是否被画面/字幕呈现\n'
            '        "emotion_match": 0.6,        // 0.0-1.0，画面情绪是否符合要求\n'
            '        "logic_consistency": 0.9,    // 0.0-1.0，画面内容是否与剧情逻辑一致\n'
            '        "continuity_with_prev": 0.5, // 0.0-1.0，与上一个剧情单元衔接是否自然（首个填0.5）\n'
            '        "continuity_with_next": 0.5, // 0.0-1.0，与下一个剧情单元衔接是否自然（末个填0.5）\n'
            '        "redundancy_risk": 0.2,      // 0.0-1.0，是否与前后镜头明显重复（0=无重复，1=严重重复）\n'
            '        "reasoning": "简要说明评分理由"'
        )

        lines.append(
            "\n评分规则：\n"
            "1. 每个分数必须是 0.0 到 1.0 之间的具体数字，不要写范围或区间。\n"
            "2. 分数越高表示越符合要求。\n"
            "3. 只输出一个 JSON 对象，不要输出任何其他文字。\n\n"
            "请输出如下 JSON：\n"
            "{\n"
            '  "sub_beats": [\n'
            "    {\n"
            f'      "sub_beat_id": "{sb_frames[0][0].sub_beat_id}",\n'
            f"      {score_fields}\n"
            "    },\n"
            "    ... // 每个剧情单元一个对象，按顺序\n"
            "  ]\n"
            "}\n"
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_overall_review_response(
        response: str,
        sub_beats: List[Any],
    ) -> Dict[str, Dict[str, Any]]:
        """解析整体 VLM 返回的 JSON"""
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result: Dict[str, Dict[str, Any]] = {}
        items: List[Dict[str, Any]] = []
        try:
            data = json.loads(text.strip())
            items = data.get("sub_beats", [])
        except Exception:
            items = []

        valid_keys = {
            "action_coverage", "dialogue_coverage", "emotion_match",
            "logic_consistency", "continuity_with_prev", "continuity_with_next",
            "redundancy_risk", "reasoning",
        }

        def _extract_scores(item: Dict[str, Any]) -> Dict[str, Any]:
            scores: Dict[str, Any] = {}
            for key in valid_keys:
                val = item.get(key, 0.0)
                if key == "reasoning":
                    scores[key] = str(val)
                else:
                    # 处理 VLM 误把范围字符串当值的情况
                    if isinstance(val, str):
                        val = val.strip()
                        # 去掉 "0.0-1.0" 这种范围，fallback 为 0.0
                        try:
                            scores[key] = float(val)
                        except (ValueError, TypeError):
                            scores[key] = 0.0
                    else:
                        try:
                            scores[key] = float(val)
                        except (ValueError, TypeError):
                            scores[key] = 0.0
            return scores

        # 1. 优先用 JSON 解析出的 items
        for sb, item in zip(sub_beats, items):
            result[sb.sub_beat_id] = _extract_scores(item)

        # 2. fallback：对没有解析到的 sub_beat，从文本中按 sub_beat_id 分块匹配
        if not items:
            for sb in sub_beats:
                sb_id = re.escape(sb.sub_beat_id)
                # 找到该 sub_beat 在 response 中的大致块
                pattern = rf'"sub_beat_id"\s*[:：]\s*"{sb_id}"(.*?)"reasoning"\s*[:：]\s*"([^"]*)"'
                m = re.search(pattern, response, re.DOTALL)
                if m:
                    block = m.group(1) + '"reasoning": "' + m.group(2) + '"'
                    scores: Dict[str, Any] = {}
                    for key in valid_keys:
                        if key == "reasoning":
                            rm = re.search(rf'"reasoning"\s*[:：]\s*"([^"]*)"', block)
                            scores[key] = rm.group(1) if rm else ""
                        else:
                            rm = re.search(rf'"{key}"\s*[:：]\s*([0-9.]+)', block)
                            scores[key] = float(rm.group(1)) if rm else 0.0
                    result[sb.sub_beat_id] = scores
                else:
                    result[sb.sub_beat_id] = {}

        # 3. 如果 items 有解析但部分 sub_beat 缺少，补空
        for sb in sub_beats:
            if sb.sub_beat_id not in result:
                result[sb.sub_beat_id] = {}

        return result

    @staticmethod
    def _build_review_prompt(sub_beat: Any) -> str:
        """构建单个 sub-beat 的 VLM 评分 prompt（保留兼容）"""
        actions = "、".join(sub_beat.key_actions) if sub_beat.key_actions else "无"
        dialogue = sub_beat.key_dialogue or "无"
        emotion = sub_beat.emotion or "无"
        pace = sub_beat.pace or "正常"
        gender = sub_beat.gender_state or "无"
        transition = sub_beat.gender_transition or "无"

        return (
            "你是一位专业影视剪辑终审员。请仔细观察以上视频截图，"
            "判断这段画面是否符合下面这个剧情单元的要求。"
            "请只输出一个 JSON 对象，不要输出任何其他文字。\n\n"
            f"剧情单元: {sub_beat.sub_beat_id}\n"
            f"内容: {sub_beat.content}\n"
            f"关键动作: {actions}\n"
            f"关键对白: {dialogue}\n"
            f"情绪: {emotion}\n"
            f"节奏: {pace}\n"
            f"性别状态: {gender}\n"
            f"状态转换: {transition}\n\n"
            "请评分（0.0-1.0，1.0 为完全匹配）:\n"
            "{\n"
            '  "action_coverage": 0.0-1.0,  // 关键动作是否被画面覆盖\n'
            '  "dialogue_coverage": 0.0-1.0, // 关键对白是否被画面/字幕呈现\n'
            '  "emotion_match": 0.0-1.0,     // 画面情绪是否符合要求\n'
            '  "logic_consistency": 0.0-1.0, // 画面内容是否与剧情逻辑一致（如追猫段落不应出现焊接）\n'
            '  "continuity_with_prev": 0.0-1.0, // 与上一个剧情单元衔接是否自然（无上下文时填0.5）\n'
            '  "continuity_with_next": 0.0-1.0, // 与下一个剧情单元衔接是否自然（无上下文时填0.5）\n'
            '  "redundancy_risk": 0.0-1.0,   // 画面是否与前后镜头明显重复（0=无重复，1=严重重复）\n'
            '  "reasoning": "简要说明评分理由"\n'
            "}"
        )

    @staticmethod
    def _parse_review_response(response: str) -> Tuple[Dict[str, float], str]:
        """解析 VLM 评分响应"""
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            data = json.loads(text.strip())
        except Exception:
            data = {}
            patterns = {
                "action_coverage": r"action_coverage[:：]\s*([0-9.]+)",
                "dialogue_coverage": r"dialogue_coverage[:：]\s*([0-9.]+)",
                "emotion_match": r"emotion_match[:：]\s*([0-9.]+)",
                "logic_consistency": r"logic_consistency[:：]\s*([0-9.]+)",
                "continuity_with_prev": r"continuity_with_prev[:：]\s*([0-9.]+)",
                "continuity_with_next": r"continuity_with_next[:：]\s*([0-9.]+)",
                "redundancy_risk": r"redundancy_risk[:：]\s*([0-9.]+)",
            }
            for key, pat in patterns.items():
                m = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
                if m:
                    try:
                        data[key] = float(m.group(1))
                    except ValueError:
                        pass

        scores = {}
        for key in [
            "action_coverage",
            "dialogue_coverage",
            "emotion_match",
            "logic_consistency",
            "continuity_with_prev",
            "continuity_with_next",
            "redundancy_risk",
        ]:
            val = data.get(key, 0.0)
            try:
                scores[key] = float(val)
            except (ValueError, TypeError):
                scores[key] = 0.0

        reasoning = str(data.get("reasoning", ""))
        return scores, reasoning

    @staticmethod
    def _aggregate_dimensions(reviews: List[SubBeatReview]) -> Dict[str, float]:
        """从 sub-beat 评分汇总整体维度"""
        if not reviews:
            return {}

        completeness_scores = []
        coherence_scores = []
        missing_scores = []
        redundancy_scores = []
        logic_error_scores = []

        for r in reviews:
            s = r.scores
            if not s:
                continue
            action = s.get("action_coverage", 0.0)
            dialogue = s.get("dialogue_coverage", 0.0)
            logic = s.get("logic_consistency", 0.0)
            redundancy = s.get("redundancy_risk", 0.0)
            prev = s.get("continuity_with_prev", 0.5)
            next_ = s.get("continuity_with_next", 0.5)

            completeness_scores.append((action + dialogue) / 2.0)
            logic_error_scores.append(logic)
            redundancy_scores.append(1.0 - redundancy)
            coherence_scores.append((prev + next_) / 2.0)
            missing_scores.append(min(action, dialogue))

        def avg(scores: List[float]) -> float:
            return round(sum(scores) / len(scores), 2) if scores else 0.0

        return {
            "plot_completeness": avg(completeness_scores),
            "logic_coherence": avg(coherence_scores),
            "content_missing": avg(missing_scores),
            "redundancy": avg(redundancy_scores),
            "logic_error": avg(logic_error_scores),
            "duration_compliance": 1.0,  # 由外部填入
        }

    @staticmethod
    def _compute_total_score(dimensions: Dict[str, float]) -> float:
        """加权总分"""
        weights = {
            "plot_completeness": 0.25,
            "logic_coherence": 0.20,
            "content_missing": 0.20,
            "redundancy": 0.15,
            "logic_error": 0.10,
            "duration_compliance": 0.10,
        }
        total = 0.0
        weight_sum = 0.0
        for key, weight in weights.items():
            total += dimensions.get(key, 0.0) * weight
            weight_sum += weight
        return round(total / weight_sum, 2) if weight_sum > 0 else 0.0

    @staticmethod
    def _detect_issues(
        sub_beats: List[Any],
        reviews: List[SubBeatReview],
        intervals: List[Tuple[float, float]],
    ) -> List[Dict[str, Any]]:
        """根据评分结果生成问题清单"""
        issues: List[Dict[str, Any]] = []

        for sb, review, (start, end) in zip(sub_beats, reviews, intervals):
            s = review.scores
            if not s:
                issues.append({
                    "type": "system_error",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": "该 sub-beat 未能完成 VLM 评分",
                    "affected_shots": [],
                })
                continue

            action = s.get("action_coverage", 0.0)
            dialogue = s.get("dialogue_coverage", 0.0)
            logic = s.get("logic_consistency", 0.0)
            redundancy = s.get("redundancy_risk", 0.0)
            prev = s.get("continuity_with_prev", 0.5)
            next_ = s.get("continuity_with_next", 0.5)

            if action < 0.5:
                issues.append({
                    "type": "content_missing",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": f"关键动作覆盖不足: {sb.key_actions}",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

            if sb.key_dialogue and dialogue < 0.5:
                issues.append({
                    "type": "content_missing",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": f"关键对白覆盖不足: {sb.key_dialogue[:40]}",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

            if logic < 0.5:
                issues.append({
                    "type": "logic_error",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": "画面内容与剧情逻辑不一致",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

            if redundancy > 0.5:
                issues.append({
                    "type": "redundancy",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": "画面与前后镜头明显重复",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

            if prev < 0.4:
                issues.append({
                    "type": "logic_coherence",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": "与上一个剧情单元衔接不自然",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

            if next_ < 0.4:
                issues.append({
                    "type": "logic_coherence",
                    "sub_beat_id": sb.sub_beat_id,
                    "description": "与下一个剧情单元衔接不自然",
                    "reasoning": review.reasoning,
                    "affected_shots": [],
                })

        return issues
