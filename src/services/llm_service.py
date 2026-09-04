"""
LLM（大语言模型）服务层

职责：
- 初始化 LLM 客户端
- 将镜头描述与剧本情节点做语义匹配
- 统一处理 JSON 提取
"""
import json
import os
from typing import List, Dict, Any, Optional, Set

from src.models import Shot, ScriptBeat
from src.utils import logger


class LLMService:
    """大语言模型服务"""

    ENV_MAP = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "doubao": "ARK_API_KEY",
        "volcengine": "ARK_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "custom": "OPENAI_API_KEY",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("models", {}).get("llm", {})
        self.provider = self.config.get("provider", "deepseek")
        self.model_name = self.config.get("model", "deepseek-chat")
        self.max_tokens = self.config.get("max_tokens", 8192)
        self.temperature = self.config.get("temperature", 0.3)
        self.client = self._init_client()

    def _resolve_api_key(self) -> str:
        """按优先级解析 API Key：配置 > 环境变量"""
        api_key = self.config.get("api_key", "")
        if api_key:
            return api_key

        env_name = self.ENV_MAP.get(self.provider, "OPENAI_API_KEY")
        return os.environ.get(env_name, "")

    def _init_client(self):
        """初始化 LLM 客户端。支持 OpenAI、DeepSeek、豆包/火山方舟等 OpenAI 兼容接口。"""
        openai_compatible_providers = {"openai", "deepseek", "doubao", "qwen", "volcengine", "custom"}
        if self.provider in openai_compatible_providers:
            import openai
            base_url = self.config.get("base_url")
            if self.provider == "deepseek" and not base_url:
                base_url = "https://api.deepseek.com"

            api_key = self._resolve_api_key()
            if not api_key:
                raise RuntimeError(
                    f"LLM provider '{self.provider}' 缺少 API Key。请在 config.yaml 中设置 models.llm.api_key "
                    f"或设置环境变量 {self.ENV_MAP.get(self.provider, 'OPENAI_API_KEY')}。"
                )

            return openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        raise NotImplementedError(f"LLM provider {self.provider} 尚未实现")

    def _call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def generate(self, prompt: str) -> str:
        """对外提供的通用文本生成接口"""
        return self._call(prompt)

    def _extract_json(self, content: str) -> Any:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())

    def _load_prompt_template(self, template_name: str, fallback: str) -> str:
        """加载提示词模板，文件不存在时返回 fallback"""
        import os
        template_path = os.path.join("prompts", f"{template_name}.txt")
        try:
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"提示词模板加载失败 {template_path}: {e}")
        return fallback

    def analyze_script_beats(
        self,
        script_beats: List[ScriptBeat],
        total_duration: float,
    ) -> Dict[str, Dict[str, Any]]:
        """用 LLM 分析剧本节奏，给每个情节点分配目标时长、节奏、情绪强度和重要性。

        返回: beat_id -> dict(estimated_duration, pace, emotion_intensity, priority, required_shots_count)
        """
        if not script_beats or total_duration <= 0:
            return {}

        beats_text = "\n\n".join([
            f"【{b.beat_id}】{b.act} / {b.scene}\n"
            f"地点: {b.location} | 时间: {b.time}\n"
            f"内容: {b.content}\n"
            f"情绪: {b.emotion}\n"
            f"关键动作: {', '.join(b.key_actions)}\n"
            f"关键台词: {b.key_dialogue}"
            for b in script_beats
        ])

        # 构造 beat_id 列表，便于在 prompt 中明确示例
        beat_ids_text = ", ".join([b.beat_id for b in script_beats])

        prompt = (
            "你是一位资深短片剪辑指导。请根据以下剧本大纲和总目标时长，"
            "为每个情节点（beat）分配合理的成片时长、节奏、情绪强度和剧情优先级。\n\n"
            f"总目标时长: {total_duration:.1f} 秒\n"
            f"情节点数量: {len(script_beats)}\n\n"
            "## 剧本大纲\n"
            f"{beats_text}\n\n"
            "## 任务\n"
            "为每个 beat 输出 JSON 对象，要求：\n"
            "1. 键名必须严格使用上述剧本大纲中的 beat_id（例如：'场1-情节点A'），禁止使用 A/B/C 等缩写。\n"
            "2. estimated_duration: 该 beat 在成片中的建议时长（秒），所有 beat 之和必须接近总目标时长。\n"
            "3. pace: 节奏标签，只能选 '爆发' / '快' / '正常' / '慢' / '静止' 之一。\n"
            "4. emotion_intensity: 情绪强度，0.0-5.0。\n"
            "5. priority: 剧情重要性，1-5，5 为最高。关键转折/高潮给 5，过渡给 2-3。\n"
            "6. required_shots_count: 完成该 beat 叙事所需的最少镜头数，建议 1-3。\n"
            "7. key_actions: 字符串数组，该 beat 必须用镜头覆盖的核心动作（如 ['奔跑追猫','黑猫上檐']）。\n"
            "8. gender_state: 该 beat 中主要角色的性别/状态，如 '男装' / '女装' / ''。\n"
            "9. gender_transition: 该 beat 是否发生状态切换，如 '男装→女装' / '女装→男装' / ''。\n"
            "10. reasoning: 为什么这样分配。\n\n"
            f"可用的 beat_id 列表（必须严格使用这些键名）：{beat_ids_text}\n\n"
            "输出格式（严格 JSON）：\n"
            "{\n"
            f'  "{script_beats[0].beat_id}": {{"estimated_duration": 8.5, "pace": "快", "emotion_intensity": 3.5, "priority": 4, "required_shots_count": 2, "reasoning": "..."}},\n'
            "  ...\n"
            "}\n\n"
            "只输出 JSON，不要其他内容。"
        )

        try:
            content = self._call(prompt)
            raw = self._extract_json(content)
            if not isinstance(raw, dict):
                logger.warning(f"LLM 剧本节奏分析返回不是 dict，尝试包装: {type(raw)}")
                if isinstance(raw, list):
                    raw = {item.get("beat_id"): item for item in raw if isinstance(item, dict) and "beat_id" in item}
        except Exception as e:
            logger.error(f"LLM 剧本节奏分析失败: {e}")
            return {}

        # 构建完整 beat_id 查找表，用于兼容 LLM 可能的缩写键
        full_beat_ids = {b.beat_id: b.beat_id for b in script_beats}
        alias_map = {}
        for b in script_beats:
            # 兼容 "A" -> "场1-情节点A"
            if "情节点" in b.beat_id:
                alias = b.beat_id.split("情节点")[-1].strip()
                alias_map[alias] = b.beat_id

        result = {}
        for raw_key, data in raw.items():
            if not isinstance(data, dict):
                continue
            # 把可能的缩写映射回完整 beat_id
            beat_id = full_beat_ids.get(raw_key) or alias_map.get(raw_key) or raw_key
            try:
                key_actions = data.get("key_actions") or []
                if isinstance(key_actions, str):
                    key_actions = [a.strip() for a in key_actions.split(",") if a.strip()]
                result[str(beat_id)] = {
                    "estimated_duration": float(data.get("estimated_duration", 0.0) or 0.0),
                    "pace": str(data.get("pace", "正常")),
                    "emotion_intensity": float(data.get("emotion_intensity", 0.0) or 0.0),
                    "priority": int(data.get("priority", 3) or 3),
                    "required_shots_count": int(data.get("required_shots_count", 1) or 1),
                    "key_actions": key_actions,
                    "gender_state": str(data.get("gender_state", "")),
                    "gender_transition": str(data.get("gender_transition", "")),
                    "reasoning": str(data.get("reasoning", "")),
                }
            except Exception as e:
                logger.warning(f"解析 beat {beat_id} 的节奏分析结果失败: {e}")
                continue

        # 后处理：优先保留剧本大纲中原有的关键动作，避免 LLM 改写导致 beat 语义混淆
        for b in script_beats:
            if b.beat_id in result and b.key_actions:
                original_actions = [a.strip() for a in b.key_actions if a.strip()]
                if original_actions:
                    result[b.beat_id]["key_actions"] = original_actions
                    logger.info(f"保留 beat {b.beat_id} 原始关键动作: {original_actions}")

        # 校验总时长
        total_est = sum(v["estimated_duration"] for v in result.values())
        if total_est > 0 and abs(total_est - total_duration) > total_duration * 0.1:
            logger.warning(
                f"LLM 分配的 beat 总时长 {total_est:.1f}s 与目标 {total_duration:.1f}s 偏差超过 10%，"
                f"将进行比例归一化"
            )
            ratio = total_duration / total_est
            for k in result:
                result[k]["estimated_duration"] *= ratio

        return result

    @staticmethod
    def _shot_config_text(shot: Shot) -> str:
        """从 Shot 和 shot_config 中提取用于 LLM 匹配的文本描述"""
        cfg = (shot.cv_metadata or {}).get("shot_config", {})

        def _list(val):
            if not val:
                return []
            if isinstance(val, list):
                return val
            return [str(val)]

        lines = [
            f"镜头 {shot.shot_id}:",
            f"- 来源: {shot.source_file} {shot.tc_in}-{shot.tc_out}",
            f"- 内容摘要: {cfg.get('content_summary') or shot.action or '未知'}",
            f"- 场景: {cfg.get('location') or shot.location or '未知'}",
            f"- 时间: {cfg.get('time_of_day') or shot.time_of_day or '未知'}",
            f"- 角色: {', '.join(_list(cfg.get('characters') or shot.characters)) or '未知'}",
            f"- 景别: {cfg.get('shot_type') or shot.shot_size or '未知'}",
            f"- 机位: {cfg.get('camera_position') or shot.camera_position or '未知'}",
            f"- 运镜: {cfg.get('camera_movement') or shot.camera_movement or '未知'}",
            f"- 方向: {cfg.get('direction') or getattr(shot, 'direction', '') or '未知'}",
            f"- 动作: {cfg.get('action') or shot.action or '未知'}",
            f"- 行为: {cfg.get('behavior') or shot.behavior or '未知'}",
            f"- 主体: {cfg.get('primary_subject') or shot.primary_subject or '未知'}",
            f"- 情绪: {cfg.get('emotion') or shot.emotion or '未知'}",
            f"- 情绪强度: {cfg.get('emotion_intensity') or shot.emotion_intensity or '未知'}",
            f"- 节奏: {cfg.get('pace') or shot.pace or '未知'}",
            f"- 风格/氛围: {(cfg.get('style') or '')} {(cfg.get('atmosphere') or '')}".strip() or "无",
            f"- 标签: {', '.join(_list(cfg.get('tags') or shot.tags)) or '无'}",
            f"- 关键物体: {', '.join(_list(cfg.get('key_objects') or shot.key_objects)) or '无'}",
            f"- 台词/ASR: {shot.asr_text or shot.dialogue or cfg.get('dialogue') or '无'}",
        ]
        return "\n".join(lines)

    def anchor_shots_to_script(
        self,
        shots: List[Shot],
        script_beats: List[ScriptBeat],
        beat_analysis: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """将镜头列表锚定到剧本情节点，返回 shot_id -> anchor 映射

        anchor 字段：beat, act, function, confidence, reasoning
        """
        beats_text = "\n".join([
            f"【{b.act} - {b.beat_id}】\n"
            f"剧情顺序: 第 {i + 1} / {len(script_beats)} 个情节点\n"
            f"地点: {b.location} | 时间: {b.time}\n"
            f"内容: {b.content}\n"
            f"情绪: {b.emotion}\n"
            f"关键动作: {', '.join(b.key_actions)}\n"
            f"关键台词: {b.key_dialogue}\n"
            f"性别状态: {b.gender_state or '无特殊要求'}\n"
            f"状态转换: {b.gender_transition or '无'}\n"
            for i, b in enumerate(script_beats)
        ])

        beat_analysis = beat_analysis or {}
        beat_analysis_text = "\n".join([
            f"【{b.beat_id}】 顺序: {i + 1}/{len(script_beats)} | "
            f"建议时长: {beat_analysis.get(b.beat_id, {}).get('estimated_duration', '未分配')}s | "
            f"节奏: {beat_analysis.get(b.beat_id, {}).get('pace', '未分配')} | "
            f"情绪强度: {beat_analysis.get(b.beat_id, {}).get('emotion_intensity', '未分配')} | "
            f"优先级: {beat_analysis.get(b.beat_id, {}).get('priority', '未分配')} | "
            f"最少镜头数: {beat_analysis.get(b.beat_id, {}).get('required_shots_count', '未分配')}"
            for i, b in enumerate(script_beats)
        ])

        template = self._load_prompt_template(
            "phase2_match",
            "你是一位资深剪辑指导，擅长将片场素材映射到剧本结构。\n\n## 剧本大纲\n{beats_text}\n\n## 剧情节奏分析\n{beat_analysis_text}\n\n## 待匹配镜头\n{shots_text}\n\n"
            "将每个镜头匹配到最合适的剧本情节点，输出 JSON 数组，字段：shot_id, beat, act, function, confidence, reasoning。无法匹配时 beat 为 UNMATCHED。只输出 JSON。",
        )

        def _normalize_anchors(raw: Any, batch: List[Shot]) -> List[Dict[str, Any]]:
            """把 LLM 各种奇形怪状的返回统一成 List[Dict]"""
            if raw is None:
                return []
            if isinstance(raw, dict):
                # 情况1: {"S001": {...}, "S002": {...}}
                if all(isinstance(v, dict) for v in raw.values()):
                    result = []
                    for k, v in raw.items():
                        v["shot_id"] = v.get("shot_id") or k
                        result.append(v)
                    return result
                # 情况2: {"results": [...]} / {"anchors": [...]} / {"data": [...]}
                for key in ("results", "anchors", "data", "shots"):
                    if key in raw and isinstance(raw[key], list):
                        return raw[key]
                # 情况3: 单条对象被包在 dict 里
                if "beat" in raw or "shot_id" in raw:
                    return [raw]
                return []
            if isinstance(raw, list):
                return raw
            return []

        anchor_map = {}
        shot_map = {s.shot_id: s for s in shots}

        # 来源文件名白名单：根据素材文件名前缀，限制可匹配的情节点范围
        # 收紧规则：C招聘 只匹配 C2（小六画外音/云琛反应），B紧身 只匹配 C1/C2
        source_beat_whitelist = {
            "A追逐": {"场1-情节点A1", "场1-情节点A2"},
            "B变身": {"场1-情节点A2", "场1-情节点B", "场1-情节点C2"},
            "B紧身": {"场1-情节点C1", "场1-情节点C2"},
            "C招聘": {"场1-情节点C2", "场1-情节点D", "场1-情节点E"},
            "D小六": {"场1-情节点D", "场1-情节点E", "场1-情节点F", "场1-情节点G"},
        }

        batch_size = 10  # 减小批次，降低模型混淆概率
        for batch_start in range(0, len(shots), batch_size):
            batch = shots[batch_start:batch_start + batch_size]
            shots_text = "\n\n".join([self._shot_config_text(s) for s in batch])

            # 用 replace 而非 format，避免模板中 JSON 花括号被误解析为占位符
            prompt = (
                template
                .replace("{beats_text}", beats_text)
                .replace("{beat_analysis_text}", beat_analysis_text)
                .replace("{shots_text}", shots_text)
            )

            try:
                content = self._call(prompt)
                raw = self._extract_json(content)
                anchors = _normalize_anchors(raw, batch)

                if not anchors:
                    logger.warning(f"LLM 返回的锚定结果无法解析为列表 (batch {batch_start})，原始内容前 500 字: {content[:500]}")

                for idx, a in enumerate(anchors):
                    if not isinstance(a, dict):
                        logger.warning(f"LLM 返回的第 {idx} 个锚定项不是字典，跳过: {a}")
                        continue
                    shot_id = a.get("shot_id") or a.get("id")
                    # 如果缺少 shot_id，按顺序用本 batch 的 shot_id 回填
                    if not shot_id and idx < len(batch):
                        shot_id = batch[idx].shot_id
                        logger.debug(f"LLM 返回缺少 shot_id，按 batch 顺序回填为 {shot_id}")
                    if not shot_id:
                        logger.warning(f"LLM 返回的锚定结果缺少 shot_id 且无法回填，跳过: {a}")
                        continue
                    # 统一字段名（兼容旧版 matched_beat / matched_act）
                    anchor = {
                        "beat": a.get("beat") or a.get("matched_beat", "UNMATCHED"),
                        "act": a.get("act") or a.get("matched_act", ""),
                        "function": a.get("function", ""),
                        "confidence": float(a.get("confidence", 0.0) or 0.0),
                        "reasoning": a.get("reasoning", ""),
                    }
                    anchor_map[shot_id] = anchor
            except Exception as e:
                logger.error(f"LLM 锚定失败 (batch {batch_start}): {e}")
                logger.error(f"原始返回内容前 1000 字: {content[:1000] if 'content' in locals() else 'N/A'}")
                for shot in batch:
                    anchor_map[shot.shot_id] = {
                        "beat": "ERROR",
                        "act": "",
                        "function": "",
                        "confidence": 0.0,
                        "reasoning": "",
                    }

        # 后处理：过滤低置信度、非法 beat 和来源文件名白名单
        valid_beats = {b.beat_id for b in script_beats}
        confidence_threshold = self.config.get("phase2", {}).get("anchor_confidence_threshold", 0.7)
        for shot_id, anchor in list(anchor_map.items()):
            beat = anchor.get("beat", "UNMATCHED")
            conf = anchor.get("confidence", 0.0)
            if beat not in valid_beats and beat not in {"UNMATCHED", "ERROR"}:
                logger.warning(f" shot {shot_id} 返回非法 beat '{beat}'，强制设为 UNMATCHED")
                anchor["beat"] = "UNMATCHED"
                anchor["reasoning"] = (anchor.get("reasoning", "") + " [后处理：beat_id 不在剧本列表中]").strip()
            if beat in valid_beats and conf < confidence_threshold:
                logger.warning(f" shot {shot_id} 对 {beat} 的置信度 {conf:.2f} 低于阈值 {confidence_threshold}，设为 UNMATCHED")
                anchor["beat"] = "UNMATCHED"
                anchor["reasoning"] = (anchor.get("reasoning", "") + f" [后处理：置信度 {conf:.2f} 过低]").strip()

            # 来源文件名白名单校验（仅对成功匹配的 beat 做兜底）
            shot = shot_map.get(shot_id)
            if shot and beat in valid_beats:
                source_file = shot.source_file or ""
                for prefix, allowed in source_beat_whitelist.items():
                    if prefix in source_file:
                        if beat not in allowed:
                            logger.warning(
                                f" shot {shot_id} 来源文件 '{source_file}' 与 beat '{beat}' 不匹配，"
                                f"根据白名单只允许 {allowed}，强制设为 UNMATCHED"
                            )
                            anchor["beat"] = "UNMATCHED"
                            anchor["reasoning"] = (
                                anchor.get("reasoning", "") + f" [后处理：来源文件 {source_file} 不在 {allowed} 白名单内]"
                            ).strip()
                        break

            anchor_map[shot_id] = anchor

        return anchor_map
