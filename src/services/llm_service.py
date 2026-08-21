"""
LLM（大语言模型）服务层

职责：
- 初始化 LLM 客户端
- 将镜头描述与剧本情节点做语义匹配
- 统一处理 JSON 提取
"""
import json
import os
from typing import List, Dict, Any

from src.models import Shot, ScriptBeat
from src.utils import logger


class LLMService:
    """大语言模型服务"""

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

        env_map = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "doubao": "ARK_API_KEY",
            "volcengine": "ARK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "custom": "OPENAI_API_KEY",
        }
        env_name = env_map.get(self.provider, "OPENAI_API_KEY")
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
                    f"或设置环境变量 {env_map.get(self.provider, 'OPENAI_API_KEY')}。"
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

    @staticmethod
    def _shot_config_text(shot: Shot) -> str:
        """从 Shot 和 shot_config 中提取用于 LLM 匹配的文本描述"""
        cfg = (shot.cv_metadata or {}).get("shot_config", {})
        lines = [
            f"镜头 {shot.shot_id}:",
            f"- 来源: {shot.source_file} {shot.tc_in}-{shot.tc_out}",
            f"- 内容摘要: {cfg.get('content_summary') or shot.action or '未知'}",
            f"- 场景: {cfg.get('location') or shot.location or '未知'}",
            f"- 时间: {cfg.get('time_of_day') or shot.time_of_day or '未知'}",
            f"- 角色: {', '.join(cfg.get('characters') or shot.characters) or '未知'}",
            f"- 景别: {cfg.get('shot_type') or shot.shot_size or '未知'}",
            f"- 机位: {cfg.get('camera_position') or shot.camera_position or '未知'}",
            f"- 运镜: {cfg.get('camera_movement') or shot.camera_movement or '未知'}",
            f"- 方向: {cfg.get('direction') or getattr(shot, 'direction', '') or '未知'}",
            f"- 动作: {cfg.get('action') or shot.action or '未知'}",
            f"- 情绪: {cfg.get('emotion') or shot.emotion or '未知'}",
            f"- 风格/氛围: {cfg.get('style') or ''} {cfg.get('atmosphere') or ''}".strip(),
            f"- 标签: {', '.join(cfg.get('tags') or shot.tags) or '无'}",
            f"- 关键物体: {', '.join(cfg.get('key_objects') or shot.key_objects) or '无'}",
            f"- 台词: {shot.asr_text or shot.dialogue or '无'}",
        ]
        return "\n".join(lines)

    def anchor_shots_to_script(self, shots: List[Shot], script_beats: List[ScriptBeat]) -> Dict[str, Dict[str, Any]]:
        """将镜头列表锚定到剧本情节点，返回 shot_id -> anchor 映射

        anchor 字段：beat, act, function, confidence, reasoning
        """
        beats_text = "\n".join([
            f"【{b.act} - {b.beat_id}】\n"
            f"地点: {b.location} | 时间: {b.time}\n"
            f"内容: {b.content}\n"
            f"情绪: {b.emotion}\n"
            f"关键动作: {', '.join(b.key_actions)}\n"
            f"关键台词: {b.key_dialogue}\n"
            for b in script_beats
        ])

        template = self._load_prompt_template(
            "phase2_match",
            "你是一位资深剪辑指导，擅长将片场素材映射到剧本结构。\n\n## 剧本大纲\n{beats_text}\n\n## 待匹配镜头\n{shots_text}\n\n"
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
        batch_size = 15  # 减小批次，降低模型混淆概率
        for batch_start in range(0, len(shots), batch_size):
            batch = shots[batch_start:batch_start + batch_size]
            shots_text = "\n\n".join([self._shot_config_text(s) for s in batch])

            # 用 replace 而非 format，避免模板中 JSON 花括号被误解析为占位符
            prompt = template.replace("{beats_text}", beats_text).replace("{shots_text}", shots_text)

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

        return anchor_map
