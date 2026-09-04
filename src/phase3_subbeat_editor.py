"""
Phase 3: 基于 Sub-beat 的迭代式剪辑决策器

核心流程：
1. 把 Phase 2 的 beat 拆分为 sub-beat
2. 每个 sub-beat 独立选择最合适的镜头（规则评分 + 可选本地 LLM）
3. 应用变速控制时长
4. 渲染轻量 preview
5. 本地 VLM 终审
6. 最多 5 轮迭代，未及格取最高分
7. 输出最佳 timeline
"""
import os
import json
import re
import shutil
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from src.utils import (
    Shot, save_json, load_json, logger, parse_duration_string,
    tc_to_sec, sec_to_tc, ensure_dir,
)
from src.phase3_subbeat import SubBeat, load_sub_beats
from src.phase3_vlm_reviewer import VLMReviewer, ReviewReport
from src.phase3_preview_renderer import PreviewRenderer


class SubBeatPhase3Editor:
    """基于 sub-beat 的 Phase 3 剪辑决策器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project = config.get("project", {})
        self.processing = config.get("processing", {})
        self.output_dir = config.get("paths", {}).get("output", "./output")
        self.temp_dir = config.get("paths", {}).get("temp", "./temp")

        self.target_duration = parse_duration_string(self.project.get("target_duration", 0))
        self.target_min = self.target_duration * 0.95 if self.target_duration else 0.0
        self.target_max = self.target_duration * 1.05 if self.target_duration else float("inf")

        self.sub_beats = load_sub_beats(self.output_dir)
        if not self.sub_beats:
            raise RuntimeError("Phase 3 无法加载 sub-beat，请确认 Phase 2 已正确运行")

        self.preview_renderer = PreviewRenderer(config)
        self.vlm_reviewer = VLMReviewer(config)
        self.enable_vlm_review = config.get("phase3", {}).get("enable_vlm_review", True)
        self.max_attempts = int(config.get("phase3", {}).get("max_attempts", 5))

        self.attempts_dir = os.path.join(self.output_dir, "phase3_attempts")
        ensure_dir(self.attempts_dir)

        # 策略状态，每轮根据终审结果调整
        self.strategy = {
            "max_shots_per_subbeat": 2,
            "source_penalty_weight": 1.0,
            "redundancy_penalty_weight": 1.0,
            "allow_supplement": False,  # 默认严格按 Phase 2 锚定，禁止跨 beat 补充
            "speed_range": (0.75, 1.5),
            "single_shot_max_duration": 8.0,
        }
        self.missing_subbeats: List[str] = []  # 记录缺失 sub-beat

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, shots: List[Shot]) -> List[Dict[str, Any]]:
        """执行基于 sub-beat 的迭代式剪辑决策"""
        logger.info("=" * 60)
        logger.info("Phase 3: 基于 Sub-beat 的迭代式剪辑决策")
        logger.info("=" * 60)
        logger.info(f"sub-beat 数量: {len(self.sub_beats)}")
        logger.info(f"镜头池数量: {len(shots)}")
        logger.info(f"目标时长: {self.target_duration:.1f}s")

        shot_map = {s.shot_id: s for s in shots}
        best_attempt: Optional[Tuple[List[Dict[str, Any]], Optional[ReviewReport]]] = None

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Phase 3 第 {attempt} 轮")
            logger.info(f"{'='*60}")

            # 1. 为每个 sub-beat 选择镜头
            decisions = self._select_all_subbeats(shots, shot_map)

            # 2. 变速和时长控制
            decisions = self._apply_speed_and_duration(decisions, shot_map)

            # 3. 排序并编号
            decisions = self._sort_and_number_decisions(decisions)

            # 4. 渲染 preview
            preview_path = self.preview_renderer.render(
                decisions,
                output_name=f"preview_attempt_{attempt}.mp4",
                include_audio=False,
            )

            # 5. VLM 终审
            report: Optional[ReviewReport] = None
            if preview_path and self.enable_vlm_review and self.vlm_reviewer.enabled:
                report = self.vlm_reviewer.review(preview_path, self.sub_beats, attempt=attempt)
                self._save_attempt(attempt, decisions, report, preview_path)

                logger.info(
                    f"[Phase3] 第 {attempt} 轮评分: "
                    f"{report.total_score:.2f}/1.00 (及格线 {report.pass_threshold}) "
                    f"通过={report.passed}"
                )

                if report.passed:
                    best_attempt = (decisions, report)
                    logger.info(f"[Phase3] 第 {attempt} 轮通过终审，提前结束")
                    break
            else:
                logger.warning("[Phase3] 未生成 preview 或 VLM 未启用，跳过终审")

            # 6. 记录最佳尝试
            current_score = report.total_score if report else 0.0
            if best_attempt is None or current_score > (best_attempt[1].total_score if best_attempt[1] else 0.0):
                best_attempt = (decisions, report)

            # 7. 调整策略（最后一轮不调整）
            if attempt < 5 and report:
                self._adjust_strategy(report, attempt)

        if best_attempt is None:
            raise RuntimeError("Phase 3 未生成任何有效 timeline")

        decisions, report = best_attempt
        self._save_final(decisions, report)

        logger.info(f"\n[Phase3] 最终选择: 第 {report.attempt if report else '?'} 轮")
        logger.info(f"[Phase3] 最终评分: {report.total_score if report else 0.0:.2f}/1.00")
        logger.info(f"[Phase3] 输出 timeline: {len(decisions)} 个镜头")

        # 输出缺失 sub-beat 报告
        if self.missing_subbeats:
            missing_path = os.path.join(self.output_dir, "phase3_missing_subbeats.json")
            save_json({
                "missing_count": len(self.missing_subbeats),
                "missing_subbeats": sorted(set(self.missing_subbeats)),
                "note": "以下 sub-beat 在当前镜头池中找不到同 beat 候选镜头，请补充素材后重跑",
            }, missing_path)
            logger.warning(f"[Phase3] 有 {len(self.missing_subbeats)} 个 sub-beat 缺失候选镜头")
            logger.warning(f"[Phase3] 缺失报告: {missing_path}")

        return decisions

    # ------------------------------------------------------------------
    # Sub-beat 镜头选择
    # ------------------------------------------------------------------
    def _select_all_subbeats(
        self,
        shots: List[Shot],
        shot_map: Dict[str, Shot],
    ) -> List[Dict[str, Any]]:
        """为每个 sub-beat 选择镜头

        每个 shot_id 在整个 timeline 中只能使用一次，避免同一个镜头片段
        在不同 sub-beat 中重复出现。跨 parent_beat 同 source_file 也尽量避免。
        """
        decisions: List[Dict[str, Any]] = []
        used_shot_ids: Set[str] = set()  # 全局已使用 shot_id
        used_shot_ids_by_beat: Dict[str, Set[str]] = defaultdict(set)
        used_sources_by_beat: Dict[str, Set[str]] = defaultdict(set)
        shot_usage_seconds: Dict[str, float] = defaultdict(float)

        for sb in self.sub_beats:
            sb_decisions = self._select_for_subbeat(
                sb,
                shots,
                shot_map,
                used_shot_ids,
                used_shot_ids_by_beat,
                used_sources_by_beat,
                shot_usage_seconds,
            )
            for d in sb_decisions:
                decisions.append(d)
                used_shot_ids.add(d["shot_id"])
                used_shot_ids_by_beat[d["beat_id"]].add(d["shot_id"])
                used_sources_by_beat[d["beat_id"]].add(d["source_file"])
                # 累加该 shot 被使用的时间（变速前原始时长）
                raw_dur = self._decision_raw_duration(d, shot_map)
                shot_usage_seconds[d["shot_id"]] += raw_dur

        return decisions

    def _select_for_subbeat(
        self,
        sub_beat: SubBeat,
        shots: List[Shot],
        shot_map: Dict[str, Shot],
        used_shot_ids: Set[str],
        used_shot_ids_by_beat: Dict[str, Set[str]],
        used_sources_by_beat: Dict[str, Set[str]],
        shot_usage_seconds: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """为单个 sub-beat 选择 1-2 个镜头

        候选分两层：
        1. 严格匹配 parent_beat 的镜头（主要来源）
        2. 如果严格匹配不足，从全局跨 beat 补充语义高度匹配的镜头

        同一 shot_id 在整个 timeline 中只能出现一次。
        """
        current_beat = sub_beat.parent_beat_id

        # 已使用的 shot 全局排除
        used_shot_ids_other_beats: Set[str] = set()
        for bid, sids in used_shot_ids_by_beat.items():
            if bid != current_beat:
                used_shot_ids_other_beats.update(sids)

        primary_candidates = []
        for s in shots:
            if s.shot_id in used_shot_ids or s.shot_id in used_shot_ids_other_beats:
                continue
            anchor = s.script_anchor or {}
            beat = anchor.get("beat", "")
            if beat == current_beat:
                primary_candidates.append(s)

        # primary candidates 不强制动作关键词过滤，避免 Phase 2 文本描述不准导致误杀
        # 是否匹配交给评分函数里的 semantic_fit + 跨 beat 惩罚来自然排序
        candidates = list(primary_candidates)

        # 跨 beat 补充默认关闭，仅当策略明确允许且本 beat 内无足够候选时才启用
        if len(candidates) < 2 and self.strategy.get("allow_supplement", False):
            logger.warning(
                f"[Phase3] {sub_beat.sub_beat_id} 本 beat 候选不足，"
                f"尝试跨 beat 语义补充 (allow_supplement=True)"
            )
            for s in shots:
                if s.shot_id in used_shot_ids or s.shot_id in used_shot_ids_other_beats:
                    continue
                if s in primary_candidates:
                    continue
                anchor = s.script_anchor or {}
                confidence = float(anchor.get("confidence", 0.0) or 0.0)
                semantic_fit = self._semantic_fit_for_subbeat(s, sub_beat)

                # 跨 beat 补充必须包含关键动作关键词，避免 totally 不相关的镜头混入
                if not self._action_keywords_match(s, sub_beat.key_actions):
                    continue

                # 状态转换 sub-beat 必须包含换装相关关键词
                if sub_beat.gender_transition and not self._transition_keywords_match(s, sub_beat.gender_transition):
                    continue

                if confidence >= 0.70 and semantic_fit >= 0.35:
                    candidates.append(s)
                elif semantic_fit >= 0.45:
                    candidates.append(s)

        if not candidates:
            msg = (
                f"[Phase3] {sub_beat.sub_beat_id} 无候选镜头 "
                f"(beat={current_beat}, actions={sub_beat.key_actions})"
            )
            logger.warning(msg)
            self.missing_subbeats.append(sub_beat.sub_beat_id)
            return []

        # 按综合评分排序
        scored = []
        for shot in candidates:
            score = self._score_shot_for_subbeat(
                shot, sub_beat, used_sources_by_beat
            )
            scored.append((score, shot))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 选择主镜头
        selected: List[Shot] = []
        used_sources_in_subbeat: Set[str] = set()
        max_shots = self.strategy["max_shots_per_subbeat"]

        for score, shot in scored:
            if len(selected) >= max_shots:
                break
            if shot.source_file in used_sources_in_subbeat:
                # 同一 sub-beat 内尽量避免同素材
                if self.strategy["redundancy_penalty_weight"] > 0.5:
                    continue
            selected.append(shot)
            used_sources_in_subbeat.add(shot.source_file)

            # 如果已经选了一个，且该镜头已经能很好覆盖 sub-beat，则不再堆叠第二个
            if len(selected) == 1 and max_shots > 1:
                first_semantic = self._semantic_fit_for_subbeat(selected[0], sub_beat)
                first_dur = selected[0].duration_sec or 0.0
                need_second = False
                if sub_beat.estimated_duration > 0 and first_dur < sub_beat.estimated_duration * 0.7:
                    need_second = True
                if first_semantic < 0.5:
                    need_second = True
                if not need_second:
                    break

        if not selected and scored:
            # 兜底：选最高分一个
            selected = [scored[0][1]]

        if not selected:
            logger.warning(f"[Phase3] {sub_beat.sub_beat_id} 最终无可用镜头")
            return []

        decisions = []
        for i, shot in enumerate(selected):
            decisions.append(self._shot_to_decision(shot, sub_beat, i, shot_usage_seconds))

        return decisions

    def _score_shot_for_subbeat(
        self,
        shot: Shot,
        sub_beat: SubBeat,
        used_sources_by_beat: Dict[str, Set[str]],
    ) -> float:
        """单个镜头对 sub-beat 的适配评分"""
        score = 0.0
        anchor = shot.script_anchor or {}
        current_beat = sub_beat.parent_beat_id

        # 1. Phase 2 锚定置信度
        score += (anchor.get("confidence") or 0.0) * 1.5

        # 2. 内容与 sub-beat 的语义匹配
        semantic_fit = self._semantic_fit_for_subbeat(shot, sub_beat)
        score += semantic_fit * 1.2

        # 3. 质量分
        score += getattr(shot, "quality_score", 0.0) * 0.25

        # 4. 状态优先级
        status = getattr(shot, "status", "备选")
        status_bonus = {"核心": 0.8, "保留": 0.5, "备选": 0.2, "强制保留": 1.0, "待复核": 1.0}
        score += status_bonus.get(status, 0.0)

        # 5. 时长匹配
        if sub_beat.estimated_duration > 0 and shot.duration_sec > 0:
            fit = 1.0 - min(abs(shot.duration_sec - sub_beat.estimated_duration) / sub_beat.estimated_duration, 1.0)
            score += fit * 0.4

        # 6. source_file 复用惩罚：仅惩罚跨 parent_beat 复用
        # 同 parent_beat 内的复用是允许的（长镜头切分给多个 sub-beat）
        for beat_id, sources in used_sources_by_beat.items():
            if beat_id != current_beat and shot.source_file in sources:
                score -= 2.0 * self.strategy["source_penalty_weight"]
                break

        # 7. 跨 beat 使用惩罚（防止 A1 选到 B 的抱猫镜头）
        anchor_beat = (shot.script_anchor or {}).get("beat", "")
        if anchor_beat and anchor_beat != current_beat:
            score -= 3.0
            # 如果语义匹配度还不够高，再扣
            if semantic_fit < 0.6:
                score -= 2.0

        # 8. 超长惩罚
        if shot.duration_sec > self.strategy["single_shot_max_duration"]:
            score -= (shot.duration_sec - self.strategy["single_shot_max_duration"]) * 0.2

        return score

    def _semantic_fit_for_subbeat(self, shot: Shot, sub_beat: SubBeat) -> float:
        """计算镜头内容与 sub-beat 的语义匹配度"""
        shot_text = " ".join(filter(None, [
            shot.action or "",
            getattr(shot, "action_details", "") or "",
            shot.asr_text or "",
            shot.dialogue or "",
            ", ".join(shot.key_objects or []),
        ])).lower()

        if not shot_text:
            return 0.0

        def _fit(text: str) -> float:
            text_tokens = set(self._tokens(text))
            shot_tokens = set(self._tokens(shot_text))
            if not text_tokens or not shot_tokens:
                return 0.0
            overlap = text_tokens & shot_tokens
            recall = len(overlap) / len(text_tokens)
            precision = len(overlap) / len(shot_tokens)
            if recall + precision <= 0:
                return 0.0
            return 2 * recall * precision / (recall + precision)

        action_fit = _fit(" ".join(sub_beat.key_actions))
        dialogue_fit = _fit(sub_beat.key_dialogue)
        content_fit = _fit(sub_beat.content)

        return action_fit * 0.55 + dialogue_fit * 0.25 + content_fit * 0.20

    @staticmethod
    def _transition_keywords_match(shot: Shot, transition: str) -> bool:
        """检查镜头是否包含状态转换相关关键词"""
        shot_text = " ".join(filter(None, [
            shot.action or "",
            getattr(shot, "action_details", "") or "",
            shot.asr_text or "",
            shot.dialogue or "",
        ])).lower()
        transition_keywords = ["女装", "男装", "换装", "变身", "变装", "穿上", "脱下", "紧身", "勒得", "衣服"]
        return any(kw in shot_text for kw in transition_keywords)

    @staticmethod
    def _action_keywords_match(shot: Shot, key_actions: List[str]) -> bool:
        """检查镜头描述是否包含关键动作关键词"""
        if not key_actions:
            return True
        shot_text = " ".join(filter(None, [
            shot.action or "",
            getattr(shot, "action_details", "") or "",
            ", ".join(shot.key_objects or []),
        ])).lower()

        for action in key_actions:
            action = action.lower()
            # 直接包含
            if action in shot_text:
                return True
            # 拆成关键词匹配（2字以上）
            for i in range(len(action) - 1):
                for j in range(i + 2, min(len(action) + 1, i + 5)):
                    kw = action[i:j]
                    if len(kw) >= 2 and kw in shot_text:
                        return True
        return False

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """中文分词（按标点）"""
        if not text:
            return []
        stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
            "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "把", "被", "让", "向",
            "过", "能", "个", "她", "他", "它", "这", "那", "为", "之", "与", "及", "等", "或",
        }
        delimiters = set("，、。！？；：""''（）(),.!?;:\"'() ")
        words: Set[str] = set()
        current = ""
        for ch in text:
            if ch in delimiters:
                if len(current) >= 2:
                    words.add(current)
                current = ""
            else:
                current += ch
        if len(current) >= 2:
            words.add(current)

        # 2-gram
        for i in range(len(text) - 1):
            bg = text[i:i + 2]
            if bg[0] not in delimiters and bg[1] not in delimiters:
                words.add(bg)

        return [w for w in words if w not in stopwords]

    def _shot_to_decision(
        self,
        shot: Shot,
        sub_beat: SubBeat,
        index: int,
        shot_usage_seconds: Dict[str, float],
        speed: str = "1x",
    ) -> Dict[str, Any]:
        """Shot 转 decision dict，同时限制单镜头最大使用时长

        同一个 shot 被多个 sub-beat 共享时，按已使用时长向后切分。
        """
        tc_in = shot.tc_in
        tc_out = shot.tc_out
        fps = getattr(shot, "fps", 30.0) or 30.0
        try:
            in_sec = tc_to_sec(tc_in, fps)
            out_sec = tc_to_sec(tc_out, fps)
            duration = out_sec - in_sec
        except Exception:
            duration = getattr(shot, "duration_sec", 0.0) or 0.0
            in_sec = 0.0
            out_sec = duration

        # 已使用时长
        used_sec = shot_usage_seconds.get(shot.shot_id, 0.0)
        available_start = in_sec + used_sec

        # 本次可分配的最大时长
        max_dur = self.strategy["single_shot_max_duration"]
        desired_dur = min(
            sub_beat.estimated_duration if sub_beat.estimated_duration > 0 else duration,
            max_dur,
            out_sec - available_start,
        )

        if desired_dur <= 0:
            # shot 已被前面 sub-beat 用完，fallback 到原始起止
            available_start = in_sec
            desired_dur = min(duration, max_dur, sub_beat.estimated_duration if sub_beat.estimated_duration > 0 else duration)

        new_in_sec = available_start
        new_out_sec = available_start + desired_dur

        try:
            tc_in = sec_to_tc(new_in_sec, fps)
            tc_out = sec_to_tc(new_out_sec, fps)
        except Exception:
            tc_in = shot.tc_in
            tc_out = shot.tc_out

        duration = new_out_sec - new_in_sec
        notes = f"sub-beat {sub_beat.sub_beat_id}"
        if duration < (shot.duration_sec or 0.0) - 0.1:
            notes += f" [从 {new_in_sec:.2f}s 切分使用 {duration:.2f}s]"

        # 注意：shot_usage_seconds 的更新由 _select_all_subbeats 统一负责，
        # 这里只读取用于切分时间段。

        return {
            "shot_id": shot.shot_id,
            "source_file": shot.source_file,
            "source_path": shot.source_path,
            "tc_in": tc_in,
            "tc_out": tc_out,
            "duration_sec": duration,
            "raw_duration": duration,
            "speed": speed,
            "technique": "连续剪辑",
            "transition": "硬切",
            "audio": "保留原声",
            "purpose": f"覆盖 {sub_beat.sub_beat_id}: {', '.join(sub_beat.key_actions)}",
            "notes": notes,
            "beat_id": sub_beat.parent_beat_id,
            "sub_beat_id": sub_beat.sub_beat_id,
            "act": sub_beat.act,
            "scene": sub_beat.scene,
        }

    # ------------------------------------------------------------------
    # 变速和时长控制
    # ------------------------------------------------------------------
    def _apply_speed_and_duration(
        self,
        decisions: List[Dict[str, Any]],
        shot_map: Dict[str, Shot],
    ) -> List[Dict[str, Any]]:
        """为每个 sub-beat 的组合应用变速，使其接近目标时长"""
        # 按 sub_beat 分组
        grouped = defaultdict(list)
        for d in decisions:
            grouped[d.get("sub_beat_id", "")].append(d)

        min_speed, max_speed = self.strategy["speed_range"]

        for sb_id, group in grouped.items():
            sub_beat = next((sb for sb in self.sub_beats if sb.sub_beat_id == sb_id), None)
            if not sub_beat:
                continue

            target = sub_beat.estimated_duration
            raw_dur = sum(
                shot_map[d["shot_id"]].duration_sec
                for d in group
                if d["shot_id"] in shot_map
            )

            if raw_dur <= 0 or target <= 0:
                continue

            desired_mult = raw_dur / target
            mult = max(min_speed, min(max_speed, desired_mult))
            speed_str = f"{mult:.2f}x"

            for d in group:
                d["speed"] = speed_str
                d["notes"] = (d.get("notes", "") + f" [段落变速: {speed_str}]").strip()

        return decisions

    # ------------------------------------------------------------------
    # 排序
    # ------------------------------------------------------------------
    def _sort_and_number_decisions(
        self,
        decisions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """按 sub-beat 顺序对 decisions 排序并编号"""
        order_map = {sb.sub_beat_id: i for i, sb in enumerate(self.sub_beats)}

        def sort_key(d):
            sb_id = d.get("sub_beat_id", "")
            seq_in_sub = d.get("sequence_in_sub", 0)
            return (order_map.get(sb_id, 9999), seq_in_sub)

        decisions.sort(key=sort_key)

        # 同一 sub_beat 内如果有多个镜头，按 source_in 排序
        grouped = defaultdict(list)
        for d in decisions:
            grouped[d.get("sub_beat_id", "")].append(d)

        final = []
        for sb_id in [sb.sub_beat_id for sb in self.sub_beats]:
            group = grouped.get(sb_id, [])
            group.sort(key=lambda d: d.get("tc_in", "00:00:00:00"))
            for i, d in enumerate(group):
                d["sequence_in_sub"] = i
                final.append(d)

        # 全局编号
        for i, d in enumerate(final):
            d["sequence"] = i + 1

        return final

    # ------------------------------------------------------------------
    # 策略调整
    # ------------------------------------------------------------------
    def _adjust_strategy(self, report: ReviewReport, attempt: int):
        """根据终审报告调整下一轮策略"""
        if not report or not report.issues:
            return

        issue_types = [i.get("type", "") for i in report.issues]
        dim = report.dimensions

        logger.info(f"[Phase3] 第 {attempt} 轮策略调整")

        # 冗余/重复问题
        if "redundancy" in issue_types or dim.get("redundancy", 1.0) < 0.6:
            self.strategy["redundancy_penalty_weight"] = min(2.0, self.strategy["redundancy_penalty_weight"] + 0.5)
            self.strategy["source_penalty_weight"] = min(2.0, self.strategy["source_penalty_weight"] + 0.3)
            logger.info("  -> 提升 source_file 复用/冗余惩罚")

        # 内容缺失
        if "content_missing" in issue_types or dim.get("content_missing", 1.0) < 0.6:
            self.strategy["max_shots_per_subbeat"] = min(3, self.strategy["max_shots_per_subbeat"] + 1)
            logger.info("  -> 允许每个 sub-beat 使用更多镜头")

        # 逻辑错误
        if "logic_error" in issue_types or dim.get("logic_error", 1.0) < 0.6:
            self.strategy["allow_supplement"] = False
            logger.info("  -> 禁止备选补充，严格使用锚定镜头")

        # 逻辑连贯性
        if "logic_coherence" in issue_types or dim.get("logic_coherence", 1.0) < 0.6:
            self.strategy["source_penalty_weight"] = min(2.0, self.strategy["source_penalty_weight"] + 0.3)
            logger.info("  -> 提升跨 sub-beat source 惩罚")

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def _save_attempt(
        self,
        attempt: int,
        decisions: List[Dict[str, Any]],
        report: ReviewReport,
        preview_path: str,
    ):
        """保存每次尝试的结果"""
        attempt_dir = os.path.join(self.attempts_dir, f"attempt_{attempt}")
        ensure_dir(attempt_dir)

        # 复制 preview
        if preview_path and os.path.exists(preview_path):
            dst = os.path.join(attempt_dir, "preview.mp4")
            shutil.copy2(preview_path, dst)

        # 保存 timeline
        timeline_data = {
            "project": self.project,
            "target_duration": self.target_duration,
            "attempt": attempt,
            "total_decisions": len(decisions),
            "timeline": decisions,
        }
        save_json(timeline_data, os.path.join(attempt_dir, "timeline.json"))

        # 保存评分报告
        save_json(report.to_dict(), os.path.join(attempt_dir, "review_report.json"))

    def _save_final(
        self,
        decisions: List[Dict[str, Any]],
        report: Optional[ReviewReport],
    ):
        """保存最终结果到 output/timeline.json 和 output/phase3_edit_decision.json"""
        timeline_data = {
            "project": self.project,
            "target_duration": self.target_duration,
            "attempt": report.attempt if report else 0,
            "total_score": report.total_score if report else 0.0,
            "passed": report.passed if report else False,
            "total_decisions": len(decisions),
            "timeline": decisions,
        }

        # 保存到 output/timeline.json（Phase 4 读取）
        save_json(timeline_data, os.path.join(self.output_dir, "timeline.json"))

        # 兼容旧格式：保存 phase3_edit_decision.json
        compatible = {
            "project": self.project,
            "target_duration": self.target_duration,
            "target_duration_min": self.target_min,
            "target_duration_max": self.target_max,
            "total_projected_duration": self._projected_duration(decisions),
            "total_decisions": len(decisions),
            "timeline": decisions,
        }
        save_json(compatible, os.path.join(self.output_dir, "phase3_edit_decision.json"))

    @staticmethod
    def _decision_raw_duration(d: Dict[str, Any], shot_map: Dict[str, Shot]) -> float:
        """从 decision 的 tc_in/tc_out 计算原始时长（秒）"""
        shot = shot_map.get(d.get("shot_id", ""))
        if not shot:
            return float(d.get("duration_sec", d.get("raw_duration", 0.0)) or 0.0)
        fps = getattr(shot, "fps", 30.0) or 30.0
        try:
            in_sec = tc_to_sec(d.get("tc_in", shot.tc_in), fps)
            out_sec = tc_to_sec(d.get("tc_out", shot.tc_out), fps)
            return max(0.0, out_sec - in_sec)
        except Exception:
            return float(d.get("duration_sec", d.get("raw_duration", 0.0)) or 0.0)

    @staticmethod
    def _projected_duration(decisions: List[Dict[str, Any]]) -> float:
        """计算变速后的投影时长"""
        total = 0.0
        for d in decisions:
            speed = str(d.get("speed", "1x"))
            mult = SubBeatPhase3Editor._parse_speed_multiplier(speed) or 1.0
            dur = float(d.get("duration_sec", d.get("raw_duration", 0.0)) or 0.0)
            total += dur / mult
        return total

    @staticmethod
    def _parse_speed_multiplier(speed: str) -> Optional[float]:
        s = str(speed).strip().lower()
        if not s or s == "删除":
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return 1.0
        if s.endswith("x"):
            try:
                return float(s[:-1])
            except ValueError:
                return 1.0
        try:
            return float(s)
        except ValueError:
            return 1.0
