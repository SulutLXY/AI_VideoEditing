"""
Phase 3: 剪辑语法决策（精剪方案生成）

为每个镜头决策:
- 保留 / 删除（独立 keep 字段）
- 播放速度（升格/降格/原速）
- 剪辑手法（连续/J-Cut/L-Cut/交叉/跳切/匹配剪辑）
- 转场类型
- 音频处理
- 是否从备选池补充镜头

后处理:
- 消费 Phase 2 的 axis_warnings / missing_beats
- 目标成片时长 ±5% 自动压缩 / 延展 / 补镜头
- PROCESSED 素材只保护删除，不限制速度与手法
"""
import os
import json
import re
from typing import List, Dict, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict

from src.utils import Shot, save_json, load_json, logger, parse_duration_string, tc_to_sec
from src.services.llm_service import LLMService


@dataclass
class EditDecision:
    """单个镜头的剪辑决策"""
    sequence: int
    shot_id: str
    source_file: str
    tc_in: str
    tc_out: str
    speed: str          # "1x" / "50%" / "200%"
    technique: str      # 剪辑手法
    transition: str     # 转场
    audio: str          # 音频处理
    purpose: str        # 叙事目的
    notes: str = ""     # 备注
    beat_id: str = ""   # 所属情节点
    act: str = ""       # 所属幕
    scene: str = ""     # 所属场

    def to_dict(self):
        return asdict(self)


class Phase3Editor:
    """阶段3剪辑决策器"""

    def __init__(self, config: Dict):
        self.config = config
        self.project = config['project']
        self.processing = config['processing']
        self.models = config['models']
        self.output_dir = config['paths']['output']

        self.target_duration = parse_duration_string(self.project.get('target_duration', 0))
        self.target_min = self.target_duration * 0.95 if self.target_duration else 0.0
        self.target_max = self.target_duration * 1.05 if self.target_duration else float('inf')

        self.prompt_template = self._load_prompt_template()
        self.llm_service = LLMService(config)
        self.beat_map = self._load_script_beats()

    def _load_prompt_template(self) -> str:
        """加载 Prompt 模板，失败时回退到内置最小模板"""
        default_path = os.path.join('prompts', 'phase3_edit.txt')
        path = self.project.get('phase3_prompt', default_path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"加载 Phase 3 prompt 模板失败 ({path}): {e}，使用内置模板")
            return self._fallback_prompt_template()

    @staticmethod
    def _fallback_prompt_template() -> str:
        return """你是一位资深电影剪辑师。\n## 项目信息\n- 片名: {project_name}\n- 风格: {project_style}\n- 类型: {project_genre}\n- 目标时长: {target_duration}s\n- 允许区间: {target_duration_min}s ~ {target_duration_max}s\n- 当前核心镜头总时长: {current_core_duration}s\n\n## 剪辑语法参考\n- 升格(慢动作, 40%-80%): 情绪高潮、关键动作细节。要求素材帧率≥60fps\n- 降格(快动作, 200%-600%): 压缩时间、过渡段落\n- 原速(1x): 正常叙事\n\n### 剪辑手法\n连续剪辑/J-Cut/L-Cut/交叉/跳切/匹配剪辑/反应镜头插入\n\n### 转场\n硬切/叠化/闪白/闪黑/黑场\n\n### 音频\n保留原声/J-Cut/L-Cut/配乐覆盖/音效强化\n\n## 前期预警\n{warnings_text}\n\n## 待决策核心镜头\n{shots_text}\n\n## 补充镜头池（时长不足时可选）\n{supplement_text}\n\n## 任务\n为每个镜头做出剪辑决策，输出 JSON 数组:\n[\n  {{\n    \"shot_id\": \"S001\",\n    \"keep\": true,\n    \"speed\": \"1x 或 50% 或 200%\",\n    \"speed_reason\": \"\",\n    \"technique\": \"\",\n    \"technique_reason\": \"\",\n    \"transition\": \"\",\n    \"audio\": \"\",\n    \"audio_reason\": \"\",\n    \"purpose\": \"\",\n    \"notes\": \"\"\n  }}\n]\n\n注意:\n1. keep=false 表示删除；keep=true 表示保留。\n2. 总时长必须落在允许区间内，超出时请优先变速，仍不足或超出请参考补充镜头池。\n3. 相邻镜头节奏要有变化。\n4. 特殊升格镜头不要连续使用 3 个以上。\n5. 只输出 JSON。"""

    # ------------------------------------------------------------------
    # 剧本节奏加载
    # ------------------------------------------------------------------
    def _load_script_beats(self) -> Dict[str, Dict]:
        """加载 Phase 2 的剧本节奏分析结果"""
        path = os.path.join(self.output_dir, 'script_beats_analysis.json')
        if not os.path.exists(path):
            return {}
        try:
            data = load_json(path)
            return {b.get('beat_id'): b for b in data.get('beats', []) if b.get('beat_id')}
        except Exception as e:
            logger.warning(f"加载剧本节奏分析失败: {e}")
            return {}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self, shots: List[Shot]) -> List[EditDecision]:
        """执行剪辑语法决策"""
        logger.info("=" * 60)
        logger.info("Phase 3: 剪辑语法决策")
        logger.info("=" * 60)

        valid_statuses = {"核心", "保留", "备选", "强制保留", "待复核"}
        kept_shots = [s for s in shots if getattr(s, 'status', '保留') in valid_statuses]
        logger.info(f"待决策镜头数: {len(kept_shots)}")

        # 核心镜头进入 LLM 决策；备选镜头仅作为补充池
        supplement_shots = [s for s in kept_shots if s.status == "备选"]
        core_shots = [s for s in kept_shots if s.status != "备选"]

        core_shots = self._sort_by_narrative(core_shots)
        supplement_shots = self._sort_by_narrative(supplement_shots)

        # 加载 Phase 2 预警
        warnings_text = self._load_phase2_warnings()

        # 分批 LLM 决策
        decisions: List[EditDecision] = []
        batch_size = 15
        for batch_start in range(0, len(core_shots), batch_size):
            batch = core_shots[batch_start:batch_start + batch_size]
            batch_decisions = self._llm_edit_decision(batch, batch_start, warnings_text, supplement_shots)
            decisions.extend(batch_decisions)

        # 后处理：强制保留/待复核不可删除
        decisions = self._post_process_protected_shots(decisions, core_shots)

        # 后处理：帧率限制（PROCESSED 素材不限制）
        decisions = self._post_process_speed(decisions, core_shots)

        # 后处理：目标时长预算控制
        decisions = self._adjust_duration(decisions, kept_shots, supplement_shots)

        # 重新排序并编号
        decisions = self._sort_decisions_by_narrative(decisions, core_shots + supplement_shots)
        for i, d in enumerate(decisions):
            d.sequence = i + 1

        # 保存结果
        result = {
            "project": self.project,
            "target_duration": self.target_duration,
            "target_duration_min": self.target_min,
            "target_duration_max": self.target_max,
            "total_projected_duration": self._projected_duration(decisions),
            "total_decisions": len(decisions),
            "timeline": [d.to_dict() for d in decisions]
        }
        save_json(result, os.path.join(self.output_dir, 'phase3_edit_decision.json'))

        # 导出 CSV
        self._export_csv(decisions)

        logger.info(
            f"剪辑方案生成完成: {len(decisions)} 个决策, "
            f"投影时长 {result['total_projected_duration']:.1f}s "
            f"(目标 {self.target_duration:.1f}s, 允许 {self.target_min:.1f}s~{self.target_max:.1f}s)"
        )
        return decisions

    # ------------------------------------------------------------------
    # 排序与描述构建
    # ------------------------------------------------------------------
    def _sort_by_narrative(self, shots: List[Shot]) -> List[Shot]:
        """按叙事逻辑排序镜头"""
        def sort_key(shot: Shot):
            anchor = shot.script_anchor or {}
            beat = anchor.get('beat', 'ZZZZ')
            act_num = 99
            scene_num = 99
            act_match = re.search(r'第(\d+)幕', anchor.get('act', ''))
            if act_match:
                act_num = int(act_match.group(1))
            scene_match = re.search(r'场(\d+)', beat)
            if scene_match:
                scene_num = int(scene_match.group(1))
            return (act_num, scene_num, beat, shot.tc_in)
        return sorted(shots, key=sort_key)

    def _sort_decisions_by_narrative(
        self,
        decisions: List[EditDecision],
        shots: List[Shot],
    ) -> List[EditDecision]:
        """按叙事顺序对决策排序：先按 beat/act/scene，再按原镜头顺序"""
        shot_map = {s.shot_id: s for s in shots}

        def sort_key(d: EditDecision) -> tuple:
            # 优先使用决策中已保存的 beat_id
            beat = d.beat_id or "ZZZZ"
            act = d.act or ""
            # 如果决策中没有，回退到 shot.script_anchor
            if beat == "ZZZZ":
                shot = shot_map.get(d.shot_id)
                anchor = shot.script_anchor if shot else None
                if anchor:
                    beat = anchor.get("beat", "ZZZZ")
                    act = anchor.get("act", "")
            act_num = 99
            scene_num = 99
            act_match = re.search(r"第(\d+)幕", act)
            if act_match:
                act_num = int(act_match.group(1))
            scene_match = re.search(r"场(\d+)", beat)
            if scene_match:
                scene_num = int(scene_match.group(1))
            # 同 beat 内按 shot_id 数字顺序
            try:
                shot_num = int(d.shot_id.replace("S", ""))
            except ValueError:
                shot_num = 9999
            return (act_num, scene_num, beat, shot_num)

        return sorted(decisions, key=sort_key)

    def _build_shot_text(self, shot: Shot) -> str:
        """构建单个镜头的文本描述"""
        vlm = shot.vlm_description
        return (
            f"【{shot.shot_id}】{shot.source_file} {shot.tc_in}-{shot.tc_out} ({shot.duration_sec:.1f}s)\n"
            f"  素材状态: {shot.state} | 选择状态: {shot.status} | 质量分: {shot.quality_score:.2f}\n"
            f"  场景: {vlm.get('location', '未知')} | 时间: {vlm.get('time_of_day', '未知')}\n"
            f"  角色: {', '.join(vlm.get('characters', []) or [])}\n"
            f"  景别: {vlm.get('shot_size', '未知')} | 机位: {vlm.get('camera_position', '未知')}\n"
            f"  方向: {vlm.get('direction', '未知')} | 运镜: {vlm.get('camera_movement', '固定')}\n"
            f"  动作: {vlm.get('action', '未知')} | 动作细节: {vlm.get('action_details', '无')}\n"
            f"  情绪: {vlm.get('emotion', '未知')} | 表演: {vlm.get('performance', '无')}\n"
            f"  连续性评分: {vlm.get('continuity_score', 0.0):.2f} "
            f"({vlm.get('continuity_notes', '无')})\n"
            f"  物理属性: {shot.fps:.1f}fps | {shot.resolution[0]}x{shot.resolution[1]} | {shot.aspect_ratio}\n"
            f"  台词: {shot.asr_text or '无'}\n"
            f"  剧本锚定: {shot.script_anchor.get('beat', '未匹配')} "
            f"({shot.script_anchor.get('function', '')}, 置信度{shot.script_anchor.get('confidence', 0):.2f})"
        )

    def _build_supplement_text(self, shots: List[Shot]) -> str:
        """构建补充镜头池文本"""
        if not shots:
            return "（无）"
        return "\n\n".join([self._build_shot_text(s) for s in shots])

    def _load_phase2_warnings(self) -> str:
        """读取 Phase 2 去重报告的预警信息"""
        report_path = os.path.join(self.output_dir, 'phase2_deduplication.json')
        if not os.path.exists(report_path):
            return "无前期预警"

        try:
            report = load_json(report_path)
        except Exception as e:
            logger.warning(f"读取 Phase 2 报告失败: {e}")
            return "无前期预警"

        lines = []
        missing = report.get('missing_beats', [])
        if missing:
            lines.append("缺失情节点（需补充素材或从备选池插入）:")
            for item in missing:
                lines.append(
                    f"  - {item.get('beat_id', '未知')}: {item.get('note', '')} "
                    f"(严重度: {item.get('severity', '高')})"
                )

        axis = report.get('axis_warnings', [])
        if axis:
            lines.append("越轴/方向跳变风险（剪辑时请用转场或刻意情绪越轴处理）:")
            for item in axis:
                chars = ', '.join(item.get('characters', []) or [])
                lines.append(
                    f"  - {item.get('beat_id', '未知')}: "
                    f"{item.get('shot_a', '')}({item.get('direction_a', '')}) vs "
                    f"{item.get('shot_b', '')}({item.get('direction_b', '')}) "
                    f"[共同角色: {chars}]"
                )

        return "\n".join(lines) if lines else "无前期预警"

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    def _llm_edit_decision(
        self,
        shots: List[Shot],
        start_seq: int,
        warnings_text: str,
        supplement_shots: List[Shot],
    ) -> List[EditDecision]:
        """调用 LLM 进行剪辑决策"""

        shots_text = "\n\n".join([self._build_shot_text(s) for s in shots])
        supplement_text = self._build_supplement_text(supplement_shots)
        current_core_duration = sum(s.duration_sec for s in shots)

        # 使用 replace 避免模板中 JSON 花括号被 format 误解析为占位符
        prompt = (
            self.prompt_template
            .replace("{project_name}", self.project.get('name', '未命名'))
            .replace("{project_style}", self.project.get('style', ''))
            .replace("{project_genre}", self.project.get('genre', '剧情短片'))
            .replace("{target_duration}", f"{self.target_duration:.1f}")
            .replace("{target_duration_min}", f"{self.target_min:.1f}")
            .replace("{target_duration_max}", f"{self.target_max:.1f}")
            .replace("{current_core_duration}", f"{current_core_duration:.1f}")
            .replace("{warnings_text}", warnings_text)
            .replace("{shots_text}", shots_text)
            .replace("{supplement_text}", supplement_text)
        )

        try:
            content = self.llm_service.generate(prompt)
            # 提取 JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            decisions_raw = json.loads(content.strip())
            if isinstance(decisions_raw, dict):
                decisions_raw = decisions_raw.get('decisions', [decisions_raw])

            return self._parse_llm_decisions(decisions_raw, shots, start_seq)

        except Exception as e:
            logger.error(f"LLM 剪辑决策失败: {e}")
            return self._default_decisions(shots, start_seq, notes="LLM决策失败，使用默认值")

    def _parse_llm_decisions(
        self,
        raw_decisions: List[Dict],
        shots: List[Shot],
        start_seq: int,
    ) -> List[EditDecision]:
        """解析 LLM 返回的 JSON 决策"""
        shot_map = {s.shot_id: s for s in shots}
        decisions = []

        for i, d in enumerate(raw_decisions):
            shot_id = d.get('shot_id')
            shot = shot_map.get(shot_id)
            if not shot:
                continue

            speed = str(d.get('speed', '1x')).strip()
            keep = d.get('keep', True)

            # speed 为 "删除" 时视为 keep=false
            if speed == '删除':
                keep = False

            notes = d.get('notes', '')
            if not keep:
                notes = (notes + " [LLM建议删除]").strip()

            decisions.append(self._shot_to_decision(
                shot=shot,
                speed=speed if keep else '删除',
                sequence=start_seq + i + 1,
                technique=d.get('technique', '连续剪辑'),
                transition=d.get('transition', '硬切'),
                audio=d.get('audio', '保留原声'),
                purpose=d.get('purpose', ''),
                notes=notes,
            ))

        return decisions

    def _default_decisions(
        self,
        shots: List[Shot],
        start_seq: int,
        notes: str = "默认保留",
    ) -> List[EditDecision]:
        """LLM 失败时的默认决策"""
        return [
            self._shot_to_decision(
                shot=s,
                speed="1x",
                sequence=start_seq + i + 1,
                technique="连续剪辑",
                transition="硬切",
                audio="保留原声",
                purpose="默认保留",
                notes=notes,
            )
            for i, s in enumerate(shots)
        ]

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------
    def _post_process_speed(self, decisions: List[EditDecision], shots: List[Shot]) -> List[EditDecision]:
        """检查升格帧率限制；PROCESSED 素材完全交给 LLM，不在这里限制"""
        min_fps = self.processing.get('slow_motion_min_fps', 60)
        shot_map = {s.shot_id: s for s in shots}

        for d in decisions:
            shot = shot_map.get(d.shot_id)
            if not shot:
                continue
            if shot.state == 'PROCESSED':
                continue

            mult = parse_speed_multiplier(d.speed)
            if mult is not None and mult < 1.0:
                # 升格
                if shot.fps < min_fps:
                    logger.warning(
                        f"{d.shot_id} 帧率{shot.fps}fps不足，取消升格 ({d.speed} → 1x)"
                    )
                    d.speed = "1x"
                    d.notes += f" [自动修正: 素材仅{shot.fps}fps，无法升格]"

        return decisions

    def _post_process_protected_shots(self, decisions: List[EditDecision], shots: List[Shot]) -> List[EditDecision]:
        """保护强制保留/待复核素材不被删除；非保护素材执行 LLM 删除决策"""
        protected_statuses = {"强制保留", "待复核"}
        shot_map = {s.shot_id: s for s in shots}

        kept_decisions = []
        for d in decisions:
            shot = shot_map.get(d.shot_id)
            if not shot:
                continue

            if d.speed == "删除":
                if getattr(shot, "status", "") in protected_statuses:
                    logger.warning(f"{d.shot_id} 为 {shot.status} 素材，撤销删除决策，改为原速保留")
                    d.speed = "1x"
                    d.notes += f" [自动修正: {shot.status} 素材不可删除]"
                else:
                    continue
            kept_decisions.append(d)

        return kept_decisions

    # ------------------------------------------------------------------
    # 时长预算控制
    # ------------------------------------------------------------------
    def _adjust_duration(
        self,
        decisions: List[EditDecision],
        shots: List[Shot],
        supplement_shots: List[Shot],
    ) -> List[EditDecision]:
        """
        段落化时长控制：按剧情段落（beat）分组，每个 beat 先只选最佳主镜头，
        通过 0.75~1.5x 变速尝试覆盖该 beat 目标时长；不能覆盖则加一个高积分衔接镜头，
        新组合再变速，直到覆盖或备选耗尽。
        """
        if self.target_duration <= 0:
            return decisions

        shot_map = {s.shot_id: s for s in shots}

        # 1. 按 beat 分组决策和素材
        beat_decisions_map = self._group_decisions_by_beat(decisions, shot_map)
        beat_candidates = self._group_shots_by_beat(shots)
        beat_supplements = self._group_shots_by_beat(supplement_shots)

        # 2. 从剧本分析读取每个 beat 的目标时长；没有则平均分配
        beat_targets = self._build_beat_targets(beat_decisions_map, shot_map)

        # 3. 逐 beat 构建时间线
        all_beat_decisions: List[EditDecision] = []
        for beat_id in self._sorted_beat_ids(beat_decisions_map, shot_map):
            target = beat_targets.get(beat_id, self.target_duration / max(len(beat_targets), 1))
            candidates = beat_candidates.get(beat_id, [])
            supps = beat_supplements.get(beat_id, [])

            # 已属于该 beat 的决策（来自 LLM 或之前处理）作为初始选择
            initial_decisions = beat_decisions_map.get(beat_id, [])

            beat_timeline = self._build_beat_timeline(
                beat_id=beat_id,
                initial_decisions=initial_decisions,
                candidate_shots=candidates,
                supplement_shots=supps,
                beat_target=target,
                shot_map=shot_map,
            )
            all_beat_decisions.extend(beat_timeline)

        # 4. 把未匹配到 beat 的决策保留在末尾（不应依赖它们，但避免丢失）
        unmatched_decisions = [d for d in decisions if self._decision_beat(d, shot_map) == "UNMATCHED"]
        all_beat_decisions.extend(unmatched_decisions)

        # 5. 合并后按叙事顺序排序
        all_beat_decisions = self._sort_decisions_by_narrative(all_beat_decisions, shots)
        for i, d in enumerate(all_beat_decisions):
            d.sequence = i + 1

        # 6. 全局保护：仍超出/不足时做整体压缩或补充
        final_total = self._projected_duration(all_beat_decisions, shots)
        logger.info(f"Beat 级时长调整后投影: {final_total:.1f}s")
        if final_total > self.target_max:
            all_beat_decisions = self._compress_to_target(all_beat_decisions, shots)
        elif final_total < self.target_min:
            all_beat_decisions = self._extend_to_target(all_beat_decisions, shots, supplement_shots)

        final_total = self._projected_duration(all_beat_decisions, shots)
        logger.info(f"最终时长调整后投影: {final_total:.1f}s")
        return all_beat_decisions

    def _decision_beat(self, decision: EditDecision, shot_map: Dict[str, Shot]) -> str:
        """获取决策对应的 beat_id"""
        shot = shot_map.get(decision.shot_id)
        if not shot or not shot.script_anchor:
            return "UNMATCHED"
        return shot.script_anchor.get("beat", "UNMATCHED")

    def _group_decisions_by_beat(
        self,
        decisions: List[EditDecision],
        shot_map: Dict[str, Shot],
    ) -> Dict[str, List[EditDecision]]:
        """按剧情段落 beat 对决策分组"""
        groups = {}
        for d in decisions:
            shot = shot_map.get(d.shot_id)
            beat = "UNMATCHED"
            if shot and shot.script_anchor:
                beat = shot.script_anchor.get("beat", "UNMATCHED")
            groups.setdefault(beat, []).append(d)
        return groups

    def _group_shots_by_beat(self, shots: List[Shot]) -> Dict[str, List[Shot]]:
        """按剧情段落 beat 对 Shot 分组"""
        groups = defaultdict(list)
        for shot in shots:
            beat = "UNMATCHED"
            if shot.script_anchor:
                beat = shot.script_anchor.get("beat", "UNMATCHED")
            if beat != "UNMATCHED":
                groups[beat].append(shot)
        return dict(groups)

    def _build_beat_targets(
        self,
        beat_decisions_map: Dict[str, List[EditDecision]],
        shot_map: Dict[str, Shot],
    ) -> Dict[str, float]:
        """从 shot.script_anchor 中附加的 beat 分析信息构建每个 beat 的目标时长"""
        targets = {}
        for beat_id, decisions in beat_decisions_map.items():
            for d in decisions:
                shot = shot_map.get(d.shot_id)
                if not shot or not shot.script_anchor:
                    continue
                est = shot.script_anchor.get("estimated_duration")
                if est is not None and float(est) > 0:
                    targets[beat_id] = float(est)
                    break

        # 剩余的 beat 平均分配剩余时长
        num_beats = len(beat_decisions_map)
        if num_beats == 0:
            return targets
        assigned_sum = sum(targets.values())
        remaining = max(0.0, self.target_duration - assigned_sum)
        unassigned = [b for b in beat_decisions_map.keys() if b not in targets]
        if unassigned:
            avg = remaining / len(unassigned)
            for b in unassigned:
                targets[b] = avg
        return targets

    def _sorted_beat_ids(
        self,
        beat_decisions_map: Dict[str, List[EditDecision]],
        shot_map: Dict[str, Shot],
    ) -> List[str]:
        """按叙事顺序返回 beat_id 列表"""
        def beat_sort_key(beat_id: str) -> tuple:
            # 通过任意一个决策里的 shot 提取 act/scene 信息
            decisions = beat_decisions_map.get(beat_id, [])
            act_num, scene_num = 99, 99
            for d in decisions:
                shot = shot_map.get(d.shot_id)
                if not shot or not shot.script_anchor:
                    continue
                anchor = shot.script_anchor
                act_match = re.search(r"第(\d+)幕", anchor.get("act", ""))
                scene_match = re.search(r"场(\d+)", beat_id)
                if act_match:
                    act_num = int(act_match.group(1))
                if scene_match:
                    scene_num = int(scene_match.group(1))
                break
            return (act_num, scene_num, beat_id)

        return sorted(beat_decisions_map.keys(), key=beat_sort_key)

    def _build_beat_timeline(
        self,
        beat_id: str,
        initial_decisions: List[EditDecision],
        candidate_shots: List[Shot],
        supplement_shots: List[Shot],
        beat_target: float,
        shot_map: Dict[str, Shot],
    ) -> List[EditDecision]:
        """
        按用户规则构建单个 beat 的时间线：
        1. 先选一个最佳主镜头，通过 0.75~1.5x 变速看能否覆盖 beat_target。
        2. 不能覆盖则恢复原速，从候选/备选池按积分加衔接镜头。
        3. 新组合再变速，循环直到覆盖或备选耗尽。
        """
        if beat_target <= 0:
            # 确保返回的决策有 beat_id
            for d in initial_decisions:
                self._ensure_beat_meta(d, shot_map)
            return initial_decisions

        # 确保 initial_decisions 的 beat 元数据正确
        for d in initial_decisions:
            self._ensure_beat_meta(d, shot_map)

        min_speed = 0.75
        max_speed = 1.5
        tolerance = 0.12  # beat 内部允许 ±12% 浮动

        # 限制补充镜头数量：最多 required_shots_count * 2 个，或最少 1 个
        beat_info = self.beat_map.get(beat_id, {})
        max_supplements = max(int(beat_info.get('required_shots_count', 2)) * 2, 1)
        supplement_count = 0

        # 当前已选决策
        selected: List[EditDecision] = list(initial_decisions)
        used_shot_ids = {d.shot_id for d in selected}
        used_source_files = {d.source_file for d in selected if d.source_file}

        # 可用的候选/备选镜头
        available_candidates = [s for s in candidate_shots if s.shot_id not in used_shot_ids]
        available_supplements = [s for s in supplement_shots if s.shot_id not in used_shot_ids]

        # 如果初始为空，先选最佳主镜头
        if not selected and available_candidates:
            best = self._pick_best_main_shot(available_candidates, beat_id, beat_target, used_source_files)
            if best:
                selected.append(self._shot_to_decision(best, "1x", beat_id=beat_id))
                used_shot_ids.add(best.shot_id)
                used_source_files.add(best.source_file)
                available_candidates = [s for s in available_candidates if s.shot_id != best.shot_id]

        # 循环：变速 -> 不足/超出 -> 加镜头或删镜头
        for _ in range(10):  # 安全上限
            if not selected:
                break

            raw_dur = sum(shot_map[d.shot_id].duration_sec for d in selected if d.shot_id in shot_map)
            if raw_dur <= 0:
                break

            desired_mult = raw_dur / beat_target
            clamped_mult = max(min_speed, min(max_speed, desired_mult))
            projected = raw_dur / clamped_mult

            # 已满足目标时长
            if beat_target * (1 - tolerance) <= projected <= beat_target * (1 + tolerance):
                self._apply_uniform_speed(selected, clamped_mult, beat_id)
                break

            # 原始时长太长：即使 1.5x 仍超出 -> 删除低质量镜头
            if clamped_mult == max_speed and projected > beat_target * (1 + tolerance):
                removable = [
                    d for d in selected
                    if shot_map.get(d.shot_id)
                    and getattr(shot_map[d.shot_id], "status", "") not in {"强制保留", "待复核"}
                ]
                if not removable:
                    self._apply_uniform_speed(selected, clamped_mult, beat_id)
                    break
                # 按综合评分删最低的，而不只是 quality_score
                removable.sort(
                    key=lambda d: self._score_shot_for_beat(
                        shot_map[d.shot_id], beat_id, beat_target, None,
                        set(), set(), set()
                    )
                )
                worst = removable[0]
                selected.remove(worst)
                used_shot_ids.discard(worst.shot_id)
                used_source_files.discard(worst.source_file)
                logger.info(f"beat {beat_id} 原始时长过长，删除低信息量镜头 {worst.shot_id}")
                continue

            # 原始时长太短：需要加衔接镜头
            # 限制补充次数，避免塞入过多镜头
            if supplement_count >= max_supplements:
                logger.info(f"beat {beat_id} 已达到最大补充镜头数 {max_supplements}，停止补充")
                self._apply_uniform_speed(selected, clamped_mult, beat_id)
                break

            # 把当前镜头先恢复原速
            for d in selected:
                if d.speed != "1x":
                    d.speed = "1x"
                    d.notes = (d.notes + " [恢复1x准备加衔接镜头]").strip()

            best_sup = self._pick_best_supplement(
                selected, shot_map, available_candidates + available_supplements,
                used_source_files, beat_id, beat_target
            )
            if not best_sup:
                logger.warning(f"beat {beat_id} 无可用衔接镜头，使用当前选择")
                self._apply_uniform_speed(selected, clamped_mult, beat_id)
                break

            selected.append(self._shot_to_decision(best_sup, "1x", beat_id=beat_id))
            used_shot_ids.add(best_sup.shot_id)
            used_source_files.add(best_sup.source_file)
            available_candidates = [s for s in available_candidates if s.shot_id != best_sup.shot_id]
            available_supplements = [s for s in available_supplements if s.shot_id != best_sup.shot_id]
            supplement_count += 1

        # 最终兜底：统一变速
        raw_dur = sum(shot_map[d.shot_id].duration_sec for d in selected if d.shot_id in shot_map)
        if raw_dur > 0:
            desired_mult = raw_dur / beat_target
            clamped_mult = max(min_speed, min(max_speed, desired_mult))
            self._apply_uniform_speed(selected, clamped_mult, beat_id)

        return selected

    def _ensure_beat_meta(self, d: EditDecision, shot_map: Dict[str, Shot]):
        """确保 EditDecision 的 beat_id/act/scene 字段已填充"""
        if d.beat_id:
            return
        shot = shot_map.get(d.shot_id)
        anchor = shot.script_anchor if shot else None
        if anchor:
            beat_id = anchor.get("beat", "")
            d.beat_id = beat_id
            d.act = anchor.get("act", "")
            d.scene = re.search(r"场\d+", beat_id).group(0) if re.search(r"场\d+", beat_id) else ""

    def _apply_uniform_speed(self, decisions: List[EditDecision], mult: float, beat_id: str):
        """对 beat 内所有决策应用统一变速"""
        speed_str = format_speed(mult)
        for d in decisions:
            d.speed = speed_str
            if "段落变速" not in d.notes:
                d.notes = (d.notes + f" [段落变速: {speed_str}]").strip()
        logger.info(f"beat {beat_id} 应用段落变速: {speed_str}")

    def _pick_best_main_shot(
        self,
        candidates: List[Shot],
        beat_id: str,
        beat_target: float,
        used_sources_global: Set[str],
    ) -> Optional[Shot]:
        """为 beat 挑选最佳主镜头"""
        if not candidates:
            return None
        scored = []
        for shot in candidates:
            score = self._score_shot_for_beat(
                shot=shot,
                beat_id=beat_id,
                beat_target=beat_target,
                prev_shot=None,
                used_sources_global=used_sources_global,
                used_sources_in_beat=set(),
                used_shot_ids=set(),
            )
            scored.append((score, shot))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        logger.info(
            f"beat {beat_id} 选择主镜头: {best[1].shot_id} "
            f"(score={best[0]:.2f}, quality={best[1].quality_score:.2f}, "
            f"dur={best[1].duration_sec:.2f}s, source={best[1].source_file})"
        )
        return best[1]

    def _shot_to_decision(
        self,
        shot: Shot,
        speed: str,
        sequence: int = 0,
        technique: str = "连续剪辑",
        transition: str = "硬切",
        audio: str = "保留原声",
        purpose: str = "",
        notes: str = "",
        beat_id: str = "",
    ) -> EditDecision:
        """把 Shot 转成 EditDecision"""
        anchor = shot.script_anchor or {}
        beat_id = beat_id or anchor.get("beat", "")
        return EditDecision(
            sequence=sequence,
            shot_id=shot.shot_id,
            source_file=shot.source_file,
            tc_in=shot.tc_in,
            tc_out=shot.tc_out,
            speed=speed,
            technique=technique,
            transition=transition,
            audio=audio,
            purpose=purpose or anchor.get("function", "") if anchor else "叙事镜头",
            notes=notes,
            beat_id=beat_id,
            act=anchor.get("act", ""),
            scene=re.search(r"场\d+", beat_id).group(0) if re.search(r"场\d+", beat_id) else "",
        )

    def _projected_duration(self, decisions: List[EditDecision], shots: List[Shot] = None) -> float:
        """计算当前决策的投影总时长"""
        shot_map = {s.shot_id: s for s in (shots or [])}
        total = 0.0
        for d in decisions:
            mult = parse_speed_multiplier(d.speed)
            if mult is None or mult <= 0:
                continue
            dur = shot_map.get(d.shot_id)
            if dur:
                dur = dur.duration_sec / mult
            else:
                dur = parse_duration_from_tc(d.tc_in, d.tc_out) / mult
            total += dur
        return total

    def _balance_beat_duration(
        self,
        beat_id: str,
        group_decisions: List[EditDecision],
        shot_map: Dict[str, Shot],
        supplement_shots: List[Shot],
        beat_target: float,
    ) -> List[EditDecision]:
        """
        平衡单个 beat 段落的时长：
        1. 先通过 0.75~1.5 倍变速尝试覆盖目标时长
        2. 仍不足则从备选池补充高积分接续镜头
        3. 补充后再变速，直到满足或备选耗尽

        返回：本次新增到段落的 EditDecision 列表
        """
        min_speed = 0.75
        max_speed = 1.5
        used_source_files = {d.source_file for d in group_decisions}
        added_decisions: List[EditDecision] = []

        def _group_raw_duration() -> float:
            return sum(
                shot_map[d.shot_id].duration_sec
                for d in group_decisions
                if d.shot_id in shot_map
            )

        def _apply_uniform_speed():
            """对段落内所有镜头应用统一变速，使总时长接近 beat_target"""
            raw = _group_raw_duration()
            if raw <= 0:
                return
            desired_mult = raw / beat_target
            desired_mult = max(min_speed, min(max_speed, desired_mult))
            speed_str = format_speed(desired_mult)
            for d in group_decisions:
                # 保护 PROCESSED 素材的 LLM 决策
                shot = shot_map.get(d.shot_id)
                if shot and shot.state == "PROCESSED":
                    continue
                d.speed = speed_str
                d.notes = (d.notes + f" [段落变速: {speed_str}]").strip()

        _apply_uniform_speed()

        # 如果变速后仍不足，从备选池补充
        while True:
            current_dur = sum(
                shot_map[d.shot_id].duration_sec / parse_speed_multiplier(d.speed)
                for d in group_decisions
                if d.shot_id in shot_map and parse_speed_multiplier(d.speed) > 0
            )
            if current_dur >= beat_target * 0.75:
                break

            # 从备选池选最佳接续镜头
            best_sup = self._pick_best_supplement(
                group_decisions, shot_map, supplement_shots, used_source_files, beat_id, beat_target
            )
            if not best_sup:
                logger.warning(f"段落 {beat_id} 无可用备选镜头，无法补足时长")
                break

            new_decision = EditDecision(
                sequence=0,
                shot_id=best_sup.shot_id,
                source_file=best_sup.source_file,
                tc_in=best_sup.tc_in,
                tc_out=best_sup.tc_out,
                speed="1x",
                technique="补充插入",
                transition="硬切",
                audio="保留原声",
                purpose="补充段落时长",
                notes=f"段落 {beat_id} 时长不足，自动补充接续镜头",
            )
            group_decisions.append(new_decision)
            added_decisions.append(new_decision)
            used_source_files.add(best_sup.source_file)
            supplement_shots = [s for s in supplement_shots if s.shot_id != best_sup.shot_id]
            _apply_uniform_speed()

        return added_decisions

    def _pick_best_supplement(
        self,
        group_decisions: List[EditDecision],
        shot_map: Dict[str, Shot],
        supplement_shots: List[Shot],
        used_source_files_global: Set[str],
        beat_id: str,
        beat_target: float,
    ) -> Optional[Shot]:
        """为段落挑选最佳接续备选镜头，使用统一评分"""
        if not supplement_shots:
            return None

        used_shot_ids = {d.shot_id for d in group_decisions}
        used_sources_in_beat = {d.source_file for d in group_decisions if d.source_file}
        last_decision = group_decisions[-1] if group_decisions else None
        last_shot = shot_map.get(last_decision.shot_id) if last_decision else None

        scored = []
        for sup in supplement_shots:
            score = self._score_shot_for_beat(
                shot=sup,
                beat_id=beat_id,
                beat_target=beat_target,
                prev_shot=last_shot,
                used_sources_global=used_source_files_global,
                used_sources_in_beat=used_sources_in_beat,
                used_shot_ids=used_shot_ids,
            )
            if score <= -900:
                continue
            scored.append((score, sup))

        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        logger.info(
            f"beat {beat_id} 选择衔接镜头: {best[1].shot_id} "
            f"(score={best[0]:.2f}, quality={best[1].quality_score:.2f}, "
            f"semantic={self._semantic_fit(best[1], beat_id):.2f})"
        )
        return best[1]

    def _is_continuous_action(self, shot_a: Shot, shot_b: Shot) -> bool:
        """判断两个镜头是否属于连续动作，允许同一 source_file 连续"""
        if not shot_a or not shot_b:
            return False
        # 同素材且时间连续
        if shot_a.source_file == shot_b.source_file:
            try:
                a_in = tc_to_sec(shot_a.tc_in, shot_a.fps)
                a_out = tc_to_sec(shot_a.tc_out, shot_a.fps)
                b_in = tc_to_sec(shot_b.tc_in, shot_b.fps)
                # 时间码相邻或重叠
                if abs(b_in - a_out) <= 1.0:
                    return True
            except Exception:
                pass
        # 动作细节暗示连续
        a_details = (shot_a.action_details or "").lower()
        b_details = (shot_b.action_details or "").lower()
        if a_details and b_details:
            for verb in ["跑", "追", "跳", "走", "飞", "转身"]:
                if verb in a_details and verb in b_details:
                    return True
        return False

    @staticmethod
    def _direction_continuity_bonus(dir_a: str, dir_b: str) -> float:
        """方向连续性奖励"""
        a = str(dir_a).lower()
        b = str(dir_b).lower()
        if ("左到右" in a and "右到左" in b) or ("右到左" in a and "左到右" in b):
            return -0.5  # 越轴风险扣分
        if ("左到右" in a and "左到右" in b) or ("右到左" in a and "右到左" in b):
            return 0.3
        if "静止" in a or "静止" in b:
            return 0.2
        return 0.0

    def _compress_to_target(self, decisions: List[EditDecision], shots: List[Shot]) -> List[EditDecision]:
        """总时长超出上限：优先删除低信息量/低语义匹配镜头；不粗暴全局提速"""
        shot_map = {s.shot_id: s for s in shots}
        protected_statuses = {"强制保留", "待复核"}

        # 第一阶段：给仍有提速空间的镜头适度提速（最大到 2.0x），但保留 beat 级变速决策
        for d in decisions:
            shot = shot_map.get(d.shot_id)
            if not shot:
                continue
            if getattr(shot, 'status', '') in protected_statuses:
                continue
            mult = parse_speed_multiplier(d.speed)
            if mult is None or mult <= 0:
                continue
            # 只在当前速度基础上适度提速，不超过 2.0x
            if mult < 1.5:
                new_mult = min(1.5, mult * 1.2)
                if abs(new_mult - mult) > 0.05:
                    d.speed = format_speed(new_mult)
                    d.notes += " [自动压缩: 适度提速以控制总时长]"

        total = self._projected_duration(decisions, shots)
        if total <= self.target_max:
            return decisions

        # 第二阶段：按综合评分从低到高删除非保护镜头
        indexed = list(enumerate(decisions))
        indexed.sort(
            key=lambda x: self._score_shot_for_beat(
                shot_map.get(x[1].shot_id),
                x[1].beat_id,
                0.0, None, set(), set(), set()
            ) if shot_map.get(x[1].shot_id) else -999
        )

        kept = []
        removed_ids = set()
        for idx, d in indexed:
            shot = shot_map.get(d.shot_id)
            if total <= self.target_max:
                kept.append(d)
                continue
            if shot and getattr(shot, 'status', '') in protected_statuses:
                kept.append(d)
                continue
            removed_ids.add(d.shot_id)
            total = self._projected_duration(
                [decisions[i] for i, _ in enumerate(decisions) if decisions[i].shot_id not in removed_ids],
                shots,
            )
            logger.info(f"自动删除低信息量镜头 {d.shot_id} 以控制时长")

        # 保留未被删除的决策（保持原有顺序）
        final = [d for d in decisions if d.shot_id not in removed_ids]
        return final

    def _extend_to_target(
        self,
        decisions: List[EditDecision],
        shots: List[Shot],
        supplement_shots: List[Shot],
    ) -> List[EditDecision]:
        """总时长低于下限：先慢放可延展镜头，仍不足则从备选池补充"""
        shot_map = {s.shot_id: s for s in shots}
        min_fps = self.processing.get('slow_motion_min_fps', 60)

        total = self._projected_duration(decisions, shots)
        deficit = self.target_min - total
        if deficit <= 0:
            return decisions

        # 第一阶段：慢放现有 RAW / 非 PROCESSED 镜头
        for i, d in enumerate(decisions):
            if deficit <= 0:
                break
            shot = shot_map.get(d.shot_id)
            if not shot or shot.state == 'PROCESSED':
                continue

            mult = parse_speed_multiplier(d.speed)
            if mult is None or mult <= 0:
                mult = 1.0

            is_special = self._is_special_slow_motion(shot, min_fps)
            max_dur_mult = 4.0 if is_special else 2.0  # 特殊镜头最多 4x，普通最多 2x

            # 物理帧率不足时禁止升格（PROCESSED 素材已交给 LLM，不在这里限制）
            if shot.state != 'PROCESSED' and shot.fps < min_fps:
                max_dur_mult = 1.0

            # 避免连续 3 个以上特殊升格镜头
            if is_special and max_dur_mult > 1.0:
                neighbors_special = 0
                for j in range(max(0, i - 2), min(len(decisions), i + 3)):
                    if j == i:
                        continue
                    neighbor = shot_map.get(decisions[j].shot_id)
                    if neighbor and self._is_special_slow_motion(neighbor, min_fps):
                        neighbors_special += 1
                if neighbors_special >= 2:
                    # 把最大延展降到 1.5x，避免连续特殊升格
                    max_dur_mult = 1.5

            current_dur = shot.duration_sec / mult
            max_dur = shot.duration_sec * max_dur_mult
            available = max_dur - current_dur
            if available <= 0:
                continue

            add = min(available, deficit)
            new_dur = current_dur + add
            new_mult = shot.duration_sec / new_dur
            d.speed = format_speed(new_mult)
            d.notes += " [自动延展: 慢放以补足时长]"
            total = self._projected_duration(decisions, shots)
            deficit = self.target_min - total

        # 第二阶段：从备选池补充镜头
        if deficit > 0 and supplement_shots:
            supplement_shots = self._sort_by_narrative(supplement_shots)
            used_source_files = {d.source_file for d in decisions}
            for sup in supplement_shots:
                if deficit <= 0:
                    break
                if any(d.shot_id == sup.shot_id for d in decisions):
                    continue

                # 避免同一原始素材多次出现，防止画面重复
                if sup.source_file in used_source_files:
                    continue

                sup_dur = sup.duration_sec
                # 补充镜头不能导致超出上限
                if total + sup_dur > self.target_max:
                    continue

                decisions.append(EditDecision(
                    sequence=0,
                    shot_id=sup.shot_id,
                    source_file=sup.source_file,
                    tc_in=sup.tc_in,
                    tc_out=sup.tc_out,
                    speed="1x",
                    technique="补充插入",
                    transition="硬切",
                    audio="保留原声",
                    purpose="补充时长 / 缺失情节点",
                    notes="由时长不足自动从备选池补充",
                ))
                used_source_files.add(sup.source_file)
                total = self._projected_duration(decisions, shots)
                deficit = self.target_min - total

        return decisions

    @staticmethod
    def _is_special_slow_motion(shot: Shot, min_fps: float) -> bool:
        """判断是否为特殊升格/子弹时间镜头"""
        special_tags = {"升格", "子弹时间", "bullet_time", "slow_motion", "慢动作"}
        tags = set(shot.tags or [])
        if tags & special_tags:
            return True
        if shot.fps >= min_fps:
            return True
        if any(t in (shot.camera_movement or "") for t in ["升格", "子弹时间"]):
            return True
        return False

    # ------------------------------------------------------------------
    # 镜头评分（重构）
    # ------------------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> Set[str]:
        """提取中文词元，用于语义匹配"""
        if not text:
            return set()
        stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
            "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "把", "被", "让", "向",
            "过", "能", "个", "她", "他", "它", "这", "那", "为", "之", "与", "及", "等", "或",
        }
        text = str(text).strip().lower()
        words: Set[str] = set()
        # 按标点分词
        delimiters = set("，、。！？；：""''（）(),.!?;:\"'() ")
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
        return {w for w in words if w not in stopwords}

    def _semantic_fit(self, shot: Shot, beat_id: str) -> float:
        """计算镜头内容与情节点关键内容的语义重叠度 0-1。

        按动作 > 剧情内容 > 台词的优先级加权，降低 ASR 识别错误的影响。
        """
        beat = self.beat_map.get(beat_id)
        if not beat:
            return 0.0

        shot_text = " ".join(filter(None, [
            shot.action or "",
            getattr(shot, 'action_details', '') or "",
            shot.asr_text or "",
            shot.dialogue or "",
            ", ".join(shot.characters or []),
        ]))
        if not shot_text:
            return 0.0

        def _fit(beat_text: str) -> float:
            beat_tokens = self._tokens(beat_text)
            shot_tokens = self._tokens(shot_text)
            if not beat_tokens or not shot_tokens:
                return 0.0
            overlap = beat_tokens & shot_tokens
            recall = len(overlap) / len(beat_tokens)
            precision = len(overlap) / len(shot_tokens)
            if recall + precision <= 0:
                return 0.0
            return 2 * recall * precision / (recall + precision)

        action_fit = _fit(" ".join(beat.get('key_actions', [])))
        content_fit = _fit(beat.get('content', ''))
        dialogue_fit = _fit(beat.get('key_dialogue', ''))

        # 动作权重最高，台词权重最低
        return action_fit * 0.55 + content_fit * 0.30 + dialogue_fit * 0.15

    def _score_shot_for_beat(
        self,
        shot: Shot,
        beat_id: str,
        beat_target: float,
        prev_shot: Optional[Shot],
        used_sources_global: Set[str],
        used_sources_in_beat: Set[str],
        used_shot_ids: Set[str],
    ) -> float:
        """综合评分：为某个 beat 选择镜头"""
        if shot.shot_id in used_shot_ids:
            return -999.0

        score = 0.0
        anchor = shot.script_anchor or {}

        # 1. 剧本锚定置信度（最高权重）
        score += (anchor.get("confidence") or 0.0) * 1.5

        # 2. 内容与 beat 的语义匹配
        semantic_fit = self._semantic_fit(shot, beat_id)
        score += semantic_fit * 1.2

        # 3. Phase 2 质量分
        score += getattr(shot, "quality_score", 0.0) * 0.25

        # 4. 选择状态优先级
        status = getattr(shot, "status", "备选")
        if status == "核心":
            score += 0.8
        elif status == "保留":
            score += 0.5
        elif status == "备选":
            score += 0.2
        elif status in {"强制保留", "待复核"}:
            score += 1.0

        # 5. 时长匹配
        if beat_target > 0 and shot.duration_sec > 0:
            fit = 1.0 - min(abs(shot.duration_sec - beat_target) / beat_target, 1.0)
            score += fit * 0.4

        # 6. 与上一镜头的连续性
        if prev_shot:
            if self._is_continuous_action(prev_shot, shot):
                score += 0.6
            # 方向连续性
            score += self._direction_continuity_bonus(
                getattr(prev_shot, "direction", ""),
                getattr(shot, "direction", ""),
            )

        # 7. source_file 复用惩罚
        if shot.source_file in used_sources_in_beat:
            score -= 3.0  # 同一 beat 内 hard penalty
        elif shot.source_file in used_sources_global:
            # 跨 beat 复用：若不是时间连续动作，给予重罚
            is_continuous = False
            if prev_shot:
                is_continuous = self._is_continuous_action(prev_shot, shot)
            if is_continuous:
                score -= 0.3  # 连续动作可接受
            else:
                score -= 1.8  # 非连续重复画面重罚

        return score

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------
    def _export_csv(self, decisions: List[EditDecision]):
        """导出 CSV 审核表"""
        import csv

        csv_path = os.path.join(self.output_dir, 'phase3_edit_decision.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                '序列', '镜头ID', '源文件', '入点', '出点', '速度',
                '剪辑手法', '转场', '音频', '叙事目的', '备注'
            ])

            for d in decisions:
                writer.writerow([
                    d.sequence, d.shot_id, d.source_file,
                    d.tc_in, d.tc_out, d.speed,
                    d.technique, d.transition, d.audio,
                    d.purpose, d.notes
                ])

        logger.info(f"剪辑决策 CSV 已导出: {csv_path}")


# ----------------------------------------------------------------------
# 速度字符串工具
# ----------------------------------------------------------------------
def parse_speed_multiplier(speed: str) -> Optional[float]:
    """把速度字符串解析为倍数

    支持:
    - "50%" -> 0.5
    - "200%" -> 2.0
    - "1x" / "2.5x" -> 1.0 / 2.5
    - "删除" -> None
    """
    if speed is None:
        return 1.0
    s = str(speed).strip().lower()
    if not s:
        return 1.0
    if s == "删除":
        return None

    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return 1.0

    if s.endswith('x'):
        try:
            return float(s[:-1])
        except ValueError:
            return 1.0

    try:
        return float(s)
    except ValueError:
        return 1.0


def format_speed(mult: float) -> str:
    """把倍数格式化为速度字符串"""
    if mult is None or abs(mult - 1.0) < 0.01:
        return "1x"
    if mult < 1.0:
        return f"{int(round(mult * 100))}%"
    # ≥1 使用 x 形式
    if abs(mult - round(mult)) < 0.01:
        return f"{int(round(mult))}x"
    return f"{mult:.1f}x"


def parse_duration_from_tc(tc_in: str, tc_out: str, fps: float = 24.0) -> float:
    """通过时间码计算时长"""
    from src.utils import tc_to_sec
    try:
        return max(0.0, tc_to_sec(tc_out, fps) - tc_to_sec(tc_in, fps))
    except Exception:
        return 0.0
