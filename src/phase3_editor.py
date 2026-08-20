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
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from src.utils import Shot, save_json, load_json, logger, parse_duration_string


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
        self.llm_client = self._init_llm_client()

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

    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        import openai
        provider = self.models['llm']['provider']
        base_url = self.models['llm'].get('base_url')
        if provider == 'deepseek':
            base_url = base_url or "https://api.deepseek.com"

        return openai.OpenAI(
            api_key=self.models['llm']['api_key'],
            base_url=base_url
        )

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
        decisions = self._adjust_duration(decisions, core_shots, supplement_shots)

        # 重新排序并编号
        decisions = self._sort_decisions_by_narrative(decisions)
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

    def _sort_decisions_by_narrative(self, decisions: List[EditDecision]) -> List[EditDecision]:
        """按叙事顺序对决策排序"""
        # 通过 shot_id 反查 shot 的 script_anchor
        # 这里无法直接拿到 shot，因此保留原有顺序，仅把 sequence=0 的补充镜头按 shot_id 排到最后
        return sorted(decisions, key=lambda d: (d.sequence if d.sequence else 9999, d.shot_id))

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

        prompt = self.prompt_template.format(
            project_name=self.project.get('name', '未命名'),
            project_style=self.project.get('style', ''),
            project_genre=self.project.get('genre', '剧情短片'),
            target_duration=f"{self.target_duration:.1f}",
            target_duration_min=f"{self.target_min:.1f}",
            target_duration_max=f"{self.target_max:.1f}",
            current_core_duration=f"{current_core_duration:.1f}",
            warnings_text=warnings_text,
            shots_text=shots_text,
            supplement_text=supplement_text,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.models['llm']['model'],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.models['llm'].get('max_tokens', 8192),
                temperature=0.4
            )

            content = response.choices[0].message.content
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

            decisions.append(EditDecision(
                sequence=start_seq + i + 1,
                shot_id=shot_id,
                source_file=shot.source_file,
                tc_in=shot.tc_in,
                tc_out=shot.tc_out,
                speed=speed if keep else '删除',
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
            EditDecision(
                sequence=start_seq + i + 1,
                shot_id=s.shot_id,
                source_file=s.source_file,
                tc_in=s.tc_in,
                tc_out=s.tc_out,
                speed="1x",
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
        """根据目标时长 ±5% 自动压缩、延展或补充镜头"""
        if self.target_duration <= 0:
            return decisions

        total = self._projected_duration(decisions)
        logger.info(
            f"目标时长: {self.target_duration:.1f}s (区间 {self.target_min:.1f}s~{self.target_max:.1f}s), "
            f"当前投影: {total:.1f}s"
        )

        if total > self.target_max:
            decisions = self._compress_to_target(decisions, shots)
        elif total < self.target_min:
            decisions = self._extend_to_target(decisions, shots, supplement_shots)

        final_total = self._projected_duration(decisions)
        logger.info(f"时长调整后投影: {final_total:.1f}s")
        return decisions

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

    def _compress_to_target(self, decisions: List[EditDecision], shots: List[Shot]) -> List[EditDecision]:
        """总时长超出上限：先提速非关键/非保护镜头，仍超则删除低质量镜头"""
        shot_map = {s.shot_id: s for s in shots}
        protected_statuses = {"强制保留", "待复核"}

        # 第一阶段：提速非保护镜头到 200%
        for d in decisions:
            shot = shot_map.get(d.shot_id)
            if not shot:
                continue
            if getattr(shot, 'status', '') in protected_statuses:
                continue
            mult = parse_speed_multiplier(d.speed)
            if mult is None:
                continue
            if mult < 2.0:
                d.speed = "200%"
                d.notes += " [自动压缩: 提速到200%以控制总时长]"

        total = self._projected_duration(decisions, shots)
        if total <= self.target_max:
            return decisions

        # 第二阶段：按质量分从低到高删除非保护镜头
        indexed = list(enumerate(decisions))
        indexed.sort(key=lambda x: getattr(shot_map.get(x[1].shot_id), 'quality_score', 0.0))

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
            for sup in supplement_shots:
                if deficit <= 0:
                    break
                if any(d.shot_id == sup.shot_id for d in decisions):
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
