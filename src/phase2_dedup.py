"""
Phase 2: 镜头选择 + 去重 + 覆盖检测

v0.2 重构目标：
- 不再是简单去重，而是基于剧本情节点的人选 take 决策。
- 完整消费 Phase 1 的状态、关系图、来源信息，避免功能割裂。
- 支持三种素材状态：RAW / PROCESSED / ANALYZED。
- 关系图高连贯性的镜头不被判重。

输出状态：
- 核心：该情节点最优 take
- 备选：同组其他可选 take
- 强制保留：PROCESSED 等不可删除素材
- 待复核：ANALYZED 等需要人工确认的素材
- 废弃：明确被淘汰
- 未匹配：未锚定到任何情节点
"""
import hashlib
import os
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

from src.models import Shot, ScriptBeat
from src.quality_scorer import QualityScorer, ScoreContext
from src.services.llm_service import LLMService
from src.utils import save_json, logger


class Phase2TakeSelector:
    """阶段2：镜头选择与去重"""

    def __init__(self, config: Dict, llm_service: LLMService = None):
        self.config = config
        self.processing = config.get("processing", {})
        self.output_dir = config["paths"]["output"]

        self.quality_scorer = QualityScorer(config.get("quality_scoring", {}))
        self.llm_service = llm_service or LLMService(config)

        # CLIP 初始化（仅当启用 L3 时）
        self.clip_model = None
        self.clip_processor = None
        if self.processing.get("enable_l3_semantic", True):
            self._init_clip()

        self._script_beats = {}

    def _init_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP 模型加载成功")
        except Exception as e:
            logger.warning(f"CLIP 加载失败，L3 语义去重将不可用: {e}")

    def run(
        self,
        shots: List[Shot],
        script_beats: List[ScriptBeat],
        beat_analysis: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[List[Shot], Dict]:
        """执行 Phase 3：按剧情顺序逐节点竞争式镜头匹配。

        流程：
        1. 全部镜头进入候选池，L1 文件级去重
        2. 按剧情顺序遍历每个情节点：
           - LLM 对候选池精排（match_score）
           - 本地 5 维积分（剧本匹配/画质/连贯/时长适配/元数据）
           - 按积分竞争，结合 0.75~1.5 倍速窗口做时长控制选镜
           - 选中镜头移出候选池，池中与其画面重复的镜头废弃（L2/L3）
        3. 剩余镜头标记备选；无核心覆盖的节点记入 missing_beats
        """
        logger.info("=" * 60)
        logger.info("Phase 3: 逐节点竞争式镜头匹配 + 去重 + 时长控制")
        logger.info("=" * 60)

        self._script_beats = {b.beat_id: b for b in script_beats}

        if not shots:
            logger.warning("Phase 3 输入为空")
            return [], self._empty_report()

        # 0. 全部镜头进入候选池，并预计算基础质量分（画质/连贯/时长/元数据 4 维，
        #    script_match 由逐节点 LLM 精排给出，竞争时再合成）
        for shot in shots:
            shot.status = "候选"
            shot.dedup_reason = ""
            shot.script_anchor = None
            shot.quality_score = self.quality_scorer.score_with_script_match(shot, 0.0, None)

        # 1. L1 文件级去重（MD5）
        self._l1_file_dedup(shots)

        # 2. 候选池
        pool = [s for s in shots if s.status != "废弃"]
        logger.info(f"候选池: {len(pool)} 个镜头，剧本节点: {len(script_beats)} 个")

        # 3. 按剧情顺序逐节点竞争选镜
        selections: Dict[str, Dict] = {}
        prev_selected: Optional[Shot] = None
        for i, beat in enumerate(script_beats):
            logger.info(f"[{i + 1}/{len(script_beats)}] 节点 {beat.beat_id} "
                        f"(预算 {beat.estimated_duration:.1f}s, 建议 {beat.required_shots_count} 镜)")
            selection, pool, prev_selected = self._select_for_beat(beat, pool, prev_selected)
            selections[beat.beat_id] = selection
            if selection["core"]:
                got = sum(c["planned_duration"] for c in selection["chosen"])
                logger.info(f"  选中 {len(selection['core'])} 镜: {' → '.join(selection['core'])}, "
                            f"配平时长 {got:.1f}s / 预算 {beat.estimated_duration:.1f}s")
            else:
                logger.warning(f"  无镜头入选，记入缺失")

        # 4. 候选池剩余 → 备选
        for shot in pool:
            shot.status = "备选"
            shot.dedup_reason = shot.dedup_reason or "未被任何节点选中"

        # 5. 缺失节点检测
        missing_beats = self._detect_missing_beats(selections, script_beats)

        # 6. 越轴/方向连续性预警
        axis_warnings = self._detect_axis_warnings(selections)

        # 7. 报告与导出
        report = self._generate_report(shots, selections, missing_beats, axis_warnings)
        save_json(report, os.path.join(self.output_dir, "phase2_deduplication.json"))
        self._export_csv(shots, report)

        selected_shots_path = os.path.join(self.output_dir, "phase2_selected_shots.json")
        save_json({"shots": [s.to_dict() for s in shots]}, selected_shots_path)
        logger.info(f"已保存 Phase 3 选择后的镜头列表: {selected_shots_path}")

        logger.info(f"Phase 3 完成: 核心 {len([s for s in shots if s.status == '核心'])}, "
                    f"备选 {len([s for s in shots if s.status == '备选'])}, "
                    f"废弃 {len([s for s in shots if s.status == '废弃'])}, "
                    f"缺失节点 {len(missing_beats)}")
        return shots, report

    # ------------------------------------------------------------------
    # 变装/状态转换节点加分
    # ------------------------------------------------------------------
    TRANSITION_KEYWORDS = [
        "变装", "换装", "变身", "男装褪去", "露出女装", "穿上男装",
        "女装", "男装", "长裙", "长袍", "跃起",
    ]

    def _is_transition_boost(self, beat: ScriptBeat, shot: Shot) -> bool:
        """变装/换装 beat 且镜头描述命中转换关键词 → 竞争积分加分。"""
        if not beat.gender_transition:
            return False
        desc = " ".join(filter(None, [
            shot.action or "",
            shot.action_details or "",
            shot.asr_text or "",
            shot.dialogue or "",
            ", ".join(shot.key_objects or []),
        ]))
        matched = [k for k in self.TRANSITION_KEYWORDS if k in desc]
        return len(matched) >= 2 or ("女装" in desc or "男装" in desc or "长裙" in desc)

    # ------------------------------------------------------------------
    # L1 文件级去重（MD5）
    # ------------------------------------------------------------------
    def _l1_file_dedup(self, shots: List[Shot]) -> None:
        """基于文件 MD5 的完全重复检测。保留质量分最高的副本；
        PROCESSED/ANALYZED 优先于 RAW。\n
        注意：RAW 素材被 Phase 0 粗剪后，多个片段可能共享同一个 source_path（原始视频），
        因此优先使用 split_clip_path（实际片段文件）计算 MD5，避免把不同粗剪片段误判为重复。
        """
        md5_groups: Dict[str, List[Shot]] = defaultdict(list)
        for shot in shots:
            # 优先使用实际片段文件路径
            cfg = (shot.cv_metadata or {}).get("shot_config", {})
            file_path = cfg.get("split_clip_path") or cfg.get("clip_path") or shot.source_path
            if not file_path or not os.path.exists(file_path):
                continue
            try:
                md5 = self._compute_md5(file_path)
                md5_groups[md5].append(shot)
            except Exception as e:
                logger.warning(f"MD5 计算失败 {shot.shot_id}: {e}")

        for md5, group in md5_groups.items():
            if len(group) <= 1:
                continue

            # 保护优先级：PROCESSED > ANALYZED > RAW
            def protect_priority(s: Shot) -> int:
                if s.state == "PROCESSED":
                    return 3
                if s.state == "ANALYZED":
                    return 2
                return 1

            sorted_group = sorted(
                group,
                key=lambda s: (protect_priority(s), s.quality_score),
                reverse=True,
            )
            keeper = sorted_group[0]
            for shot in sorted_group[1:]:
                if shot.status in ["核心", "强制保留"]:
                    continue
                shot.status = "废弃"
                shot.dedup_reason = f"L1文件级重复(MD5 {md5[:8]}...): 保留 {keeper.shot_id}"
                logger.info(f"L1 去重: {shot.shot_id} 与 {keeper.shot_id} 文件重复")

    @staticmethod
    def _compute_md5(file_path: str, chunk_size: int = 8192) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()

    # ------------------------------------------------------------------
    # 情节点分组
    # ------------------------------------------------------------------
    def _are_related(self, shot1: Shot, shot2: Shot) -> bool:
        """
        判断两个 Shot 是否强关联（不应判重）。

        强关联条件（满足其一即不判重）：
        1. 同一 shot_id
        2. 关系图直接关联且连贯性高（coherence_score >= 0.6）
        3. 关系类型为强连贯（情绪延续 / 动作衔接 / 对话衔接）
        4. 同一源文件且时间相邻，并且内容明显连续（continuity_score 高）——长镜头拆分保护
        5. 同角色且方向明显冲突（越轴风险），应保留供人工判断

        注意：同一源文件不再无条件保护，只有真正连续的长镜头拆分才保护。
        """
        # 同一镜头
        if shot1.shot_id == shot2.shot_id:
            return True

        # 关系图直接关联且连贯性高
        rel1 = shot1.relationships
        if rel1.prev and rel1.prev.shot_id == shot2.shot_id and rel1.prev.coherence_score >= 0.6:
            return True
        if rel1.next and rel1.next.shot_id == shot2.shot_id and rel1.next.coherence_score >= 0.6:
            return True

        # 关系类型为强连贯
        for rel in [rel1.prev, rel1.next]:
            if rel and rel.shot_id == shot2.shot_id and rel.relationship_type in ["情绪延续", "动作衔接", "对话衔接"]:
                return True

        # 来自同一源文件且时间相邻：只有真正连续的长镜头拆分才保护
        if (shot1.source_file == shot2.source_file
                and shot1.source_path == shot2.source_path):
            # 判断时间是否相邻（间隔 < 0.5 秒）
            try:
                from src.utils import tc_to_sec
                end1 = tc_to_sec(shot1.tc_out, shot1.fps)
                start2 = tc_to_sec(shot2.tc_in, shot2.fps)
                end2 = tc_to_sec(shot2.tc_out, shot2.fps)
                start1 = tc_to_sec(shot2.tc_in, shot2.fps)
                gap = min(abs(start2 - end1), abs(start1 - end2))
                if gap < 0.5:
                    # 必须内容连续（continuity_score 高 或 action 明显连续）
                    if max(shot1.continuity_score or 0.0, shot2.continuity_score or 0.0) >= 0.7:
                        return True
            except Exception:
                pass

        # 同角色且方向明显冲突：可能是越轴镜头，不应判重，应保留供人工判断
        if self._are_directions_conflicting(shot1, shot2):
            return True

        return False

    @staticmethod
    def _direction_to_sign(direction: str) -> int:
        """把方向描述转换为符号：左→右=+1，右→左=-1，其他=0"""
        if not direction:
            return 0
        d = str(direction)
        # 中文方向
        if "从左向右" in d or "左到右" in d or "向左到右" in d:
            return 1
        if "从右向左" in d or "右到左" in d or "向右到左" in d:
            return -1
        # 英文方向
        if "left to right" in d.lower() or "left-to-right" in d.lower():
            return 1
        if "right to left" in d.lower() or "right-to-left" in d.lower():
            return -1
        return 0

    def _are_directions_conflicting(self, shot1: Shot, shot2: Shot) -> bool:
        """判断两个镜头是否存在 180度/方向冲突（越轴风险）"""
        # 角色必须重叠
        chars1 = set(shot1.characters or [])
        chars2 = set(shot2.characters or [])
        if not chars1 or not chars2 or not (chars1 & chars2):
            return False

        sign1 = self._direction_to_sign(shot1.direction)
        sign2 = self._direction_to_sign(shot2.direction)
        # 只有双方都明确水平方向时才判断冲突
        if sign1 == 0 or sign2 == 0:
            return False

        # 方向相反 = 越轴风险
        return sign1 == -sign2

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        if len(hash1) != len(hash2):
            return 999
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _get_clip_features(self, image_path: str):
        from PIL import Image
        import numpy as np
        import torch
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return features.squeeze().numpy()

    # ------------------------------------------------------------------
    # 逐节点竞争式选镜（含时长变速控制）
    # ------------------------------------------------------------------
    def _select_for_beat(
        self,
        beat: ScriptBeat,
        pool: List[Shot],
        prev_shot: Optional[Shot],
    ) -> Tuple[Dict, List[Shot], Optional[Shot]]:
        """为单个情节点从候选池竞争选镜。

        逻辑：
        1. LLM 对候选池精排得到 match_score（剧情匹配度）
        2. 本地 5 维积分（script_match 用 LLM 精排分，其余本地算）
        3. 按积分从高到低，结合 0.75~1.5 倍速窗口做时长控制：
           - 镜头变速后能落在预算内 → 选定，speed = clamp(时长/预算, 0.75, 1.5)
           - 1.5 倍速仍超出预算 → 放弃该候选，看下一个
           - 0.75 倍速仍不足 → 作为衔接镜保留，继续追加下一个候选
        4. 选中镜头移出候选池，池中与其画面重复（L2/L3，受关系保护）的废弃

        返回: (selection, 新候选池, 本片最后一个已选镜头)
        """
        budget = float(beat.estimated_duration or 0.0)
        if budget <= 0:
            budget = 10.0  # 无预算时的兜底
        min_shots = max(1, int(beat.required_shots_count or 1))
        cfg_phase2 = self.config.get("phase2", {})
        hard_max = max(int(cfg_phase2.get("max_core_per_beat", 3)) * 2, min_shots + 2, 5)
        min_match = float(cfg_phase2.get("min_match_score", 0.25))

        selection: Dict[str, Any] = {"core": [], "chosen": []}

        if not pool:
            return selection, pool, prev_shot

        # 1) LLM 精排
        try:
            ranked = self.llm_service.rank_candidates_for_beat(beat, pool)
        except Exception as e:
            logger.error(f"节点 {beat.beat_id} LLM 精排失败: {e}")
            ranked = {}
        if not ranked:
            # 安全网：LLM 精排失败/返回空时，用本地积分兜底（script_match 取中性 0.3），
            # 保证节点不为空（尤其是用户锁定的高光节点）
            logger.warning(f"节点 {beat.beat_id} LLM 精排无结果，改用本地积分兜底选镜")
            ranked = {s.shot_id: {"match_score": 0.3, "reasoning": "LLM精排缺失，本地兜底"}
                      for s in pool}

        # 2) 积分（含跨节点连贯性上下文）
        cands = []
        for shot in pool:
            r = ranked.get(shot.shot_id)
            if not r:
                continue
            m = float(r.get("match_score", 0.0))
            if m < min_match:
                continue
            ctx = ScoreContext(prev_shot=prev_shot, target_segment_duration=budget)
            total = self.quality_scorer.score_with_script_match(shot, m, ctx)
            if self._is_transition_boost(beat, shot):
                total += 1.0  # 变装/状态转换镜头加分（满分 10 分制）
            cands.append({"shot": shot, "score": round(total, 2), "match": m,
                          "reasoning": r.get("reasoning", "")})
        cands.sort(key=lambda c: c["score"], reverse=True)

        if not cands:
            # 放宽阈值重试一次：取 match 最高者（哪怕低于阈值）
            best = max(
                ({"shot": s, "score": 0.0, "match": float(ranked[s.shot_id].get("match_score", 0.0)),
                  "reasoning": ranked[s.shot_id].get("reasoning", "")}
                 for s in pool if s.shot_id in ranked),
                key=lambda c: c["match"], default=None,
            )
            if best and best["match"] >= 0.1:
                logger.info(f"  阈值 {min_match} 内无候选，放宽至 best match={best['match']:.2f}")
                cands = [best]
            else:
                return selection, pool, prev_shot

        # 3) 时长变速竞争选镜
        chosen = []
        acc = 0.0
        for c in cands:
            if len(chosen) >= hard_max:
                break
            if acc >= budget * 0.9 and len(chosen) >= min_shots:
                break
            shot = c["shot"]
            L = float(shot.duration_sec or 0.0)
            if L <= 0:
                continue
            remaining = budget - acc
            target = remaining if chosen else budget
            if target <= 0 and len(chosen) >= min_shots:
                break

            # 镜头数未达下限时放宽预算窗口：允许用短镜衔接凑够镜头数
            effective_target = max(target, 1.0) if len(chosen) < min_shots else target
            if L / 1.5 <= effective_target * 1.05:
                # 1.5 倍速后能放进预算 → 可选；speed  clamp 在 0.75~1.5
                speed = min(1.5, max(0.75, L / max(effective_target, 0.1)))
                planned = L / speed
                chosen.append({**c, "speed": round(speed, 2), "planned_duration": round(planned, 2)})
                acc += planned
            # else: 1.5 倍速仍超出预算 → 放弃该候选，继续看下一个

        if not chosen:
            return selection, pool, prev_shot

        # 4) 落状态、出池、L2/L3 池内去重
        for c in chosen:
            shot = c["shot"]
            shot.status = "核心"
            shot.quality_score = c["score"]
            shot.dedup_reason = (
                f"竞争匹配胜出，积分 {c['score']:.2f}（match {c['match']:.2f}，"
                f"{c['speed']}x → {c['planned_duration']}s）"
            )
            shot.script_anchor = {
                "beat": beat.beat_id,
                "act": beat.act,
                "function": "",
                "confidence": round(c["match"], 2),
                "reasoning": c["reasoning"],
                "match_score": round(c["match"], 2),
                "total_score": c["score"],
                "planned_speed": c["speed"],
                "planned_duration": c["planned_duration"],
                "seq": len(selection["core"]),  # 节点内选中顺序（0=主镜头，其后为衔接镜）
            }
            pool.remove(shot)
            prev_shot = shot
            self._dedup_against_selected(shot, pool)
            selection["core"].append(shot.shot_id)
            selection["chosen"].append(c)

        return selection, pool, prev_shot

    # ------------------------------------------------------------------
    # 选中镜头 vs 候选池 的 L2/L3 去重
    # ------------------------------------------------------------------
    def _dedup_against_selected(self, selected: Shot, pool: List[Shot]) -> None:
        """把候选池中与已选镜头画面重复（且非强关联）的镜头标记废弃。"""
        if not pool:
            return
        do_l2 = self.processing.get("enable_l2_visual", True)
        do_l3 = self.processing.get("enable_l3_semantic", True) and self.clip_model

        if not (do_l2 or do_l3):
            return

        # L2: pHash
        if do_l2:
            try:
                import imagehash
                from PIL import Image
                sel_hash = None
                if selected.keyframes:
                    try:
                        img = Image.open(selected.keyframes[len(selected.keyframes) // 2])
                        sel_hash = str(imagehash.phash(img))
                    except Exception:
                        sel_hash = None
                if sel_hash:
                    threshold = self.processing.get("phash_threshold", 10)
                    for shot in pool:
                        if shot.status == "废弃" or self._are_related(selected, shot):
                            continue
                        if not shot.keyframes:
                            continue
                        try:
                            h = str(imagehash.phash(Image.open(shot.keyframes[len(shot.keyframes) // 2])))
                        except Exception:
                            continue
                        dist = self._hamming_distance(sel_hash, h)
                        if dist <= threshold:
                            shot.status = "废弃"
                            shot.dedup_reason = f"L2视觉重复: 与已选 {selected.shot_id} pHash距离{dist}"
            except ImportError:
                logger.warning("imagehash 或 PIL 未安装，跳过 L2 池内去重")

        # L3: CLIP 语义
        if do_l3:
            try:
                feat_sel = self._get_clip_features(selected.keyframes[0])
                import numpy as np
                sim_threshold = self.processing.get("duplicate_similarity", 0.92)
                for shot in pool:
                    if shot.status == "废弃" or self._are_related(selected, shot):
                        continue
                    if not shot.keyframes:
                        continue
                    try:
                        feat = self._get_clip_features(shot.keyframes[0])
                    except Exception:
                        continue
                    a = feat_sel / (np.linalg.norm(feat_sel) + 1e-8)
                    b = feat / (np.linalg.norm(feat) + 1e-8)
                    sim = float(np.dot(a, b))
                    if sim >= sim_threshold:
                        shot.status = "废弃"
                        shot.dedup_reason = f"L3语义重复: 与已选 {selected.shot_id} 余弦相似{sim:.3f}"
            except Exception as e:
                logger.warning(f"L3 池内去重失败: {e}")

    # ------------------------------------------------------------------
    # 180度规则 / 方向连续性预警
    # ------------------------------------------------------------------
    def _detect_axis_warnings(self, selections: Dict[str, Dict]) -> List[Dict]:
        """检测同一节点已选镜头间的越轴/方向跳变，供人工复核"""
        warnings = []
        for beat_id, sel in selections.items():
            alive = [c["shot"] for c in sel.get("chosen", []) if c["shot"].status == "核心"]
            for i in range(len(alive)):
                for j in range(i + 1, len(alive)):
                    if self._are_directions_conflicting(alive[i], alive[j]):
                        warnings.append({
                            "beat_id": beat_id,
                            "shot_a": alive[i].shot_id,
                            "shot_b": alive[j].shot_id,
                            "direction_a": alive[i].direction,
                            "direction_b": alive[j].direction,
                            "characters": list(set(alive[i].characters or []) & set(alive[j].characters or [])),
                            "note": "方向冲突/越轴风险，请人工确认是刻意情绪越轴还是技术失误",
                        })
        return warnings

    # ------------------------------------------------------------------
    # 缺失情节点检测
    # ------------------------------------------------------------------
    def _detect_missing_beats(self, selections: Dict[str, Dict], script_beats: List[ScriptBeat]) -> List[Dict]:
        """检测哪些情节点没有核心镜头覆盖"""
        missing = []
        for beat in script_beats:
            sel = selections.get(beat.beat_id, {})
            if not sel.get("core"):
                missing.append({
                    "beat_id": beat.beat_id,
                    "act": beat.act,
                    "scene": beat.scene,
                    "severity": "高",
                    "note": "无核心素材覆盖，需补拍或重新匹配",
                })
        return missing

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------
    def _generate_report(self, shots: List[Shot], selections: Dict[str, Dict],
                         missing_beats: List[Dict], axis_warnings: List[Dict]) -> Dict:
        """生成 Phase 3 完整报告"""
        duplicate_groups = []
        for beat_id, sel in selections.items():
            beat = self._script_beats.get(beat_id)
            group_info = {
                "group_id": f"G{len(duplicate_groups)+1:03d}",
                "script_beat": beat_id,
                "core_shot": sel.get("core", []),
                "budget": beat.estimated_duration if beat else 0,
                "shots": []
            }
            for c in sel.get("chosen", []):
                shot = c["shot"]
                group_info["shots"].append({
                    "shot_id": shot.shot_id,
                    "source_file": shot.source_file,
                    "duration": shot.duration_sec,
                    "state": shot.state,
                    "status": shot.status,
                    "quality_score": shot.quality_score,
                    "match_score": c.get("match"),
                    "planned_speed": c.get("speed"),
                    "planned_duration": c.get("planned_duration"),
                    "reason": shot.dedup_reason,
                })
            duplicate_groups.append(group_info)

        unmatched = [
            {
                "shot_id": s.shot_id,
                "source_file": s.source_file,
                "reason": s.dedup_reason or "未被任何节点选中",
            }
            for s in shots if s.status in ["备选", "未匹配"]
        ]

        report = {
            "total_shots": len(shots),
            "selected_shots": len([s for s in shots if s.status in ["核心", "保留", "备选"]]),
            "core_shots": len([s for s in shots if s.status == "核心"]),
            "alternate_shots": len([s for s in shots if s.status == "备选"]),
            "protected_shots": len([s for s in shots if s.status == "强制保留"]),
            "needs_review_shots": len([s for s in shots if s.status == "待复核"]),
            "discarded_shots": len([s for s in shots if s.status == "废弃"]),
            "unmatched_shots": len(unmatched),
            "missing_beats": missing_beats,
            "duplicate_groups": duplicate_groups,
            "unmatched": unmatched,
            "axis_warnings": axis_warnings,
        }
        return report

    def _empty_report(self) -> Dict:
        return {
            "total_shots": 0,
            "selected_shots": 0,
            "core_shots": 0,
            "alternate_shots": 0,
            "protected_shots": 0,
            "needs_review_shots": 0,
            "discarded_shots": 0,
            "unmatched_shots": 0,
            "missing_beats": [],
            "duplicate_groups": [],
            "unmatched": [],
            "axis_warnings": [],
        }

    def _export_csv(self, shots: List[Shot], report: Dict):
        """导出 CSV 审核表"""
        import csv
        csv_path = os.path.join(self.output_dir, "phase2_deduplication.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "镜头ID", "源文件", "素材状态", "选择状态", "情节点",
                "功能", "匹配置信度", "质量分", "去重原因"
            ])
            for shot in sorted(shots, key=lambda s: s.shot_id):
                anchor = shot.script_anchor or {}
                writer.writerow([
                    shot.shot_id,
                    shot.source_file,
                    shot.state,
                    shot.status,
                    anchor.get("beat", "UNMATCHED"),
                    anchor.get("function", ""),
                    anchor.get("confidence", 0),
                    shot.quality_score,
                    shot.dedup_reason,
                ])
        logger.info(f"Phase 2 CSV 审核表已导出: {csv_path}")
