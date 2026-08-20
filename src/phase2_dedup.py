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
from typing import List, Dict, Tuple, Set
from collections import defaultdict

from src.models import Shot, ScriptBeat
from src.quality_scorer import QualityScorer
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

    def run(self, shots: List[Shot], script_beats: List[ScriptBeat]) -> Tuple[List[Shot], Dict]:
        """执行 Phase 2 镜头选择"""
        logger.info("=" * 60)
        logger.info("Phase 2: 镜头选择 + 去重 + 覆盖检测")
        logger.info("=" * 60)

        self._script_beats = {b.beat_id: b for b in script_beats}

        if not shots:
            logger.warning("Phase 2 输入为空")
            return [], self._empty_report()

        # 0. 用 DeepSeek 做剧本-镜头语义锚定（Phase 1 不再负责剧本锚定）
        logger.info(f"Phase 2: 调用 LLM 对 {len(shots)} 个镜头进行剧本锚定...")
        try:
            anchor_map = self.llm_service.anchor_shots_to_script(shots, script_beats)
        except Exception as e:
            logger.error(f"Phase 2 LLM 锚定调用失败: {e}")
            anchor_map = {}

        for shot in shots:
            anchor = anchor_map.get(shot.shot_id)
            if anchor:
                shot.script_anchor = anchor
            else:
                shot.script_anchor = {
                    "beat": "UNMATCHED",
                    "act": "",
                    "function": "",
                    "confidence": 0.0,
                    "reasoning": "LLM 未返回该镜头锚定结果",
                }
        matched = sum(1 for s in shots if s.script_anchor.get("beat") not in ["UNMATCHED", "ERROR"])
        logger.info(f"剧本锚定完成: {matched}/{len(shots)} 个镜头匹配到情节点")

        # 1. 计算质量分
        self._compute_quality_scores(shots)

        # 2. 基于状态做初始标记
        self._initial_state_mark(shots)

        # 3. L1 文件级去重（MD5）
        self._l1_file_dedup(shots)

        # 4. 按情节点分组
        beat_groups = self._group_by_beat(shots)

        # 5. 组内去重（受关系图 + 方向保护）
        for beat_id, group in beat_groups.items():
            beat_groups[beat_id] = self._dedup_within_group(group)

        # 6. 核心 take 选择
        beat_selections = self._select_takes(beat_groups)

        # 7. 缺失情节点检测
        missing_beats = self._detect_missing_beats(beat_groups, script_beats)

        # 8. 180度/方向连续性预警
        axis_warnings = self._detect_axis_warnings(beat_groups)

        # 9. 生成报告
        report = self._generate_report(shots, beat_groups, beat_selections, missing_beats, axis_warnings)

        # 8. 保存结果
        save_json(report, os.path.join(self.output_dir, "phase2_deduplication.json"))
        self._export_csv(shots, report)

        selected = [s for s in shots if s.status in ["核心", "保留", "备选", "强制保留", "待复核"]]
        logger.info(f"Phase 2 完成: 选中 {len(selected)}/{len(shots)} 个镜头")
        logger.info(f"  核心: {len([s for s in shots if s.status == '核心'])}")
        logger.info(f"  备选: {len([s for s in shots if s.status == '备选'])}")
        logger.info(f"  强制保留: {len([s for s in shots if s.status == '强制保留'])}")
        logger.info(f"  待复核: {len([s for s in shots if s.status == '待复核'])}")
        logger.info(f"  废弃: {len([s for s in shots if s.status == '废弃'])}")
        logger.info(f"  未匹配: {len([s for s in shots if s.status == '未匹配'])}")
        logger.info(f"  缺失情节点: {len(missing_beats)}")

        return shots, report

    # ------------------------------------------------------------------
    # 初始评分与状态标记
    # ------------------------------------------------------------------
    def _compute_quality_scores(self, shots: List[Shot]):
        """为每个 Shot 计算质量分"""
        logger.info("计算 Shot 质量分...")
        for shot in shots:
            shot.quality_score = self.quality_scorer.score(shot)

    def _initial_state_mark(self, shots: List[Shot]):
        """基于素材状态做初始标记"""
        for shot in shots:
            # 无剧本锚定或锚定失败
            if not shot.script_anchor or not shot.script_anchor.get("beat") or shot.script_anchor.get("beat") in ["UNMATCHED", "ERROR", ""]:
                shot.status = "未匹配"
                shot.dedup_reason = "未匹配到剧本情节点"
                continue

            # 按状态标记保护/待复核
            if shot.state == "PROCESSED":
                shot.status = "强制保留"
                shot.dedup_reason = "PROCESSED 素材强制保留"
                shot.needs_review = False
            elif shot.state == "ANALYZED":
                shot.status = "待复核"
                shot.dedup_reason = "ANALYZED 素材待复核"
                shot.needs_review = True
            elif shot.state == "RAW":
                shot.status = "候选"
                shot.dedup_reason = ""
                shot.needs_review = False
            else:
                shot.status = "未匹配"
                shot.dedup_reason = f"未知状态: {shot.state}"

    # ------------------------------------------------------------------
    # L1 文件级去重（MD5）
    # ------------------------------------------------------------------
    def _l1_file_dedup(self, shots: List[Shot]) -> None:
        """基于文件 MD5 的完全重复检测。保留质量分最高的副本；
        PROCESSED/ANALYZED 优先于 RAW。"""
        md5_groups: Dict[str, List[Shot]] = defaultdict(list)
        for shot in shots:
            file_path = shot.source_path
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
    def _group_by_beat(self, shots: List[Shot]) -> Dict[str, List[Shot]]:
        """按剧本情节点 beat 分组"""
        groups = defaultdict(list)
        for shot in shots:
            if shot.status == "未匹配":
                continue
            beat = shot.script_anchor.get("beat") if shot.script_anchor else None
            if beat:
                groups[beat].append(shot)

        logger.info(f"情节点分组: {len(groups)} 个组有素材")
        return dict(groups)

    # ------------------------------------------------------------------
    # 组内去重
    # ------------------------------------------------------------------
    def _dedup_within_group(self, group_shots: List[Shot]) -> List[Shot]:
        """对单个情节点组内去重，受关系图保护"""
        # 分离受保护素材和 RAW 候选
        protected = [s for s in group_shots if s.status in ["强制保留", "待复核"]]
        candidates = [s for s in group_shots if s.status == "候选"]

        # L2 视觉去重
        if self.processing.get("enable_l2_visual", True):
            candidates = self._l2_visual_dedup(candidates)

        # L3 语义去重
        if self.processing.get("enable_l3_semantic", True) and self.clip_model:
            candidates = self._l3_semantic_dedup(candidates)

        return protected + candidates

    def _are_related(self, shot1: Shot, shot2: Shot) -> bool:
        """判断两个 Shot 是否强关联（不应判重）"""
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

        # 来自同一源文件且时间相邻（可能是长镜头拆分）
        if shot1.source_file == shot2.source_file and shot1.source_path == shot2.source_path:
            return True

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

    def _l2_visual_dedup(self, candidates: List[Shot]) -> List[Shot]:
        """L2: 基于 pHash 的视觉去重，受关系图保护"""
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            logger.warning("imagehash 或 PIL 未安装，跳过 L2 去重")
            return candidates

        logger.info(f"L2 视觉去重: {len(candidates)} 个候选")

        # 计算 pHash
        shot_hashes = {}
        for shot in candidates:
            if not shot.keyframes:
                continue
            try:
                img = Image.open(shot.keyframes[len(shot.keyframes) // 2])
                shot_hashes[shot.shot_id] = str(imagehash.phash(img))
            except Exception as e:
                logger.warning(f"pHash 计算失败 {shot.shot_id}: {e}")

        threshold = self.processing.get("phash_threshold", 10)
        to_remove = set()
        shot_ids = list(shot_hashes.keys())

        for i in range(len(shot_ids)):
            if shot_ids[i] in to_remove:
                continue
            shot_i = next((s for s in candidates if s.shot_id == shot_ids[i]), None)
            if not shot_i:
                continue

            for j in range(i + 1, len(shot_ids)):
                if shot_ids[j] in to_remove:
                    continue
                shot_j = next((s for s in candidates if s.shot_id == shot_ids[j]), None)
                if not shot_j:
                    continue

                # 关系图保护
                if self._are_related(shot_i, shot_j):
                    continue

                dist = self._hamming_distance(shot_hashes[shot_ids[i]], shot_hashes[shot_ids[j]])
                if dist <= threshold:
                    # 保留质量高的
                    if shot_i.quality_score >= shot_j.quality_score:
                        to_remove.add(shot_ids[j])
                        shot_j.status = "废弃"
                        shot_j.dedup_reason = f"L2视觉重复: 与 {shot_i.shot_id} pHash距离{dist}"
                    else:
                        to_remove.add(shot_ids[i])
                        shot_i.status = "废弃"
                        shot_i.dedup_reason = f"L2视觉重复: 与 {shot_j.shot_id} pHash距离{dist}"
                        break

        logger.info(f"L2 标记废弃: {len(to_remove)} 个")
        return [s for s in candidates if s.shot_id not in to_remove]

    def _l3_semantic_dedup(self, candidates: List[Shot]) -> List[Shot]:
        """L3: 基于 CLIP 的语义去重，受关系图保护"""
        if not self.clip_model:
            return candidates

        import numpy as np

        logger.info(f"L3 语义去重: {len(candidates)} 个候选")

        # 提取特征
        features = []
        valid_shots = []
        for shot in candidates:
            if not shot.keyframes:
                continue
            try:
                feat = self._get_clip_features(shot.keyframes[0])
                features.append(feat)
                valid_shots.append(shot)
            except Exception as e:
                logger.warning(f"CLIP 特征提取失败 {shot.shot_id}: {e}")

        if len(valid_shots) < 2:
            return candidates

        features = np.array(features)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (norms + 1e-8)
        sim_matrix = np.dot(features_norm, features_norm.T)

        similarity_threshold = self.processing.get("duplicate_similarity", 0.92)
        to_remove = set()

        for i in range(len(valid_shots)):
            if valid_shots[i].shot_id in to_remove:
                continue
            for j in range(i + 1, len(valid_shots)):
                if valid_shots[j].shot_id in to_remove:
                    continue

                # 关系图保护
                if self._are_related(valid_shots[i], valid_shots[j]):
                    continue

                sim = sim_matrix[i, j]
                if sim >= similarity_threshold:
                    if valid_shots[i].quality_score >= valid_shots[j].quality_score:
                        to_remove.add(valid_shots[j].shot_id)
                        valid_shots[j].status = "废弃"
                        valid_shots[j].dedup_reason = f"L3语义重复: 与 {valid_shots[i].shot_id} 余弦相似{sim:.3f}"
                    else:
                        to_remove.add(valid_shots[i].shot_id)
                        valid_shots[i].status = "废弃"
                        valid_shots[i].dedup_reason = f"L3语义重复: 与 {valid_shots[j].shot_id} 余弦相似{sim:.3f}"
                        break

        logger.info(f"L3 标记废弃: {len(to_remove)} 个")
        return [s for s in candidates if s.shot_id not in to_remove]

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
    # 核心 take 选择
    # ------------------------------------------------------------------
    def _select_takes(self, beat_groups: Dict[str, List[Shot]]) -> Dict[str, Dict]:
        """为每个情节点选择核心 take：质量分最高者即为核心，不区分 RAW/PROCESSED/ANALYZED"""
        selections = {}

        for beat_id, group in beat_groups.items():
            # 过滤掉已废弃的
            alive = [s for s in group if s.status != "废弃"]
            if not alive:
                selections[beat_id] = {"core": None, "shots": [s.shot_id for s in group]}
                continue

            # 按质量分排序，最高分即为核心
            alive_sorted = sorted(alive, key=lambda s: s.quality_score, reverse=True)
            core = alive_sorted[0]

            # 核心状态统一为「核心」，其余保持原来的强制保留/待复核状态或标记为备选
            core.status = "核心"
            core.dedup_reason = f"该情节点最优 take，质量分 {core.quality_score}"

            for shot in alive_sorted:
                if shot.shot_id == core.shot_id:
                    continue
                if shot.status == "强制保留":
                    shot.dedup_reason = f"PROCESSED 素材保留，质量分 {shot.quality_score}"
                elif shot.status == "待复核":
                    shot.dedup_reason = f"ANALYZED 素材备选，质量分 {shot.quality_score}"
                else:
                    shot.status = "备选"
                    shot.dedup_reason = f"同组备选，质量分 {shot.quality_score}"

            selections[beat_id] = {
                "core": core.shot_id,
                "shots": [s.shot_id for s in alive],
                "script_beat": self._script_beats.get(beat_id, ScriptBeat("", "", beat_id, "", "", "", "")).to_dict(),
            }

        return selections

    # ------------------------------------------------------------------
    # 180度规则 / 方向连续性预警
    # ------------------------------------------------------------------
    def _detect_axis_warnings(self, beat_groups: Dict[str, List[Shot]]) -> List[Dict]:
        """检测同组内可能存在的越轴/方向跳变镜头对，供人工复核"""
        warnings = []
        for beat_id, group in beat_groups.items():
            alive = [s for s in group if s.status != "废弃"]
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
    def _detect_missing_beats(self, beat_groups: Dict[str, List[Shot]], script_beats: List[ScriptBeat]) -> List[Dict]:
        """检测哪些情节点没有被核心 Shot 覆盖"""
        covered_beats = set()
        for beat_id, group in beat_groups.items():
            if any(s.status == "核心" for s in group):
                covered_beats.add(beat_id)

        missing = []
        for beat in script_beats:
            if beat.beat_id not in covered_beats:
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
    def _generate_report(self, shots: List[Shot], beat_groups: Dict[str, List[Shot]],
                         selections: Dict[str, Dict], missing_beats: List[Dict],
                         axis_warnings: List[Dict]) -> Dict:
        """生成 Phase 2 完整报告"""
        duplicate_groups = []
        for beat_id, group in beat_groups.items():
            sel = selections.get(beat_id, {})
            group_info = {
                "group_id": f"G{len(duplicate_groups)+1:03d}",
                "script_beat": beat_id,
                "core_shot": sel.get("core"),
                "shots": []
            }
            for shot in sorted(group, key=lambda s: s.quality_score, reverse=True):
                group_info["shots"].append({
                    "shot_id": shot.shot_id,
                    "source_file": shot.source_file,
                    "duration": shot.duration_sec,
                    "state": shot.state,
                    "status": shot.status,
                    "quality_score": shot.quality_score,
                    "reason": shot.dedup_reason,
                })
            duplicate_groups.append(group_info)

        unmatched = [
            {
                "shot_id": s.shot_id,
                "source_file": s.source_file,
                "reason": s.dedup_reason or "未匹配剧本节点",
            }
            for s in shots if s.status == "未匹配"
        ]

        report = {
            "total_shots": len(shots),
            "selected_shots": len([s for s in shots if s.status in ["核心", "保留", "备选", "强制保留", "待复核"]]),
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
