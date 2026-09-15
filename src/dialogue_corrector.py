"""
台词修正器（Phase 3 启动时的独立可选环节）

背景：ASR（SenseVoice）转录的台词常有同音/近音错别字（如"云琛"→"云深"），
直接影响 Phase 3 的台词 bigram 匹配积分。本环节用 Phase 2 剧情节点的关键台词做参照，
调用 LLM 对镜头台词做"只改字、不改语义"的校对归正。

流程（幂等，可反复执行）：
1. 取所有含关键台词的节点（key_dialogue 优先，其次 dialogue_entries 拼接）
2. 对每个有台词的镜头，用 bigram 覆盖率（复用 QualityScorer._dialogue_match_score）
   找出最匹配的节点台词作为参照
3. LLM 校对：参照节点台词修正 ASR 错字（同音/近音归正，禁止改写语义、增删内容）；
   解析失败或返回空则保留原文
4. 回写三处产物：
   - phase1_split_clips/Sxxx_config.json（shot.dialogue / shot.asr_text / 顶层 dialogue）
   - phase0_rough_clips/Sxxx_frames/audio_profile.json（text + transcript 分段 +
     dialogue_corrected 标记）
   - phase1_analysis.json（总表同步更新）

已带 dialogue_corrected 标记的镜头直接跳过（断点续跑）。
开关：processing.dialogue_correction（默认 true）；LLM 不可用时整体跳过并告警。
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple

from src.models import Shot, ScriptBeat
from src.quality_scorer import QualityScorer
from src.utils import logger, load_json, save_json


class DialogueCorrector:
    """台词错字校对：节点关键台词为参照，LLM 归正 ASR 台词"""

    def __init__(self, config: Dict):
        self.config = config
        self.processing = config.get("processing", {})
        self.enabled = bool(self.processing.get("dialogue_correction", True))
        self.output_dir = config.get("paths", {}).get("output", "./output")
        self.min_match = float(self.processing.get("dialogue_correction_min_match", 0.2))
        self._scorer = QualityScorer(config)
        self._llm = None
        self._llm_failed = False

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run(self, shots: List[Shot], beats: List[ScriptBeat]) -> Dict:
        """执行台词修正，返回统计 {total, corrected, skipped, failed}"""
        stats = {"total": 0, "corrected": 0, "skipped": 0, "failed": 0}
        if not self.enabled:
            logger.info("[台词修正] 未启用（processing.dialogue_correction=false），跳过")
            return stats

        beat_dialogues = self._collect_beat_dialogues(beats)
        if not beat_dialogues:
            logger.info("[台词修正] 无含关键台词的节点，跳过")
            return stats

        any_change = False
        for shot in shots:
            src_text = (getattr(shot, "asr_text", "") or shot.dialogue or "").strip()
            if not src_text:
                continue
            if self._already_corrected(shot.shot_id):
                stats["skipped"] += 1
                continue
            stats["total"] += 1

            beat_id, ref = self._best_match_beat(shot, beat_dialogues)
            if not ref:
                stats["skipped"] += 1
                continue

            corrected = self._llm_correct(src_text, ref)
            if not corrected or corrected == src_text:
                stats["skipped"] += 1
                continue

            # 基本 sanity：修正后不应大幅偏离原文（防 LLM 擅自改写）
            if abs(len(corrected) - len(src_text)) > max(6, len(src_text) // 2):
                logger.warning(
                    f"[台词修正] {shot.shot_id} 修正结果长度异常，保留原文: "
                    f"{src_text[:20]}... -> {corrected[:20]}..."
                )
                stats["failed"] += 1
                continue

            self._apply(shot, corrected)
            self._write_back(shot, corrected)
            stats["corrected"] += 1
            any_change = True
            logger.info(
                f"[台词修正] {shot.shot_id} (节点{beat_id}): "
                f"\"{src_text[:24]}\" -> \"{corrected[:24]}\""
            )

        if any_change:
            self._rewrite_phase1_analysis(shots)
            logger.info(f"[台词修正] 完成: 修正 {stats['corrected']}/{stats['total']} 个镜头台词")
        else:
            logger.info(f"[台词修正] 完成: 无需修正（检查 {stats['total']} 个镜头）")
        return stats

    # ------------------------------------------------------------------
    # 节点台词收集与匹配
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_dialogue(text: str) -> str:
        """剥掉首尾引号后判空（无台词节点可能存字面量 '\"\"'）"""
        t = (text or "").strip()
        if t in {'""', "''", "「」", "『』"}:
            return ""
        return t.strip("\"'“”‘’「」『』").strip()

    def _collect_beat_dialogues(self, beats: List[ScriptBeat]) -> Dict[str, str]:
        """beat_id -> 节点关键台词（仅含有台词的节点）"""
        out: Dict[str, str] = {}
        for b in beats:
            dlg = self._clean_dialogue(getattr(b, "key_dialogue", "") or "")
            if not dlg and getattr(b, "dialogue_entries", None):
                parts = [
                    self._clean_dialogue(getattr(e, "text", "") or getattr(e, "dialogue", "") or "")
                    for e in b.dialogue_entries
                ]
                dlg = " ".join(p for p in parts if p).strip()
            if dlg:
                out[b.beat_id] = dlg
        return out

    def _best_match_beat(
        self, shot: Shot, beat_dialogues: Dict[str, str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """bigram 覆盖率最高的节点台词；最高覆盖率低于 min_match 视为匹配不上"""
        best_id, best_score, best_text = None, 0.0, None
        for beat_id, dlg in beat_dialogues.items():
            try:
                s = self._scorer._dialogue_match_score(dlg, shot)
            except Exception:
                continue
            if s > best_score:
                best_id, best_score, best_text = beat_id, s, dlg
        if best_score < self.min_match or best_text is None:
            return None, None
        return best_id, best_text

    # ------------------------------------------------------------------
    # LLM 校对
    # ------------------------------------------------------------------
    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self._llm_failed:
            return None
        try:
            from src.services.llm_service import LLMService
            self._llm = LLMService(self.config)
            return self._llm
        except Exception as e:
            logger.warning(f"[台词修正] LLM 不可用，跳过台词修正: {e}")
            self._llm_failed = True
            return None

    def _llm_correct(self, asr_text: str, ref_dialogue: str) -> Optional[str]:
        """LLM 同音/近音错字归正；失败返回 None（保留原文）"""
        llm = self._get_llm()
        if llm is None:
            return None
        prompt = (
            "你是中文台词校对专家。下面给你两段文字：\n"
            f"【剧情节点原台词】{ref_dialogue}\n"
            f"【视频语音识别结果】{asr_text}\n\n"
            "语音识别结果可能有同音字、近音字错误。请参照剧情节点原台词的用字习惯"
            "（特别是人名、地名、功法名等专有名词），对语音识别结果做校对归正。\n"
            "规则：\n"
            "1. 只修正错别字，禁止改写语义、增删内容、调整语序；\n"
            "2. 字数尽量保持不变；\n"
            "3. 语音识别结果若与剧情无关，不要硬套原台词；\n"
            "4. 只输出修正后的台词文本本身，不要输出任何解释、引号或 JSON。\n"
            "修正后台词："
        )
        try:
            text = (llm.generate(prompt) or "").strip()
        except Exception as e:
            logger.warning(f"[台词修正] LLM 调用失败: {e}")
            return None
        # 防御性清理：去引号/代码块
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        text = text.strip("\"'“”").strip()
        if not text or len(text) < 2:
            return None
        return text

    # ------------------------------------------------------------------
    # 回写
    # ------------------------------------------------------------------
    def _already_corrected(self, shot_id: str) -> bool:
        """幂等标记：config.json 或 audio_profile.json 带 dialogue_corrected 即跳过"""
        cfg_path = os.path.join(self.output_dir, "phase1_split_clips", f"{shot_id}_config.json")
        if os.path.exists(cfg_path):
            try:
                data = load_json(cfg_path)
                if data.get("dialogue_corrected"):
                    return True
            except Exception:
                pass
        profile_path = self._audio_profile_path(shot_id)
        if profile_path and os.path.exists(profile_path):
            try:
                if load_json(profile_path).get("dialogue_corrected"):
                    return True
            except Exception:
                pass
        return False

    def _audio_profile_path(self, shot_id: str) -> Optional[str]:
        frames_dir = os.path.join(self.output_dir, "phase0_rough_clips", f"{shot_id}_frames")
        p = os.path.join(frames_dir, "audio_profile.json")
        return p if os.path.exists(p) else None

    @staticmethod
    def _apply(shot: Shot, corrected: str):
        shot.dialogue = corrected
        shot.asr_text = corrected

    def _write_back(self, shot: Shot, corrected: str):
        """回写镜头 config 与音频档案（尽力而为，单处失败不阻断）"""
        # 1) phase1_split_clips/Sxxx_config.json
        cfg_path = os.path.join(self.output_dir, "phase1_split_clips", f"{shot.shot_id}_config.json")
        if os.path.exists(cfg_path):
            try:
                data = load_json(cfg_path)
                data["dialogue"] = corrected
                shot_data = data.get("shot")
                if isinstance(shot_data, dict):
                    shot_data["dialogue"] = corrected
                    shot_data["asr_text"] = corrected
                data["dialogue_corrected"] = True
                save_json(data, cfg_path)
            except Exception as e:
                logger.warning(f"[台词修正] 回写镜头配置失败 {shot.shot_id}: {e}")

        # 2) phase0_rough_clips/Sxxx_frames/audio_profile.json（text + 分段）
        profile_path = self._audio_profile_path(shot.shot_id)
        if profile_path:
            try:
                profile = load_json(profile_path)
                transcript = profile.get("transcript") or []
                if transcript:
                    profile["transcript"] = self._align_segments(transcript, corrected)
                    profile["text"] = corrected
                else:
                    profile["text"] = corrected
                profile["dialogue_corrected"] = True
                save_json(profile, profile_path)
            except Exception as e:
                logger.warning(f"[台词修正] 回写音频档案失败 {shot.shot_id}: {e}")

    @staticmethod
    def _align_segments(transcript: List[Dict], corrected: str) -> List[Dict]:
        """把修正后的全文按原分段长度比例映射回各段（保留时间戳）。

        纯按字数比例切片；若原文只有单段则整段替换。
        """
        total = sum(len(str(t.get("text", ""))) for t in transcript)
        if total <= 0:
            return transcript
        if len(transcript) == 1:
            out = [dict(transcript[0])]
            out[0]["text"] = corrected
            return out
        out = []
        pos = 0
        n = len(corrected)
        for i, seg in enumerate(transcript):
            seg_len = len(str(seg.get("text", "")))
            if i == len(transcript) - 1:
                piece = corrected[pos:]
            else:
                take = round(n * seg_len / total)
                piece = corrected[pos:pos + take]
                pos += take
            new_seg = dict(seg)
            new_seg["text"] = piece
            out.append(new_seg)
        return out

    def _rewrite_phase1_analysis(self, shots: List[Shot]):
        """同步更新 phase1_analysis.json 总表"""
        path = os.path.join(self.output_dir, "phase1_analysis.json")
        if not os.path.exists(path):
            return
        try:
            data = load_json(path)
            data["shots"] = [s.to_dict() for s in shots]
            data["total_shots"] = len(shots)
            save_json(data, path)
        except Exception as e:
            logger.warning(f"[台词修正] 更新 phase1_analysis.json 失败: {e}")
