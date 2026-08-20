"""Phase 2: 镜头选择 / 去重 / 状态决策 单元测试"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from src.models import Shot, ScriptBeat, Relationship, Relationships
from src.phase2_dedup import Phase2TakeSelector


class MockLLMService:
    """模拟 LLMService.anchor_shots_to_script，根据 shot._test_beat 返回锚定结果"""

    def anchor_shots_to_script(self, shots, script_beats):
        result = {}
        for shot in shots:
            beat = getattr(shot, "_test_beat", "UNMATCHED")
            if beat and beat != "UNMATCHED":
                result[shot.shot_id] = {
                    "beat": beat,
                    "act": "第一幕",
                    "function": "叙事",
                    "confidence": 0.8,
                    "reasoning": "mock 匹配",
                }
            else:
                result[shot.shot_id] = {
                    "beat": "UNMATCHED",
                    "act": "",
                    "function": "",
                    "confidence": 0.0,
                    "reasoning": "",
                }
        return result


class ErrorLLMService:
    """模拟 LLM 锚定失败"""

    def anchor_shots_to_script(self, shots, script_beats):
        raise RuntimeError("mock llm error")


class TestPhase2TakeSelector(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase2_test_")
        self.config = {
            "processing": {
                "enable_l2_visual": False,
                "enable_l3_semantic": False,
            },
            "paths": {"output": self.output_dir},
        }

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @staticmethod
    def _shot(shot_id, beat, state="RAW", duration=5.0, visual_quality=3.0, stability=3.0,
              source_file="test.mp4", asr_text="", prev=None, next=None):
        s = Shot(
            shot_id=shot_id,
            state=state,
            source_file=source_file,
            source_path=f"/tmp/{source_file}",
            tc_in="00:00:00:00",
            tc_out="00:00:05:00",
            duration_sec=duration,
            visual_quality=visual_quality,
            stability=stability,
            asr_text=asr_text,
        )
        s._test_beat = beat
        if prev:
            s.relationships.prev = Relationship(**prev)
        if next:
            s.relationships.next = Relationship(**next)
        return s

    @staticmethod
    def _beat(beat_id, scene="场1"):
        return ScriptBeat(
            act="第一幕",
            scene=scene,
            beat_id=beat_id,
            location="咖啡馆",
            time="傍晚",
            content="测试情节点",
            emotion="焦虑",
        )

    def test_raw_candidates_select_core_and_alternate(self):
        """RAW 素材按情节点分组，质量分最高者为核心，其余为备选"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", visual_quality=3.0, stability=3.0),
            self._shot("S002", "场1-A", visual_quality=5.0, stability=5.0),
        ]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "备选")
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(report["core_shots"], 1)
        self.assertEqual(report["alternate_shots"], 1)
        self.assertEqual(report["discarded_shots"], 0)

    def test_processed_material_is_forced_kept(self):
        """PROCESSED 素材强制保留，不参与去重"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", state="PROCESSED", visual_quality=2.0),
            self._shot("S002", "场1-A", state="RAW", visual_quality=5.0),
        ]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "强制保留")
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(report["protected_shots"], 1)

    def test_analyzed_material_can_be_core(self):
        """ANALYZED 素材质量最高时可成为核心 take，但仍保留 needs_review 标记"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", state="ANALYZED", visual_quality=5.0),
        ]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        self.assertEqual(result[0].status, "核心")
        self.assertTrue(result[0].needs_review)
        self.assertEqual(report["core_shots"], 1)

    def test_unmatched_shots_are_marked(self):
        """没有剧本锚定的素材标记为未匹配"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "UNMATCHED"),
        ]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        self.assertEqual(result[0].status, "未匹配")
        self.assertEqual(report["unmatched_shots"], 1)
        self.assertEqual(report["missing_beats"][0]["beat_id"], "场1-A")

    def test_missing_beats_detected(self):
        """有情节点没有核心素材覆盖时应被检测出来"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A"),
        ]
        beats = [self._beat("场1-A"), self._beat("场1-B")]

        _, report = selector.run(shots, beats)

        missing_ids = [m["beat_id"] for m in report["missing_beats"]]
        self.assertIn("场1-B", missing_ids)
        self.assertNotIn("场1-A", missing_ids)

    def test_relationship_protection_avoids_dedup(self):
        """关系图标记为强连贯的相邻镜头不应被判定为重复"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", visual_quality=5.0),
            self._shot("S002", "场1-A", visual_quality=4.0,
                       prev={"shot_id": "S001", "relationship_type": "情绪延续", "coherence_score": 0.85}),
        ]
        beats = [self._beat("场1-A")]

        # 打开 L2 去重，但两个候选因关系保护不应互相废弃
        config = dict(self.config)
        config["processing"] = {"enable_l2_visual": True, "enable_l3_semantic": False}
        selector = Phase2TakeSelector(config, llm_service=MockLLMService())

        result, _ = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        # 因关系保护，即使两个素材很相似，也不应被 L2 废弃
        self.assertNotEqual(statuses.get("S001"), "废弃")
        self.assertNotEqual(statuses.get("S002"), "废弃")

    def test_same_source_adjacent_shots_are_related(self):
        """来自同一源文件且时间相邻的镜头应被关系图保护"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        s1 = self._shot("S001", "场1-A", source_file="a.mp4")
        s2 = self._shot("S002", "场1-A", source_file="a.mp4")
        self.assertTrue(selector._are_related(s1, s2))

    def test_empty_input_returns_empty_report(self):
        """输入为空时返回空报告而不崩溃"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        result, report = selector.run([], [self._beat("场1-A")])
        self.assertEqual(result, [])
        self.assertEqual(report["total_shots"], 0)

    def test_llm_anchor_is_called(self):
        """Phase 2 应调用 LLM 锚定服务"""
        mock_llm = MockLLMService()
        mock_llm.anchor_shots_to_script = Mock(wraps=mock_llm.anchor_shots_to_script)
        selector = Phase2TakeSelector(self.config, llm_service=mock_llm)
        shots = [self._shot("S001", "场1-A")]
        beats = [self._beat("场1-A")]

        selector.run(shots, beats)

        mock_llm.anchor_shots_to_script.assert_called_once_with(shots, beats)
        self.assertEqual(shots[0].script_anchor["beat"], "场1-A")
        self.assertEqual(shots[0].script_anchor["confidence"], 0.8)

    def test_llm_anchor_failure_fallback(self):
        """LLM 锚定失败时，所有镜头应 fallback 为未匹配/ERROR"""
        selector = Phase2TakeSelector(self.config, llm_service=ErrorLLMService())
        shots = [self._shot("S001", "场1-A")]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        self.assertEqual(result[0].status, "未匹配")
        self.assertEqual(report["unmatched_shots"], 1)

    def test_script_confidence_affects_core_selection(self):
        """剧本匹配置信度应影响质量分并改变核心 take 选择"""
        class ConfidenceMockLLMService:
            def anchor_shots_to_script(self, shots, script_beats):
                return {
                    "S001": {"beat": "场1-A", "act": "第一幕", "function": "叙事", "confidence": 0.5, "reasoning": ""},
                    "S002": {"beat": "场1-A", "act": "第一幕", "function": "叙事", "confidence": 0.95, "reasoning": ""},
                }

        config = dict(self.config)
        # 提高 script_confidence 权重，让匹配度决定核心 take
        config["quality_scoring"] = {
            "weights": {
                "visual_quality": 0.10,
                "stability": 0.10,
                "script_confidence": 0.60,
                "dialogue": 0.05,
                "duration": 0.10,
                "metadata_complete": 0.05,
            }
        }

        selector = Phase2TakeSelector(config, llm_service=ConfidenceMockLLMService())
        # S001 视觉质量高但匹配置信低；S002 视觉质量低但匹配置信高
        shots = [
            self._shot("S001", "场1-A", visual_quality=5.0, stability=5.0),
            self._shot("S002", "场1-A", visual_quality=3.0, stability=3.0),
        ]
        beats = [self._beat("场1-A")]

        result, _ = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        # 在 script_confidence 权重下，S002 应被选为核心
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(statuses["S001"], "备选")


    def test_l1_file_dedup_discards_duplicate_raw_files(self):
        """L1 文件级 MD5 去重：相同文件保留质量分最高的，其余废弃"""
        with tempfile.TemporaryDirectory() as d:
            path1 = os.path.join(d, "a.mp4")
            path2 = os.path.join(d, "b.mp4")
            with open(path1, "w") as f:
                f.write("same content")
            with open(path2, "w") as f:
                f.write("same content")

            shots = [
                self._shot("S001", "场1-A", state="RAW", visual_quality=3.0,
                           source_file="a.mp4"),
                self._shot("S002", "场1-A", state="RAW", visual_quality=5.0,
                           source_file="b.mp4"),
            ]
            shots[0].source_path = path1
            shots[1].source_path = path2
            beats = [self._beat("场1-A")]

            selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
            result, report = selector.run(shots, beats)
            statuses = {s.shot_id: s.status for s in result}
            self.assertEqual(statuses["S002"], "核心")
            self.assertEqual(statuses["S001"], "废弃")
            self.assertIn("L1文件级重复", result[0].dedup_reason)

    def test_axis_warning_for_opposite_directions(self):
        """同角色同组内方向相反时应标记越轴预警，且不应判重"""
        shots = [
            self._shot("S001", "场1-A", source_file="s1.mp4"),
            self._shot("S002", "场1-A", source_file="s2.mp4"),
        ]
        shots[0].source_path = "/tmp/s1.mp4"
        shots[1].source_path = "/tmp/s2.mp4"
        shots[0].characters = ["男主"]
        shots[0].direction = "从左向右"
        shots[1].characters = ["男主"]
        shots[1].direction = "从右向左"
        beats = [self._beat("场1-A")]

        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        result, report = selector.run(shots, beats)

        # 方向冲突的两个镜头都不应被废弃
        self.assertNotEqual(result[0].status, "废弃")
        self.assertNotEqual(result[1].status, "废弃")
        # 报告中应包含越轴预警
        self.assertEqual(len(report["axis_warnings"]), 1)
        warning = report["axis_warnings"][0]
        self.assertEqual(warning["characters"], ["男主"])
        self.assertIn("越轴", warning["note"])


if __name__ == "__main__":
    unittest.main()
