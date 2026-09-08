"""Phase 3: 逐节点竞争式镜头匹配 / 去重 / 时长变速控制 单元测试"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from src.models import Shot, ScriptBeat, Relationship, Relationships
from src.phase2_dedup import Phase2TakeSelector


class MockLLMService:
    """模拟 LLMService.rank_candidates_for_beat：按 shot._test_beat 返回 match_score"""

    def rank_candidates_for_beat(self, beat, candidates):
        result = {}
        for shot in candidates:
            if getattr(shot, "_test_beat", "") == beat.beat_id:
                result[shot.shot_id] = {
                    "match_score": getattr(shot, "_test_match", 0.8),
                    "reasoning": "mock 匹配",
                }
            else:
                result[shot.shot_id] = {"match_score": 0.0, "reasoning": ""}
        return result


class ErrorLLMService:
    """模拟 LLM 精排失败"""

    def rank_candidates_for_beat(self, beat, candidates):
        raise RuntimeError("mock llm error")


class TestPhase2TakeSelector(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase3_test_")
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
              source_file="test.mp4", asr_text="", prev=None, next=None, match=None):
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
        if match is not None:
            s._test_match = match
        if prev:
            s.relationships.prev = Relationship(**prev)
        if next:
            s.relationships.next = Relationship(**next)
        return s

    @staticmethod
    def _beat(beat_id, scene="场1", duration=5.0, required_shots=1):
        return ScriptBeat(
            act="第一幕",
            scene=scene,
            beat_id=beat_id,
            location="咖啡馆",
            time="傍晚",
            content="测试情节点",
            emotion="焦虑",
            estimated_duration=duration,
            required_shots_count=required_shots,
        )

    def test_raw_candidates_select_core_and_alternate(self):
        """同节点竞争：积分最高者入选核心，其余留候选池为备选"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", visual_quality=3.0, stability=3.0),
            self._shot("S002", "场1-A", visual_quality=5.0, stability=5.0),
        ]
        beats = [self._beat("场1-A", duration=5.0)]

        result, report = selector.run(shots, beats)

        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "备选")
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(report["core_shots"], 1)
        self.assertEqual(report["alternate_shots"], 1)
        self.assertEqual(report["discarded_shots"], 0)

    def test_duration_speed_window_skips_too_long_shot(self):
        """1.5 倍速仍超出预算 → 放弃高积分长镜头，选次优的可变速镜头"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", duration=10.0, visual_quality=5.0, match=0.9),
            self._shot("S002", "场1-A", duration=5.0, visual_quality=3.0, match=0.8),
        ]
        beats = [self._beat("场1-A", duration=4.0)]

        result, _ = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "备选")
        self.assertEqual(statuses["S002"], "核心")
        # 5s 镜头配 4s 预算：speed = clamp(5/4) = 1.25
        anchor = result[1].script_anchor
        self.assertEqual(anchor["planned_speed"], 1.25)
        self.assertEqual(anchor["beat"], "场1-A")

    def test_duration_shortfall_appends_shot(self):
        """0.75 倍速仍不足 → 以首个镜头为基础继续追加衔接镜头，直到满足预算"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", duration=5.0),
            self._shot("S002", "场1-A", duration=5.0),
        ]
        beats = [self._beat("场1-A", duration=10.0)]

        result, report = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "核心")
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(report["core_shots"], 2)
        # 首镜 0.75x → 6.67s，次镜补剩余 3.33s → 1.5x
        speeds = {s.shot_id: s.script_anchor["planned_speed"] for s in result}
        self.assertEqual(speeds["S001"], 0.75)
        self.assertEqual(speeds["S002"], 1.5)

    def test_min_shots_count_respected(self):
        """达到预算但镜头数不足 required_shots_count 时继续补镜"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "场1-A", duration=5.0, visual_quality=5.0),
            self._shot("S002", "场1-A", duration=1.0, visual_quality=3.0),
            self._shot("S003", "场1-A", duration=1.0, visual_quality=2.0),
        ]
        beats = [self._beat("场1-A", duration=5.0, required_shots=3)]

        result, report = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "核心")
        self.assertEqual(statuses["S002"], "核心")
        # S001(5s,1x) 后 acc=5 ≥ 4.5，但 len=1 < 3 → 继续补 S002(1s, 0.75x→1.33s)
        # len=2 < 3 → 继续补 S003(1s, 0.75x→1.33s)
        self.assertEqual(statuses["S003"], "核心")
        self.assertEqual(report["core_shots"], 3)

    def test_unmatched_shots_become_alternates(self):
        """没有匹配到任何节点的镜头留在候选池，最终标记为备选"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        shots = [
            self._shot("S001", "UNMATCHED"),
        ]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        self.assertEqual(result[0].status, "备选")
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

    def test_beats_compete_for_pool_in_order(self):
        """按剧情顺序逐节点消费候选池：先处理的节点优先选走高匹配镜头"""
        class OrderedMockLLM(MockLLMService):
            calls = []
            def rank_candidates_for_beat(self, beat, candidates):
                self.calls.append(beat.beat_id)
                return super().rank_candidates_for_beat(beat, candidates)

        mock = OrderedMockLLM()
        selector = Phase2TakeSelector(self.config, llm_service=mock)
        shots = [
            self._shot("S001", "场1-A", duration=5.0),
            self._shot("S002", "场1-B", duration=5.0),
        ]
        beats = [self._beat("场1-A"), self._beat("场1-B")]

        result, _ = selector.run(shots, beats)
        self.assertEqual(mock.calls, ["场1-A", "场1-B"])
        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S001"], "核心")
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(result[0].script_anchor["beat"], "场1-A")
        self.assertEqual(result[1].script_anchor["beat"], "场1-B")

    def test_same_source_adjacent_shots_are_related(self):
        """来自同一源文件且时间相邻、内容连续（continuity 高）的镜头应被关系图保护"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        s1 = self._shot("S001", "场1-A", source_file="a.mp4")
        s2 = self._shot("S002", "场1-A", source_file="a.mp4")
        # 时间相邻：s2 紧接 s1 之后
        s1.tc_in, s1.tc_out = "00:00:00:00", "00:00:05:00"
        s2.tc_in, s2.tc_out = "00:00:05:00", "00:00:08:00"
        s1.continuity_score = 0.8
        s2.continuity_score = 0.8
        self.assertTrue(selector._are_related(s1, s2))

    def test_empty_input_returns_empty_report(self):
        """输入为空时返回空报告而不崩溃"""
        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        result, report = selector.run([], [self._beat("场1-A")])
        self.assertEqual(result, [])
        self.assertEqual(report["total_shots"], 0)

    def test_llm_rank_is_called_per_beat(self):
        """Phase 3 应按节点逐个调用 LLM 精排"""
        mock_llm = MockLLMService()
        mock_llm.rank_candidates_for_beat = Mock(wraps=mock_llm.rank_candidates_for_beat)
        selector = Phase2TakeSelector(self.config, llm_service=mock_llm)
        shots = [self._shot("S001", "场1-A")]
        beats = [self._beat("场1-A")]

        selector.run(shots, beats)

        self.assertEqual(mock_llm.rank_candidates_for_beat.call_count, 1)
        self.assertEqual(shots[0].script_anchor["beat"], "场1-A")
        self.assertEqual(shots[0].script_anchor["confidence"], 0.8)

    def test_llm_rank_failure_falls_back_to_local(self):
        """LLM 精排失败时，本地积分兜底选镜，节点不为空"""
        selector = Phase2TakeSelector(self.config, llm_service=ErrorLLMService())
        shots = [self._shot("S001", "场1-A")]
        beats = [self._beat("场1-A")]

        result, report = selector.run(shots, beats)

        self.assertEqual(result[0].status, "核心")
        self.assertEqual(len(report["missing_beats"]), 0)

    def test_match_score_affects_selection(self):
        """LLM 精排的 match_score 应影响竞争结果"""
        class RankMockLLMService:
            def rank_candidates_for_beat(self, beat, candidates):
                return {
                    "S001": {"match_score": 0.5, "reasoning": ""},
                    "S002": {"match_score": 0.95, "reasoning": ""},
                }

        config = dict(self.config)
        config["quality_scoring"] = {"weights": {"script_match": 0.60}}

        selector = Phase2TakeSelector(config, llm_service=RankMockLLMService())
        shots = [
            self._shot("S001", "场1-A", visual_quality=5.0, stability=5.0),
            self._shot("S002", "场1-A", visual_quality=3.0, stability=3.0),
        ]
        beats = [self._beat("场1-A", duration=5.0)]

        result, _ = selector.run(shots, beats)
        statuses = {s.shot_id: s.status for s in result}
        self.assertEqual(statuses["S002"], "核心")
        self.assertEqual(statuses["S001"], "备选")

    def test_l1_file_dedup_discards_duplicate_raw_files(self):
        """L1 文件级 MD5 去重：相同文件保留基础分最高的，其余废弃"""
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
        """同节点已选镜头方向相反时应标记越轴预警，且不应判重"""
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
        beats = [self._beat("场1-A", duration=10.0)]

        selector = Phase2TakeSelector(self.config, llm_service=MockLLMService())
        result, report = selector.run(shots, beats)

        # 方向冲突的两个镜头都能入选（10s 预算，各 5s）
        self.assertEqual(result[0].status, "核心")
        self.assertEqual(result[1].status, "核心")
        self.assertEqual(len(report["axis_warnings"]), 1)
        warning = report["axis_warnings"][0]
        self.assertEqual(warning["characters"], ["男主"])
        self.assertIn("越轴", warning["note"])


if __name__ == "__main__":
    unittest.main()
