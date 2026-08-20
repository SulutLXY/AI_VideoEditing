"""切分积分器单元测试"""
import unittest

from src.models import Segment, Boundary
from src.split_scorer import SplitScorer


class TestSplitScorer(unittest.TestCase):
    def get_default_config(self):
        return {
            "weights": {
                "camera_change": 0.35,
                "subject_change": 0.25,
                "emotion_break": 0.15,
                "plot_shift": 0.10,
                "action_break": 0.10,
                "dialogue_break": 0.05,
            },
            "thresholds": {"high": 0.55, "medium": 0.40},
            "long_take_protection": {
                "enabled": True,
                "min_duration": 30.0,
                "threshold_boost": 0.15,
            },
        }

    def test_camera_change_boundary_scores_point_35(self):
        segments = [
            Segment(start=0.0, end=10.0, camera_position="A机位"),
            Segment(start=10.0, end=20.0, camera_position="B机位"),
        ]
        scorer = SplitScorer(self.get_default_config())
        boundaries = scorer.compute_boundaries(segments, [10.0], 20.0)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].confidence, "low")
        self.assertEqual(boundaries[0].score, 0.35)
        self.assertIn("机位变化", boundaries[0].reason)

    def test_camera_and_subject_change_boundary_is_high(self):
        segments = [
            Segment(start=0.0, end=10.0, camera_position="A机位", characters=["男主"]),
            Segment(start=10.0, end=20.0, camera_position="B机位", characters=["女主"]),
        ]
        scorer = SplitScorer(self.get_default_config())
        boundaries = scorer.compute_boundaries(segments, [10.0], 20.0)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].confidence, "high")
        self.assertEqual(boundaries[0].score, 0.60)
        self.assertIn("机位变化", boundaries[0].reason)
        self.assertIn("主体变化", boundaries[0].reason)

    def test_no_change_boundary_is_low(self):
        segments = [
            Segment(start=0.0, end=10.0, camera_position="A机位", emotion="焦虑"),
            Segment(start=10.0, end=20.0, camera_position="A机位", emotion="焦虑"),
        ]
        scorer = SplitScorer(self.get_default_config())
        boundaries = scorer.compute_boundaries(segments, [10.0], 20.0)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].confidence, "low")
        self.assertLess(boundaries[0].score, 0.4)

    def test_emotion_and_subject_change_is_medium(self):
        segments = [
            Segment(start=0.0, end=10.0, characters=["男主"], emotion="焦虑"),
            Segment(start=10.0, end=20.0, characters=["女主"], emotion="紧张"),
        ]
        scorer = SplitScorer(self.get_default_config())
        boundaries = scorer.compute_boundaries(segments, [10.0], 20.0)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].confidence, "medium")

    def test_long_take_protection_reduces_confidence(self):
        segments = [
            Segment(start=0.0, end=40.0, camera_position="A机位", emotion="焦虑"),
            Segment(start=40.0, end=50.0, camera_position="B机位", emotion="焦虑"),
        ]
        scorer = SplitScorer(self.get_default_config())
        boundaries = scorer.compute_boundaries(segments, [40.0], 50.0)
        boundaries = scorer.apply_long_take_protection(boundaries, segments, 50.0)
        self.assertEqual(len(boundaries), 1)
        # 原 score 0.35 被降低 0.15 -> 0.20, 低于 medium 阈值
        self.assertEqual(boundaries[0].confidence, "low")
        self.assertIn("长镜头/高连贯保护", boundaries[0].reason)

    def test_merge_short_segments(self):
        scorer = SplitScorer(self.get_default_config())
        merged = scorer.merge_short_segments([0.0, 0.5, 2.0, 2.6, 10.0], 1.0)
        # 0.5 和 2.0 之间距离 1.5 >= 1.0 保留；2.6 与 2.0 距离 0.6 < 1.0 合并
        self.assertEqual(merged, [0.0, 2.0, 10.0])


if __name__ == "__main__":
    unittest.main()
