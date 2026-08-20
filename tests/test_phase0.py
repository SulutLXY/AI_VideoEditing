"""Phase 0: 纯 CV 粗剪单元测试"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.phase0_rough_cut import RoughCutAnalyzer
from src.models import Shot


def mock_cv_meta(duration=10.0, fps=24.0):
    return {
        "resolution": (1920, 1080),
        "aspect_ratio": "16:9",
        "fps": fps,
        "duration": duration,
        "bitrate": "5000k",
        "codec": "h264",
        "visual_quality": 3.5,
        "scene_change_candidates": [2.0, 5.0],
    }


def flat_scores(duration=10.0, step=0.1, value=0.02):
    """生成一段平稳低分的 scene score 序列

    返回 (timestamp, cut_score, hist_score) 三元组，测试中两者保持一致。
    """
    scores = []
    t = 0.0
    while t <= duration:
        scores.append((round(t, 3), value, value))
        t += step
    return scores


def inject_peak(scores, peak_time, peak_value, width=0.1):
    """在指定时间注入一个尖锐峰值"""
    return [
        (t, peak_value if abs(t - peak_time) <= width / 2 else s,
         peak_value if abs(t - peak_time) <= width / 2 else s)
        for t, s, _ in scores
    ]


def inject_ramp(scores, start, end, low=0.05, high=0.35):
    """在指定区间注入一个缓坡/平台，模拟软转场"""
    result = []
    for t, s, h in scores:
        if start <= t <= end:
            # 梯形：中间保持 high，两边渐变
            if t < start + 0.3:
                ratio = (t - start) / 0.3
            elif t > end - 0.3:
                ratio = (end - t) / 0.3
            else:
                ratio = 1.0
            new_s = low + (high - low) * min(1.0, ratio)
            result.append((t, new_s, new_s))
        else:
            result.append((t, s, h))
    return result


class TestRoughCutAnalyzer(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase0_test_")
        self.config = {
            "project": {"name": "测试项目"},
            "paths": {"output": self.output_dir, "temp": tempfile.mkdtemp()},
            "processing": {"scene_threshold": 0.3, "min_shot_duration": 1.0},
            "split_scoring": {
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
            },
        }

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_no_scene_change_outputs_single_shot(self, mock_scores, mock_split, mock_cv):
        """无场景变化时，输出单个 Shot"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        mock_scores.return_value = flat_scores(10.0)

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer.run(["/tmp/test.mp4"])

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].duration_sec, 10.0)
        self.assertEqual(shots[0].state, "RAW")
        self.assertTrue(shots[0].needs_review)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_hard_cut_is_split(self, mock_scores, mock_split, mock_cv):
        """硬切场景应被拆分"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        # 5s 处孤立尖锐峰值
        scores = flat_scores(10.0, value=0.02)
        scores = inject_peak(scores, 5.0, 0.45, width=0.05)
        mock_scores.return_value = scores

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer.run(["/tmp/test.mp4"])

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0].duration_sec, 5.0)
        self.assertEqual(shots[1].duration_sec, 5.0)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_motion_burst_is_suppressed(self, mock_scores, mock_split, mock_cv):
        """运动/快闪场景应被抑制，不拆分"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        # 4.5~5.5s 持续高分，模拟运动/快闪
        scores = flat_scores(10.0, value=0.02)
        scores = inject_ramp(scores, 4.5, 5.5, low=0.05, high=0.45)
        mock_scores.return_value = scores

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer.run(["/tmp/test.mp4"])

        # 快闪区域被抑制，应只输出一个片段
        self.assertEqual(len(shots), 1)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_short_segments_merged(self, mock_scores, mock_split, mock_cv):
        """过短片段应被合并"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        scores = flat_scores(10.0, value=0.02)
        scores = inject_peak(scores, 0.5, 0.45, width=0.05)
        scores = inject_peak(scores, 5.0, 0.45, width=0.05)
        mock_scores.return_value = scores

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer.run(["/tmp/test.mp4"])

        # 0.5s 切点产生 0-0.5(过短) 应被合并，最终两段：0-5, 5-10
        self.assertEqual(len(shots), 2)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_rough_config_saved(self, mock_scores, mock_split, mock_cv):
        """应保存 phase0_rough_config.json"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        mock_scores.return_value = flat_scores(10.0)

        analyzer = RoughCutAnalyzer(self.config)
        analyzer.run(["/tmp/test.mp4"])

        config_path = os.path.join(self.output_dir, "phase0_rough_config.json")
        self.assertTrue(os.path.exists(config_path))

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_soft_transition_is_marked_not_split(self, mock_scores, mock_split, mock_cv):
        """软转场应被标记但不切分"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        # 4.5~6.5s 持续中高分，模拟叠化/淡入淡出
        scores = flat_scores(10.0, value=0.02)
        scores = inject_ramp(scores, 4.5, 6.5, low=0.05, high=0.30)
        mock_scores.return_value = scores

        analyzer = RoughCutAnalyzer(self.config)
        shots = analyzer.run(["/tmp/test.mp4"])

        # 软转场不应触发切分
        self.assertEqual(len(shots), 1)
        self.assertTrue(shots[0].soft_transitions)
        self.assertEqual(shots[0].soft_transitions[0]["type"], "soft_transition")


if __name__ == "__main__":
    unittest.main()
