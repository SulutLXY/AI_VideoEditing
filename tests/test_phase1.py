"""Phase 1: 素材分析、状态分发与切分决策单元测试"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.models import Shot, Segment, ScriptBeat
from src.processors.raw_processor import RawProcessor
from src.processors.processed_processor import ProcessedProcessor
from src.processors.analyzed_processor import AnalyzedProcessor
from src.material_processor import MaterialProcessor
from src.services.video_vlm_service import VideoSegment, VideoAnalysisResult


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


class MockVLMService:
    def sample_frames(self, video_path, duration, temp_dir):
        return []

    def analyze_whole_video(self, video_path, frames, duration):
        return {
            "location": "咖啡馆",
            "characters": ["男主"],
            "action": "等待",
            "emotion": "焦虑",
            "shot_size": "中景",
            "camera_position": "A机位",
            "camera_movement": "固定",
        }


class MockVideoVLMService:
    def __init__(self, config=None):
        self.config = config or {}

    def analyze_video(self, video_path, duration, temp_dir):
        return VideoAnalysisResult(
            segments=[
                VideoSegment(
                    start=0.0,
                    end=duration,
                    description="男主在咖啡馆等待",
                    coherence_score=0.95,
                    is_long_take=True,
                    shot_size="中景",
                    camera_position="A机位",
                    camera_movement="固定",
                    location="咖啡馆",
                    characters=["男主"],
                    action="等待",
                    emotion="焦虑",
                )
            ],
            suggested_cut_points=[],
        )


class MockVideoVLMServiceWithCut:
    """模拟两段弱连贯，建议切分"""
    def __init__(self, config=None):
        self.config = config or {}

    def analyze_video(self, video_path, duration, temp_dir):
        return VideoAnalysisResult(
            segments=[
                VideoSegment(
                    start=0.0,
                    end=5.0,
                    description="男主在咖啡馆",
                    coherence_score=0.5,
                    is_long_take=False,
                    shot_size="中景",
                    camera_position="A机位",
                    camera_movement="固定",
                    location="咖啡馆",
                    characters=["男主"],
                    action="等待",
                    emotion="焦虑",
                ),
                VideoSegment(
                    start=5.0,
                    end=duration,
                    description="女主在门口",
                    coherence_score=0.5,
                    is_long_take=False,
                    shot_size="近景",
                    camera_position="B机位",
                    camera_movement="固定",
                    location="门口",
                    characters=["女主"],
                    action="推门",
                    emotion="紧张",
                ),
            ],
            suggested_cut_points=[5.0],
        )


class MockASRService:
    def transcribe(self, video_path, temp_dir):
        return []


class Counter:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return f"S{self.n:03d}"


class TestRawProcessor(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase1_raw_test_")
        self.temp_dir = tempfile.mkdtemp(prefix="phase1_raw_temp_")
        self.config = {
            "paths": {"output": self.output_dir, "temp": self.temp_dir},
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
            "processing": {"min_shot_duration": 1.0, "scene_threshold": 0.3},
        }
        from src.split_scorer import SplitScorer
        self.split_scorer = SplitScorer(self.config["split_scoring"])
        self.asr = MockASRService()

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _processor(self):
        return RawProcessor(
            vlm_service=MockVLMService(),
            video_vlm_service=MockVideoVLMService(),
            asr_service=self.asr,
            split_scorer=self.split_scorer,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            config=self.config,
        )

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_raw_no_cut(self, mock_scores, mock_split, mock_cv):
        """无场景变化时，Phase 0 输出单个片段"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        mock_scores.return_value = []

        proc = self._processor()
        counter = Counter()
        shots = proc.process("/tmp/test.mp4", counter.next)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].state, "RAW")
        self.assertEqual(shots[0].duration_sec, 10.0)
        self.assertIn("resolution", shots[0].cv_metadata)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_raw_scene_change_is_split(self, mock_scores, mock_split, mock_cv):
        """CV 检测到硬切时，Phase 0 应拆分"""
        mock_cv.return_value = mock_cv_meta(duration=10.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        # 在 5s 处有明显变化，前后平和
        mock_scores.return_value = [
            (4.8, 0.05, 0.05), (4.9, 0.05, 0.05), (5.0, 0.45, 0.45),
            (5.1, 0.05, 0.05), (5.2, 0.05, 0.05),
        ]

        proc = self._processor()
        counter = Counter()
        shots = proc.process("/tmp/test.mp4", counter.next)

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0].source_file, "test.mp4")
        self.assertEqual(shots[1].source_file, "test.mp4")
        self.assertTrue(mock_split.call_count >= 2)


class TestProcessedProcessor(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase1_proc_test_")
        self.temp_dir = tempfile.mkdtemp(prefix="phase1_proc_temp_")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.processors.processed_processor.split_video")
    @patch("src.processors.processed_processor.cv_pre_scan")
    @patch("src.processors.processed_processor.extract_keyframes")
    def test_processed_not_split(self, mock_extract, mock_cv, mock_split):
        """PROCESSED 素材应只生成一个 Shot，不切分，并生成配置与物理片段"""
        mock_cv.return_value = mock_cv_meta(duration=15.0)
        mock_extract.return_value = []
        mock_split.return_value = "/tmp/fake_processed_clip.mp4"

        proc = ProcessedProcessor(
            vlm_service=MockVLMService(),
            asr_service=MockASRService(),
            temp_dir=self.temp_dir,
            output_dir=self.output_dir,
        )
        counter = Counter()
        shots = proc.process("/tmp/processed.mp4", counter.next)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].state, "PROCESSED")
        self.assertEqual(shots[0].duration_sec, 15.0)
        self.assertTrue(shots[0].do_not_split)
        self.assertEqual(shots[0].camera_position, "A机位")
        self.assertIn("shot_config", shots[0].cv_metadata)
        self.assertTrue(mock_split.called)


class TestAnalyzedProcessor(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase1_ana_test_")
        self.temp_dir = tempfile.mkdtemp(prefix="phase1_ana_temp_")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.processors.analyzed_processor.split_video")
    @patch("src.processors.analyzed_processor.extract_keyframes")
    @patch("src.processors.analyzed_processor.cv_pre_scan")
    def test_analyzed_adapter_read(self, mock_cv, mock_extract, mock_split):
        """ANALYZED 素材通过 custom_v1 adapter 读取，不分析不切分，并生成配置"""
        mock_cv.return_value = mock_cv_meta(duration=12.0)
        mock_extract.return_value = []
        mock_split.return_value = "/tmp/fake_analyzed_clip.mp4"

        video_dir = tempfile.mkdtemp()
        video_path = os.path.join(video_dir, "scene_01.mp4")
        meta_path = video_path.replace(".mp4", "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": "scene_01.mp4",
                "description": "男主在咖啡馆焦虑等待",
                "location": "咖啡馆",
                "emotion": "焦虑",
                "characters": ["男主"],
                "camera_position": "A机位",
            }, f)

        proc = AnalyzedProcessor(output_dir=self.output_dir, temp_dir=self.temp_dir)
        counter = Counter()
        shots = proc.process(video_path, "custom_v1", counter.next)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].state, "ANALYZED")
        self.assertTrue(shots[0].do_not_split)
        self.assertEqual(shots[0].location, "咖啡馆")
        self.assertEqual(shots[0].emotion, "焦虑")
        self.assertIn("shot_config", shots[0].cv_metadata)
        self.assertFalse(shots[0].needs_review)  # 关键字段齐全
        self.assertTrue(mock_split.called)

    @patch("src.processors.analyzed_processor.split_video")
    @patch("src.processors.analyzed_processor.extract_keyframes")
    @patch("src.processors.analyzed_processor.cv_pre_scan")
    def test_analyzed_missing_meta_fallback(self, mock_cv, mock_extract, mock_split):
        """ANALYZED 素材找不到分析文件时 fallback 并标记复核"""
        mock_cv.return_value = mock_cv_meta(duration=8.0)
        mock_extract.return_value = []
        mock_split.return_value = "/tmp/fake_analyzed_clip.mp4"

        video_path = "/tmp/nonexistent_analyzed.mp4"
        proc = AnalyzedProcessor(output_dir=self.output_dir, temp_dir=self.temp_dir)
        counter = Counter()
        shots = proc.process(video_path, "custom_v1", counter.next)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].state, "ANALYZED")
        self.assertTrue(shots[0].needs_review)
        self.assertEqual(shots[0].duration_sec, 8.0)


class TestMaterialProcessorDispatch(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase1_mp_test_")
        self.temp_dir = tempfile.mkdtemp(prefix="phase1_mp_temp_")
        self.config = {
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
            "processing": {"min_shot_duration": 1.0},
        }
        from src.split_scorer import SplitScorer
        self.split_scorer = SplitScorer(self.config["split_scoring"])

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.phase0_rough_cut.cv_pre_scan")
    @patch("src.phase0_rough_cut.split_video")
    @patch("src.phase0_rough_cut.compute_frame_scene_scores")
    def test_dispatch_raw(self, mock_scores, mock_split, mock_cv):
        """MaterialProcessor 对 RAW 状态分发到 RawProcessor（Phase 0 粗剪）"""
        mock_cv.return_value = mock_cv_meta(duration=40.0)
        mock_split.return_value = "/tmp/fake_split.mp4"
        mock_scores.return_value = []

        mp = MaterialProcessor(
            vlm_service=MockVLMService(),
            video_vlm_service=MockVideoVLMService(),
            asr_service=MockASRService(),
            split_scorer=self.split_scorer,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            processing_config=self.config["processing"],
            config={"paths": {"output": self.output_dir, "temp": self.temp_dir}},
        )
        counter = Counter()
        shots = mp.process("/tmp/raw.mp4", "RAW", None, counter.next)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].state, "RAW")


if __name__ == "__main__":
    unittest.main()
