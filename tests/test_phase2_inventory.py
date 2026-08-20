"""Phase 2 素材库轻量清点单元测试"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.phase2_inventory import MaterialInventoryBuilder


class TestMaterialInventoryBuilder(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="phase2_inventory_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _touch(self, filename):
        path = os.path.join(self.test_dir, filename)
        with open(path, "wb") as f:
            f.write(b"dummy")
        return path

    def _fake_cv_meta(self, video_path, duration=10.0, fps=24.0, resolution=(1920, 1080)):
        width, height = resolution
        return {
            "resolution": resolution,
            "aspect_ratio": f"{width}:{height}",
            "fps": fps,
            "duration": duration,
            "bitrate": "5000000",
            "codec": "h264",
            "visual_quality": 3.5,
            "scene_change_candidates": [],
        }

    def test_build_shots_from_directory(self):
        """从目录生成基础 Shot 清单"""
        self._touch("scene_a.mp4")
        self._touch("scene_b.mov")

        with patch("src.phase2_inventory.cv_pre_scan") as mock_scan:
            mock_scan.side_effect = lambda p: self._fake_cv_meta(p, duration=12.0 if "scene_a" in p else 8.0)
            builder = MaterialInventoryBuilder()
            shots = builder.build_shots_from_directory(self.test_dir)

        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0].shot_id, "S001")
        self.assertEqual(shots[1].shot_id, "S002")
        self.assertEqual(shots[0].duration_sec, 12.0)
        self.assertEqual(shots[1].duration_sec, 8.0)
        self.assertEqual(shots[0].state, "RAW")
        self.assertIn("cv_inventory", shots[0].tags)
        self.assertTrue(shots[0].needs_review)
        self.assertIsNotNone(shots[0].cv_metadata)

    def test_skip_invalid_video(self):
        """遇到无法解析的视频时跳过，不中断流程"""
        self._touch("valid.mp4")
        self._touch("invalid.mp4")

        def side_effect(p):
            if "invalid" in p:
                raise ValueError("mock decode error")
            return self._fake_cv_meta(p)

        with patch("src.phase2_inventory.cv_pre_scan") as mock_scan:
            mock_scan.side_effect = side_effect
            builder = MaterialInventoryBuilder()
            shots = builder.build_shots_from_directory(self.test_dir)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].source_file, "valid.mp4")

    def test_empty_directory_returns_empty(self):
        """空目录返回空列表"""
        builder = MaterialInventoryBuilder()
        shots = builder.build_shots_from_directory(self.test_dir)
        self.assertEqual(shots, [])

    def test_nonexistent_directory_returns_empty(self):
        """目录不存在返回空列表"""
        builder = MaterialInventoryBuilder()
        shots = builder.build_shots_from_directory("/nonexistent/path/12345")
        self.assertEqual(shots, [])

    def test_shot_id_sequence(self):
        """Shot ID 按顺序分配"""
        for i in range(3):
            self._touch(f"clip{i}.mp4")

        with patch("src.phase2_inventory.cv_pre_scan") as mock_scan:
            mock_scan.side_effect = lambda p: self._fake_cv_meta(p)
            builder = MaterialInventoryBuilder()
            shots = builder.build_shots_from_directory(self.test_dir)

        self.assertEqual([s.shot_id for s in shots], ["S001", "S002", "S003"])


if __name__ == "__main__":
    unittest.main()
