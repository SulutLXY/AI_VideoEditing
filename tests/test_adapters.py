"""Adapter 单元测试"""
import json
import os
import tempfile
import unittest

from src.adapters.base import build_analyzed_shot
from src.adapters.custom_v1 import CustomV1Adapter
from src.adapters.csv_meta import CsvMetaAdapter
from src.adapters.autocut_v1 import AutocutV1Adapter


class TestBuildAnalyzedShot(unittest.TestCase):
    def test_basic(self):
        shot = build_analyzed_shot(
            shot_id="S001",
            source_file="test.mp4",
            source_path="/tmp/test.mp4",
            duration_sec=10.0,
            vlm_description={
                "location": "咖啡馆",
                "emotion": "焦虑",
                "characters": ["男主"],
            },
            adapter_name="custom_v1",
            missing_fields=["action"],
        )
        self.assertEqual(shot.state, "ANALYZED")
        self.assertTrue(shot.do_not_split)
        self.assertTrue(shot.needs_review)
        self.assertEqual(shot.location, "咖啡馆")
        self.assertEqual(shot.provenance.conversion["missing_fields"], ["action"])


class TestCustomV1AdapterCanRead(unittest.TestCase):
    def test_recognizes_custom_v1(self):
        adapter = CustomV1Adapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"filename": "a.mp4", "description": "test"}, f)
            path = f.name
        try:
            self.assertTrue(adapter.can_read(path))
        finally:
            os.remove(path)

    def test_rejects_invalid_json(self):
        adapter = CustomV1Adapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            path = f.name
        try:
            self.assertFalse(adapter.can_read(path))
        finally:
            os.remove(path)


class TestCsvMetaAdapterCanRead(unittest.TestCase):
    def test_recognizes_csv(self):
        adapter = CsvMetaAdapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="") as f:
            f.write("filename,location,emotion\na.mp4,咖啡馆,焦虑\n")
            path = f.name
        try:
            self.assertTrue(adapter.can_read(path))
        finally:
            os.remove(path)


class TestAutocutV1Adapter(unittest.TestCase):
    def test_recognizes_autocut_config(self):
        adapter = AutocutV1Adapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "shot_id": "S001",
                "clip_path": "S001.mp4",
                "duration_sec": 10.0,
                "content_summary": "男主等待",
                "shot_type": "中景",
                "emotion": "焦虑",
                "location": "咖啡馆",
                "tags": ["男主", "等待"],
            }, f)
            path = f.name
        try:
            self.assertTrue(adapter.can_read(path))
        finally:
            os.remove(path)

    def test_rejects_plain_dict(self):
        adapter = AutocutV1Adapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"foo": "bar"}, f)
            path = f.name
        try:
            self.assertFalse(adapter.can_read(path))
        finally:
            os.remove(path)

    def test_maps_fields(self):
        adapter = AutocutV1Adapter()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "shot_id": "S001",
                "clip_path": "S001.mp4",
                "duration_sec": 10.0,
                "resolution": [1920, 1080],
                "fps": 24.0,
                "content_summary": "男主等待",
                "shot_type": "中景",
                "camera_position": "A机位",
                "camera_movement": "固定",
                "emotion": "焦虑",
                "location": "咖啡馆",
                "tags": ["男主", "等待"],
                "is_long_take": False,
                "coherence_score": 0.9,
            }, f)
            path = f.name
        try:
            shot = adapter.read(path, video_path="/tmp/S001.mp4")
            self.assertEqual(shot.state, "ANALYZED")
            self.assertEqual(shot.action, "男主等待")
            self.assertEqual(shot.shot_size, "中景")
            self.assertEqual(shot.emotion, "焦虑")
            self.assertEqual(shot.location, "咖啡馆")
            self.assertEqual(shot.camera_position, "A机位")
            self.assertEqual(shot.tags, ["男主", "等待"])
            self.assertEqual(shot.coherence_score, 0.9)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
