"""Phase 4: 二次匹配与执行输出单元测试"""
import csv
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.models import Shot
from src.phase3_editor import EditDecision
from src.phase4_exporter import Phase4Exporter


class TestPhase4Exporter(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase4_test_")
        self.config = {
            "project": {"name": "测试项目", "style": "紧张"},
            "paths": {"output": self.output_dir, "raw_materials": "/tmp"},
            "processing": {
                "export_edl": True,
                "export_fcpxml": True,
                "export_csv": True,
                "export_json": True,
            },
        }

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @staticmethod
    def _shot(shot_id, source_path, fps=24.0, duration=5.0):
        shot = Shot(
            shot_id=shot_id,
            state="RAW",
            source_file=os.path.basename(source_path),
            source_path=source_path,
            tc_in="00:00:00:00",
            tc_out="00:00:05:00",
            duration_sec=duration,
            fps=fps,
            resolution=(1920, 1080),
            aspect_ratio="16:9",
            location="咖啡馆",
            characters=["男主"],
            action="等待",
            emotion="焦虑",
        )
        shot.vlm_description = {
            "location": shot.location,
            "characters": shot.characters,
            "action": shot.action,
            "emotion": shot.emotion,
        }
        return shot

    def _decisions(self):
        return [
            EditDecision(1, "S001", "clip1.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "开场"),
            EditDecision(2, "S002", "clip2.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "叠化", "保留原声", "过渡"),
        ]

    def _shots(self):
        return [
            self._shot("S001", "/tmp/clip1.mp4"),
            self._shot("S002", "/tmp/clip2.mp4"),
        ]

    @patch("os.path.exists")
    def test_export_all_formats(self, mock_exists):
        """Phase 4 导出 EDL/FCPXML/CSV/JSON 四种格式"""
        mock_exists.return_value = True

        exporter = Phase4Exporter(self.config)
        exporter.run(self._decisions(), self._shots())

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "timeline.edl")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "timeline.fcpxml")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "timeline_final.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "timeline.json")))

    @patch("os.path.exists")
    def test_edl_content(self, mock_exists):
        """EDL 包含标题、时间码和注释"""
        mock_exists.return_value = True

        exporter = Phase4Exporter(self.config)
        exporter.run(self._decisions(), self._shots())

        edl_path = os.path.join(self.output_dir, "timeline.edl")
        with open(edl_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("TITLE: 测试项目", content)
        self.assertIn("CLIP1", content)
        self.assertIn("CLIP2", content)
        self.assertIn("FROM CLIP NAME: clip1.mp4", content)
        self.assertIn("TRANSITION: 叠化", content)

    @patch("os.path.exists")
    def test_csv_content(self, mock_exists):
        """CSV 包含正确字段和顺序"""
        mock_exists.return_value = True

        exporter = Phase4Exporter(self.config)
        exporter.run(self._decisions(), self._shots())

        csv_path = os.path.join(self.output_dir, "timeline_final.csv")
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["镜头ID"], "S001")
        self.assertEqual(rows[1]["镜头ID"], "S002")
        self.assertEqual(rows[1]["转场"], "叠化")

    @patch("os.path.exists")
    def test_json_content(self, mock_exists):
        """JSON 时间线包含镜头描述和源路径"""
        mock_exists.return_value = True

        exporter = Phase4Exporter(self.config)
        exporter.run(self._decisions(), self._shots())

        json_path = os.path.join(self.output_dir, "timeline.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_clips"], 2)
        self.assertEqual(len(data["timeline"]), 2)
        self.assertEqual(data["timeline"][0]["shot_id"], "S001")
        self.assertIn("vlm_description", data["timeline"][0])

    @patch("os.path.exists")
    def test_missing_file_skipped(self, mock_exists):
        """物理文件不存在时跳过该决策"""
        def exists_side_effect(path):
            return path == "/tmp/clip1.mp4"
        mock_exists.side_effect = exists_side_effect

        exporter = Phase4Exporter(self.config)
        exporter.run(self._decisions(), self._shots())

        json_path = os.path.join(self.output_dir, "timeline.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_clips"], 1)
        self.assertEqual(data["timeline"][0]["shot_id"], "S001")

    @patch("os.path.exists")
    def test_speed_parsing(self, mock_exists):
        """速度解析正确影响时间线时长"""
        mock_exists.return_value = True

        decisions = [
            EditDecision(1, "S001", "clip1.mp4", "00:00:00:00", "00:00:10:00", "2x", "连续剪辑", "硬切", "保留原声", "快放"),
        ]
        shots = [self._shot("S001", "/tmp/clip1.mp4", duration=10.0)]

        exporter = Phase4Exporter(self.config)
        exporter.run(decisions, shots)

        json_path = os.path.join(self.output_dir, "timeline.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 2x 速度：10s 源素材 -> 5s 时间线
        timeline_in = self._tc_to_sec(data["timeline"][0]["timeline_in"])
        timeline_out = self._tc_to_sec(data["timeline"][0]["timeline_out"])
        self.assertAlmostEqual(timeline_out - timeline_in, 5.0, places=1)

    @staticmethod
    def _tc_to_sec(tc: str, fps: float = 24.0) -> float:
        parts = tc.split(":")
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + f / fps


if __name__ == "__main__":
    unittest.main()
