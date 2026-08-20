"""Phase 3: 剪辑语法决策单元测试"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from src.models import Shot
from src.phase3_editor import Phase3Editor, EditDecision, parse_speed_multiplier, format_speed


class TestPhase3Editor(unittest.TestCase):

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix="phase3_test_")
        self.config = {
            "project": {"name": "测试", "style": "紧张", "target_duration": "5m", "genre": "剧情短片"},
            "processing": {
                "slow_motion_min_fps": 60,
                "export_csv": True,
                "export_json": True,
            },
            "models": {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "api_key": "fake",
                    "base_url": "",
                    "max_tokens": 4096,
                }
            },
            "paths": {"output": self.output_dir},
        }

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    @staticmethod
    def _shot(
        shot_id,
        status="核心",
        state="RAW",
        fps=24.0,
        duration=5.0,
        beat="场1-A",
        quality_score=0.5,
        tags=None,
    ):
        shot = Shot(
            shot_id=shot_id,
            state=state,
            source_file="test.mp4",
            source_path="/tmp/test.mp4",
            tc_in="00:00:00:00",
            tc_out=f"00:00:0{int(duration)}:00",
            duration_sec=duration,
            fps=fps,
            status=status,
            quality_score=quality_score,
            script_anchor={"beat": beat, "function": "叙事", "confidence": 0.9},
            location="咖啡馆",
            characters=["男主"],
            action="等待",
            emotion="焦虑",
            shot_size="中景",
            camera_position="A机位",
            camera_movement="固定",
            tags=tags or [],
        )
        shot.vlm_description = {
            "location": shot.location,
            "characters": shot.characters,
            "action": shot.action,
            "emotion": shot.emotion,
            "shot_size": shot.shot_size,
            "camera_position": shot.camera_position,
            "camera_movement": shot.camera_movement,
            "direction": "从左向右",
            "performance": "自然",
            "action_details": "无",
            "continuity_score": 0.8,
            "continuity_notes": "连续",
        }
        return shot

    def _editor(self):
        with patch.object(Phase3Editor, "_init_llm_client", return_value=None):
            return Phase3Editor(self.config)

    def test_filters_status(self):
        """只有核心/备选/强制保留/待复核/保留的镜头进入剪辑决策"""
        editor = self._editor()
        shots = [
            self._shot("S001", "核心"),
            self._shot("S002", "备选"),
            self._shot("S003", "强制保留"),
            self._shot("S004", "待复核"),
            self._shot("S005", "废弃"),
            self._shot("S006", "未匹配"),
        ]

        decisions = [
            EditDecision(1, "S001", "test.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "叙事"),
            EditDecision(2, "S002", "test.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "叙事"),
            EditDecision(3, "S003", "test.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "叙事"),
            EditDecision(4, "S004", "test.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "叙事"),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run(shots)

        shot_ids = {d.shot_id for d in result}
        self.assertIn("S001", shot_ids)
        self.assertIn("S002", shot_ids)
        self.assertIn("S003", shot_ids)
        self.assertIn("S004", shot_ids)
        self.assertNotIn("S005", shot_ids)
        self.assertNotIn("S006", shot_ids)

    def test_protected_shots_not_deleted(self):
        """强制保留/待复核素材即使 LLM 建议删除，后处理也会撤销"""
        editor = self._editor()
        shots = [
            self._shot("S001", "核心"),
            self._shot("S002", "强制保留"),
            self._shot("S003", "待复核"),
        ]

        # LLM 错误地建议删除所有镜头
        decisions = [
            EditDecision(1, "S001", "test.mp4", "00:00:00:00", "00:00:05:00", "删除", "连续剪辑", "硬切", "保留原声", ""),
            EditDecision(2, "S002", "test.mp4", "00:00:00:00", "00:00:05:00", "删除", "连续剪辑", "硬切", "保留原声", ""),
            EditDecision(3, "S003", "test.mp4", "00:00:00:00", "00:00:05:00", "删除", "连续剪辑", "硬切", "保留原声", ""),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run(shots)

        result_map = {d.shot_id: d for d in result}
        # 核心素材被 LLM 删除，不应出现在结果中
        self.assertNotIn("S001", result_map)
        # 强制保留/待复核被保护
        self.assertEqual(result_map["S002"].speed, "1x")
        self.assertIn("强制保留", result_map["S002"].notes)
        self.assertEqual(result_map["S003"].speed, "1x")
        self.assertIn("待复核", result_map["S003"].notes)

    def test_slow_motion_fps_guard(self):
        """素材帧率不足 60fps 时，升格决策被改回原速"""
        editor = self._editor()
        shots = [self._shot("S001", "核心", fps=30.0)]

        decisions = [
            EditDecision(1, "S001", "test.mp4", "00:00:00:00", "00:00:05:00", "50%", "连续剪辑", "硬切", "保留原声", "情绪高潮"),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run(shots)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speed, "1x")
        self.assertIn("30", result[0].notes)

    def test_outputs_files(self):
        """Phase 3 输出 JSON 和 CSV 文件"""
        editor = self._editor()
        shots = [self._shot("S001", "核心")]
        decisions = [
            EditDecision(1, "S001", "test.mp4", "00:00:00:00", "00:00:05:00", "1x", "连续剪辑", "硬切", "保留原声", "开场"),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            editor.run(shots)

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "phase3_edit_decision.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "phase3_edit_decision.csv")))

        with open(os.path.join(self.output_dir, "phase3_edit_decision.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_decisions"], 1)
        self.assertIn("total_projected_duration", data)

    def test_parse_keep_field_deletes_unprotected(self):
        """LLM 返回 keep=false 时非保护素材被删除"""
        editor = self._editor()
        shots = [
            self._shot("S001", "核心"),
            self._shot("S002", "核心"),
        ]

        raw = [
            {"shot_id": "S001", "keep": True, "speed": "1x"},
            {"shot_id": "S002", "keep": False, "speed": "1x"},
        ]
        decisions = editor._parse_llm_decisions(raw, shots, 0)

        self.assertEqual(len(decisions), 2)
        # parse 阶段保留两个对象，第二个 speed 被标记为删除
        self.assertEqual(decisions[1].speed, "删除")

        # 经过保护后处理，S002 被删除
        kept = editor._post_process_protected_shots(decisions, shots)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].shot_id, "S001")

    def test_compress_when_over_target(self):
        """总时长超过上限时自动提速并删除低质量镜头"""
        self.config["project"]["target_duration"] = "3s"
        editor = self._editor()

        shots = [
            self._shot("S001", "核心", duration=5.0, quality_score=0.9),
            self._shot("S002", "核心", duration=5.0, quality_score=0.5),
            self._shot("S003", "核心", duration=5.0, quality_score=0.2),
            self._shot("S004", "核心", duration=5.0, quality_score=0.1),
        ]

        decisions = [
            EditDecision(i + 1, s.shot_id, "test.mp4", s.tc_in, s.tc_out, "1x", "连续剪辑", "硬切", "保留原声", "")
            for i, s in enumerate(shots)
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run(shots)

        total = editor._projected_duration(result, shots)
        self.assertLessEqual(total, editor.target_max)
        # 低质量镜头应被删除以控制时长
        kept_ids = {d.shot_id for d in result}
        self.assertNotIn("S004", kept_ids)

    def test_extend_and_supplement_when_under_target(self):
        """总时长低于下限时先慢放核心镜头，再从备选池补充"""
        self.config["project"]["target_duration"] = "30s"
        editor = self._editor()

        core_shot = self._shot("S001", "核心", duration=5.0, fps=60.0)
        supplement1 = self._shot("S002", "备选", duration=10.0, beat="场1-B")
        supplement2 = self._shot("S003", "备选", duration=10.0, beat="场1-B")

        decisions = [
            EditDecision(1, core_shot.shot_id, "test.mp4", core_shot.tc_in, core_shot.tc_out, "1x", "连续剪辑", "硬切", "保留原声", ""),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run([core_shot, supplement1, supplement2])

        total = editor._projected_duration(result, [core_shot, supplement1, supplement2])
        self.assertGreaterEqual(total, editor.target_min)
        self.assertLessEqual(total, editor.target_max)
        # 至少补充了一个备选镜头
        kept_ids = {d.shot_id for d in result}
        self.assertTrue("S002" in kept_ids or "S003" in kept_ids)

    def test_processed_material_speed_unrestricted(self):
        """PROCESSED 素材完全交给 LLM，帧率不足也不取消升格"""
        editor = self._editor()
        shots = [self._shot("S001", "核心", state="PROCESSED", fps=30.0)]

        decisions = [
            EditDecision(1, "S001", "test.mp4", "00:00:00:00", "00:00:05:00", "50%", "连续剪辑", "硬切", "保留原声", "情绪高潮"),
        ]

        with patch.object(editor, "_llm_edit_decision", return_value=decisions):
            result = editor.run(shots)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].speed, "50%")

    def test_speed_multiplier_helpers(self):
        """速度字符串解析与格式化"""
        self.assertAlmostEqual(parse_speed_multiplier("50%"), 0.5)
        self.assertAlmostEqual(parse_speed_multiplier("200%"), 2.0)
        self.assertAlmostEqual(parse_speed_multiplier("1x"), 1.0)
        self.assertAlmostEqual(parse_speed_multiplier("2.5x"), 2.5)
        self.assertIsNone(parse_speed_multiplier("删除"))

        self.assertEqual(format_speed(1.0), "1x")
        self.assertEqual(format_speed(0.5), "50%")
        self.assertEqual(format_speed(2.0), "2x")
        self.assertEqual(format_speed(2.5), "2.5x")


if __name__ == "__main__":
    unittest.main()
