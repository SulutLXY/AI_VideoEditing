"""关系图构建器单元测试"""
import unittest

from src.models import Shot
from src.relationship_builder import RelationshipBuilder


class TestRelationshipBuilder(unittest.TestCase):
    def test_camera_change_is_scene_cut(self):
        shots = [
            Shot(
                shot_id="S001", state="RAW", source_file="a.mp4", source_path="/a.mp4",
                tc_in="00:00:00:00", tc_out="00:00:05:00", duration_sec=5.0,
                camera_position="A机位", location="咖啡馆", emotion="焦虑",
            ),
            Shot(
                shot_id="S002", state="RAW", source_file="a.mp4", source_path="/a.mp4",
                tc_in="00:00:05:00", tc_out="00:00:10:00", duration_sec=5.0,
                camera_position="B机位", location="咖啡馆", emotion="焦虑",
            ),
        ]
        builder = RelationshipBuilder()
        result = builder.build(shots)
        self.assertEqual(result[1].relationships.prev.shot_id, "S001")
        self.assertEqual(result[1].relationships.prev.relationship_type, "机位跳切")
        self.assertLessEqual(result[1].relationships.prev.coherence_score, 0.5)

    def test_same_scene_emotion_is_continuation(self):
        shots = [
            Shot(
                shot_id="S001", state="RAW", source_file="a.mp4", source_path="/a.mp4",
                tc_in="00:00:00:00", tc_out="00:00:05:00", duration_sec=5.0,
                camera_position="A机位", location="咖啡馆", emotion="焦虑",
            ),
            Shot(
                shot_id="S002", state="RAW", source_file="a.mp4", source_path="/a.mp4",
                tc_in="00:00:05:00", tc_out="00:00:10:00", duration_sec=5.0,
                camera_position="A机位", location="咖啡馆", emotion="焦虑",
            ),
        ]
        builder = RelationshipBuilder()
        result = builder.build(shots)
        self.assertEqual(result[1].relationships.prev.relationship_type, "情绪延续")
        self.assertGreaterEqual(result[1].relationships.prev.coherence_score, 0.75)


if __name__ == "__main__":
    unittest.main()
