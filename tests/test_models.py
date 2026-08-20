"""数据模型单元测试"""
import unittest

from src.models import Shot, Provenance, Relationship, Relationships, Segment, Boundary, ScriptBeat


class TestShot(unittest.TestCase):
    def test_round_trip(self):
        shot = Shot(
            shot_id="S001",
            state="RAW",
            source_file="test.mp4",
            source_path="/tmp/test.mp4",
            tc_in="00:00:00:00",
            tc_out="00:00:10:00",
            duration_sec=10.0,
            location="咖啡馆",
            emotion="焦虑",
            characters=["男主"],
        )
        shot.provenance = Provenance(state="RAW", generated_by="test")
        shot.relationships.prev = Relationship(shot_id="S000", relationship_type="情绪延续", coherence_score=0.85)
        shot.relationships.next = Relationship(shot_id="S002", relationship_type="场景切换", coherence_score=0.3)

        data = shot.to_dict()
        shot2 = Shot.from_dict(data)

        self.assertEqual(shot2.shot_id, "S001")
        self.assertEqual(shot2.location, "咖啡馆")
        self.assertEqual(shot2.emotion, "焦虑")
        self.assertEqual(shot2.vlm_description["location"], "咖啡馆")
        self.assertEqual(shot2.vlm_description["emotion"], "焦虑")
        self.assertEqual(shot2.relationships.prev.shot_id, "S000")
        self.assertEqual(shot2.relationships.next.coherence_score, 0.3)

    def test_legacy_vlm_description_compatibility(self):
        """兼容旧版 JSON 中 vlm_description 字典"""
        data = {
            "shot_id": "S001",
            "state": "RAW",
            "source_file": "test.mp4",
            "source_path": "/tmp/test.mp4",
            "tc_in": "00:00:00:00",
            "tc_out": "00:00:10:00",
            "duration_sec": 10.0,
            "vlm_description": {
                "location": "咖啡馆",
                "emotion": "焦虑",
                "characters": ["男主"],
                "shot_size": "中景",
            },
        }
        shot = Shot.from_dict(data)
        self.assertEqual(shot.location, "咖啡馆")
        self.assertEqual(shot.emotion, "焦虑")
        self.assertEqual(shot.shot_size, "中景")
        self.assertEqual(shot.vlm_description["characters"], ["男主"])


class TestSegmentAndBoundary(unittest.TestCase):
    def test_segment_to_dict(self):
        seg = Segment(start=0.0, end=10.0, description="测试", camera_position="A机位")
        data = seg.to_dict()
        self.assertEqual(data["start"], 0.0)
        self.assertEqual(data["camera_position"], "A机位")

    def test_boundary_confidence(self):
        b = Boundary(time_sec=5.0, score=0.6, confidence="high")
        self.assertEqual(b.confidence, "high")


class TestScriptBeat(unittest.TestCase):
    def test_to_dict(self):
        beat = ScriptBeat(
            act="第一幕",
            scene="场1",
            beat_id="场1-情节点A",
            location="咖啡馆",
            time="傍晚",
            content="男主等待",
            emotion="焦虑",
            key_actions=["看表"],
            key_dialogue="",
        )
        data = beat.to_dict()
        self.assertEqual(data["act"], "第一幕")
        self.assertEqual(data["key_actions"], ["看表"])


if __name__ == "__main__":
    unittest.main()
