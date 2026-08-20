"""剧本预处理服务单元测试"""
import os
import tempfile
import unittest

from src.services.script_service import (
    read_script_file,
    is_structured_outline,
    clean_script_text,
    split_into_scenes,
    ScriptPreprocessor,
)


class MockLLMService:
    """模拟 LLMService，按 prompt 中的场景文本返回固定解析结果"""

    def generate(self, prompt: str) -> str:
        # 简单判断：prompt 中实际场景内容以哪个场号开头
        scene_text = prompt.split("## 这场戏的内容")[-1] if "## 这场戏的内容" in prompt else prompt
        if scene_text.strip().startswith("第二场") or "门口" in scene_text:
            return """{
  "act": "第一幕",
  "scene": "场2",
  "beats": [
    {
      "beat_id": "场2-情节点A",
      "location": "门口",
      "time": "雨夜",
      "content": "女主推门而入",
      "emotion": "紧张",
      "key_actions": ["推门", "对视"],
      "key_dialogue": "对不起，我来晚了",
      "suggested_shots": "近景跟随镜头"
    }
  ]
}"""
        return """{
  "act": "第一幕",
  "scene": "场1",
  "beats": [
    {
      "beat_id": "场1-情节点A",
      "location": "咖啡馆",
      "time": "傍晚",
      "content": "男主独自等待，表现焦虑",
      "emotion": "焦虑",
      "key_actions": ["看表", "深呼吸"],
      "key_dialogue": "",
      "suggested_shots": "中景固定机位"
    }
  ]
}"""


class TestReadScriptFile(unittest.TestCase):
    def test_read_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("第一场\n男主等待。\n")
            path = f.name
        try:
            text = read_script_file(path)
            self.assertIn("第一场", text)
        finally:
            os.remove(path)

    def test_read_md(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("## 第一幕\n### 场1\n- 地点：咖啡馆\n")
            path = f.name
        try:
            text = read_script_file(path)
            self.assertIn("地点：咖啡馆", text)
        finally:
            os.remove(path)


class TestOutlineDetection(unittest.TestCase):
    def test_recognizes_structured(self):
        text = "## 第一幕\n### 场1-情节点A\n- 地点：咖啡馆\n- 内容：等待\n"
        self.assertTrue(is_structured_outline(text))

    def test_rejects_plain_text(self):
        text = "第一场。男主在咖啡馆等待，表现焦虑。"
        self.assertFalse(is_structured_outline(text))


class TestSceneSplitting(unittest.TestCase):
    def test_split_by_chinese_scene_heading(self):
        text = "第一场 咖啡馆\n男主等待。\n\n第二场 门口\n女主推门。"
        scenes = split_into_scenes(text)
        self.assertEqual(len(scenes), 2)
        self.assertIn("第一场", scenes[0])
        self.assertIn("第二场", scenes[1])

    def test_fallback_chunk_when_no_heading(self):
        text = "这是一些没有场景标题的连续文本。" * 100
        scenes = split_into_scenes(text)
        # 没有场景标题时会按长度切分，至少有一块
        self.assertGreaterEqual(len(scenes), 1)


class TestScriptPreprocessor(unittest.TestCase):
    def test_structured_outline_pass_through(self):
        text = "## 第一幕\n### 场1-情节点A\n- 地点：咖啡馆\n- 内容：等待\n"
        preprocessor = ScriptPreprocessor(MockLLMService())
        result = preprocessor.preprocess(text)
        self.assertEqual(result.strip(), text.strip())

    def test_unstructured_script_parsed(self):
        text = "第一场 咖啡馆\n男主独自等待，表现焦虑，反复看表。\n\n第二场 门口\n女主推门而入，两人对视。"
        preprocessor = ScriptPreprocessor(MockLLMService())
        result = preprocessor.preprocess(text)
        self.assertIn("### 场1-情节点A", result)
        self.assertIn("### 场2-情节点A", result)
        self.assertIn("咖啡馆", result)
        self.assertIn("门口", result)

    def test_output_shot_requirements(self):
        text = "第一场 咖啡馆\n男主等待。"
        preprocessor = ScriptPreprocessor(MockLLMService(), {"output_shot_requirements": True})
        result = preprocessor.preprocess(text)
        self.assertIn("建议镜头", result)
        self.assertIn("中景固定机位", result)


if __name__ == "__main__":
    unittest.main()
