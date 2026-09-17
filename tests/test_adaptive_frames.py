# -*- coding: utf-8 -*-
"""自适应抽帧器纯函数单元测试（normalize_sims / assign_tiers / motion_summary）

五档语义（v2）：静止 static / 慢镜头 slow / 普通镜 normal / 快镜头 fast / 特快 very_fast。
折算感知速度为主轴、相似度为门槛；特快档仅由持续感知速度≥fast_flow 触发。
"""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.adaptive_frame_extractor import (
    DEFAULTS,
    TIER_FAST,
    TIER_NORMAL,
    TIER_SLOW,
    TIER_STATIC,
    TIER_VERY_FAST,
    assign_tiers,
    global_scale_motion,
    motion_summary,
    normalize_sims,
    scale_factor,
    extract_adaptive_frames,
    cache_signature,
)


class TestNormalizeSims(unittest.TestCase):
    def test_median_maps_to_half(self):
        sims = [0.5] * 10 + [0.9, 0.1]
        norm = normalize_sims(sims)
        med = sorted(norm)[len(norm) // 2 - 1]
        self.assertAlmostEqual(med, 0.5, places=6)

    def test_clipped_to_unit_range(self):
        norm = normalize_sims([0.0, 1.0, 0.2, 0.4, 0.6, 0.8, 0.55, 0.45, 0.3, 0.7])
        self.assertTrue(all(0.0 <= v <= 1.0 for v in norm))

    def test_constant_clip_falls_back_normal(self):
        # 全片无变化（MAD=0）时退化为全 0.5，避免除零/档位翻转
        norm = normalize_sims([0.95] * 20)
        self.assertEqual(norm, [0.5] * 20)

    def test_empty(self):
        self.assertEqual(normalize_sims([]), [])


class TestScaleCorrection(unittest.TestCase):
    def test_closeup_is_discounted_more_than_wide_shot(self):
        self.assertLess(scale_factor(0.918, DEFAULTS), scale_factor(0.880, DEFAULTS))

    def test_global_translation_does_not_look_like_zoom(self):
        flow = np.zeros((180, 320, 2), dtype=np.float32)
        flow[..., 0] = 6.0
        self.assertLess(global_scale_motion(flow), 0.1)

    def test_global_zoom_is_detected(self):
        h, w = 180, 320
        yy, xx = np.mgrid[:h, :w]
        flow = np.empty((h, w, 2), dtype=np.float32)
        flow[..., 0] = (xx - w / 2) * 0.02
        flow[..., 1] = (yy - h / 2) * 0.02
        self.assertGreater(global_scale_motion(flow), 1.5)


class TestAssignTiers(unittest.TestCase):
    CFG = {**DEFAULTS, "min_tier_hold": 0.3}

    @staticmethod
    def _times(n):
        return [float(i) * 0.1 for i in range(n)]

    @staticmethod
    def _flows(n, v=0.2):
        return [v] * n

    def test_static_when_high_sim_low_flow(self):
        times = self._times(20)
        tiers = assign_tiers(times, [0.95] * 20, [0.5] * 20, self._flows(20), self.CFG)
        self.assertEqual(set(tiers), {TIER_STATIC})

    def test_slow_band(self):
        # 光流 0.5（0.3~0.8）且 sim 0.8（≥0.75）→ 慢镜头档
        times = self._times(20)
        tiers = assign_tiers(times, [0.8] * 20, [0.5] * 20, self._flows(20, 0.5), self.CFG)
        self.assertEqual(set(tiers), {TIER_SLOW})

    def test_normal_band(self):
        # 光流 1.0（0.8~1.5）且 sim 0.6（≥0.55）→ 普通镜档
        times = self._times(20)
        tiers = assign_tiers(times, [0.6] * 20, [0.5] * 20, self._flows(20, 1.0), self.CFG)
        self.assertEqual(set(tiers), {TIER_NORMAL})

    def test_fast_band(self):
        # 光流 2.0（1.5~2.5）且 sim 0.4（≥0.35）→ 快镜头档
        times = self._times(20)
        tiers = assign_tiers(times, [0.4] * 20, [0.5] * 20, self._flows(20, 2.0), self.CFG)
        self.assertEqual(set(tiers), {TIER_FAST})

    def test_static_rejected_by_flow_becomes_very_fast(self):
        # 直方图盲区：相似度 0.97 但光流 6.0（对象位移）→ 不得判静止，光流≥5.5 判特快
        times = self._times(20)
        tiers = assign_tiers(times, [0.97] * 20, [0.5] * 20, self._flows(20, 6.0), self.CFG)
        self.assertEqual(set(tiers), {TIER_VERY_FAST})

    def test_fast_band_high_end_chase(self):
        # 追逐类快镜头（S001 实测）：光流 3.6、相似度 0.99 → 快镜头档，不静不快
        times = self._times(20)
        tiers = assign_tiers(times, [0.99] * 20, [0.5] * 20, self._flows(20, 3.6), self.CFG)
        self.assertEqual(set(tiers), {TIER_FAST})

    def test_visual_burst_is_fast_not_very_fast(self):
        # 单帧内容爆变属于抽帧事件，不应污染持续速度档位。
        times = self._times(20)
        tiers = assign_tiers(times, [0.3] * 20, [0.5] * 20, self._flows(20), self.CFG)
        self.assertEqual(set(tiers), {TIER_FAST})

    def test_relative_similarity_does_not_raise_speed_tier(self):
        # 归一化相似度仅作诊断记录，不再把局部骤变误判为持续特快。
        times = self._times(20)
        sims = [0.85] * 10 + [0.8] * 10
        norm = [0.9] * 10 + [0.3] * 10
        tiers = assign_tiers(times, sims, norm, self._flows(20), self.CFG)
        self.assertEqual(set(tiers), {TIER_SLOW})

    def test_very_fast_when_flow_spike(self):
        # 相似度在慢镜头档区间，但光流持续 6.0 → 特快（主体突动）
        times = self._times(20)
        flows = [0.2] * 10 + [6.0] * 10
        tiers = assign_tiers(times, [0.8] * 20, [0.5] * 20, flows, self.CFG)
        self.assertEqual(set(tiers[:10]), {TIER_SLOW})
        # 特快使用 0.1s 的独立滞回，捕获短促爆发。
        self.assertEqual(set(tiers[11:]), {TIER_VERY_FAST})

    def test_fast_when_big_visual_change_slow_flow(self):
        # 光流低但 raw 相似度 0.5（0.35~0.55）→ 快镜头档（画面大改）
        times = self._times(20)
        tiers = assign_tiers(times, [0.5] * 20, [0.5] * 20, self._flows(20), self.CFG)
        self.assertEqual(set(tiers), {TIER_FAST})

    def test_hysteresis_suppresses_brief_flip(self):
        # 0.2s 的瞬时掉档不应换档（候选档需持续满 min_tier_hold 0.3s 才晋升）
        times = self._times(20)
        sims = [0.95] * 10 + [0.6, 0.6] + [0.95] * 8
        tiers = assign_tiers(times, sims, [0.5] * 20, self._flows(20), self.CFG)
        self.assertEqual(set(tiers), {TIER_STATIC})

    def test_sustained_change_switches_tier(self):
        times = self._times(20)
        sims = [0.95] * 5 + [0.3] * 15
        tiers = assign_tiers(times, sims, [0.5] * 20, self._flows(20), self.CFG)
        self.assertEqual(tiers[0], TIER_STATIC)
        # 内容持续大改但光流低：满滞回后进入快镜头，不冒充特快。
        self.assertEqual(set(tiers[9:]), {TIER_FAST})
        self.assertIn(TIER_STATIC, tiers[:9])


class TestSamplingContract(unittest.TestCase):
    def test_signature_changes_with_settings_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.bin'
            source.write_bytes(b'a')
            original = cache_signature(str(source), DEFAULTS)
            self.assertNotEqual(original, cache_signature(str(source), {**DEFAULTS, 'normal_interval': 0.9}))
            source.write_bytes(b'abc')
            self.assertNotEqual(original, cache_signature(str(source), DEFAULTS))

    def test_static_motion_event_cannot_create_continuous_burst(self):
        class Capture:
            index = 0
            def isOpened(self): return True
            def get(self, key):
                import cv2
                return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 64,
                        cv2.CAP_PROP_FRAME_HEIGHT: 48,
                        cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) / 30 * 1000}.get(key, 0)
            def read(self):
                self.index += 1
                return (True, np.zeros((48, 64, 3), np.uint8)) if self.index <= 61 else (False, None)
            def release(self): pass
        flow = np.zeros((48, 64, 2), np.float32)
        flow[..., 0] = 4
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.bin'
            source.write_bytes(b'video')
            with patch('src.adaptive_frame_extractor.cv2.VideoCapture', return_value=Capture()), \
                 patch('src.adaptive_frame_extractor.cv2.calcOpticalFlowFarneback', return_value=flow), \
                 patch('src.adaptive_frame_extractor.assign_tiers', return_value=['static'] * 60):
                result = extract_adaptive_frames(str(source), str(Path(folder) / 'frames'), {'analysis_width': 64})
            self.assertLessEqual(len(result['frames']), 4)
            self.assertEqual(result['frames'][-1]['index'], 60)
            self.assertTrue(all('capture_reason' in f for f in result['frames']))
            with patch('src.adaptive_frame_extractor.cv2.VideoCapture', side_effect=AssertionError('不应重解码')):
                cached = extract_adaptive_frames(str(source), str(Path(folder) / 'frames'), {'analysis_width': 64})
            self.assertEqual(cached['frames'], result['frames'])
            (Path(folder) / 'frames' / result['frames'][0]['file']).unlink()
            with patch('src.adaptive_frame_extractor.cv2.VideoCapture', side_effect=RuntimeError('触发重建')):
                with self.assertRaisesRegex(RuntimeError, '触发重建'):
                    extract_adaptive_frames(str(source), str(Path(folder) / 'frames'), {'analysis_width': 64})


class TestMotionSummary(unittest.TestCase):
    CFG = DEFAULTS

    def test_subject_speed_bands(self):
        cases = [
            ([0.1, 0.2, 0.3], "static"),
            ([0.4, 0.6, 0.7], "slow"),
            ([1.0, 1.2, 1.4], "normal"),
            ([1.6, 2.0, 2.4], "fast"),
            ([3.0, 4.0, 5.0], "fast"),
            ([6.0, 7.0, 8.0], "very_fast"),
        ]
        for flows, expected in cases:
            m = motion_summary(flows, [TIER_NORMAL] * len(flows), self.CFG)
            self.assertEqual(m["subject_speed"], expected)

    def test_dominant_tier(self):
        tiers = [TIER_STATIC] * 8 + [TIER_NORMAL] * 2
        m = motion_summary([0.1] * 10, tiers, self.CFG)
        self.assertEqual(m["dominant_tier"], TIER_STATIC)
        self.assertEqual(m["tier_frame_counts"][TIER_STATIC], 8)
        # 五档计数键齐全
        for tier in (TIER_STATIC, TIER_SLOW, TIER_NORMAL, TIER_FAST, TIER_VERY_FAST):
            self.assertIn(tier, m["tier_frame_counts"])


if __name__ == "__main__":
    unittest.main()
