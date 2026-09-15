# -*- coding: utf-8 -*-
"""自适应抽帧器纯函数单元测试（normalize_sims / assign_tiers / motion_summary）"""
import unittest

from src.adaptive_frame_extractor import (
    DEFAULTS,
    TIER_FAST,
    TIER_MEDIUM,
    TIER_STATIC,
    assign_tiers,
    motion_summary,
    normalize_sims,
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

    def test_constant_clip_falls_back_medium(self):
        # 全片无变化（MAD=0）时退化为全 0.5，避免除零/档位翻转
        norm = normalize_sims([0.95] * 20)
        self.assertEqual(norm, [0.5] * 20)

    def test_empty(self):
        self.assertEqual(normalize_sims([]), [])


class TestAssignTiers(unittest.TestCase):
    CFG = {**DEFAULTS, "min_tier_hold": 0.3}

    @staticmethod
    def _times(n):
        return [float(i) * 0.1 for i in range(n)]

    def test_static_when_high_sim(self):
        times = self._times(20)
        tiers = assign_tiers(times, [0.95] * 20, [0.5] * 20, self.CFG)
        self.assertEqual(set(tiers), {TIER_STATIC})

    def test_fast_when_low_sim(self):
        times = self._times(20)
        tiers = assign_tiers(times, [0.5] * 20, [0.5] * 20, self.CFG)
        self.assertEqual(set(tiers), {TIER_FAST})

    def test_fast_when_relative_drop(self):
        # raw 在 0.7~0.9 中速区间，但片段内归一化后掉到 0.6 → 局部快变化
        times = self._times(20)
        sims = [0.85] * 10 + [0.8] * 10
        norm = [0.9] * 10 + [0.6] * 10
        tiers = assign_tiers(times, sims, norm, self.CFG)
        self.assertEqual(set(tiers[:10]), {TIER_MEDIUM})
        # 候选档从 t=1.0 起，满 0.3s 滞回后（t>=1.3，下标 13）晋升快档
        self.assertEqual(set(tiers[13:]), {TIER_FAST})

    def test_hysteresis_suppresses_brief_flip(self):
        # 0.2s 的瞬时掉档不应换档（候选档需持续满 min_tier_hold 0.3s 才晋升）
        times = self._times(20)
        sims = [0.95] * 10 + [0.6, 0.6] + [0.95] * 8
        tiers = assign_tiers(times, sims, [0.5] * 20, self.CFG)
        self.assertEqual(set(tiers), {TIER_STATIC})

    def test_sustained_change_switches_tier(self):
        times = self._times(20)
        sims = [0.95] * 5 + [0.6] * 15
        tiers = assign_tiers(times, sims, [0.5] * 20, self.CFG)
        self.assertEqual(tiers[0], TIER_STATIC)
        # 快档从 t=0.5 起持续，满 0.3s 滞回后（t>=0.8，下标 9）完成晋升
        self.assertEqual(set(tiers[9:]), {TIER_FAST})
        self.assertIn(TIER_STATIC, tiers[:9])


class TestMotionSummary(unittest.TestCase):
    CFG = DEFAULTS

    def test_subject_speed_bands(self):
        slow = motion_summary([0.1, 0.2, 0.3], [TIER_STATIC] * 3, self.CFG)
        self.assertEqual(slow["subject_speed"], "slow")
        mid = motion_summary([1.0, 1.2, 1.4], [TIER_MEDIUM] * 3, self.CFG)
        self.assertEqual(mid["subject_speed"], "medium")
        fast = motion_summary([3.0, 4.0, 5.0], [TIER_FAST] * 3, self.CFG)
        self.assertEqual(fast["subject_speed"], "fast")

    def test_dominant_tier(self):
        tiers = [TIER_STATIC] * 8 + [TIER_MEDIUM] * 2
        m = motion_summary([0.1] * 10, tiers, self.CFG)
        self.assertEqual(m["dominant_tier"], TIER_STATIC)
        self.assertEqual(m["tier_frame_counts"][TIER_STATIC], 8)


if __name__ == "__main__":
    unittest.main()
