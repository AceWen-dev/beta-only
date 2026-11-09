"""span_calibration 的黄金测试：修正后 k=truth/S' 应跨尺寸恒定。"""

from __future__ import annotations

import unittest

from core.data.datasets.v11_frames import fit_span_calibration, span_scale_factor


def make_rows() -> list[dict]:
    """两档尺寸，k=truth/S 分别为 0.015 / 0.022（模拟小息肉框相对偏大）。"""
    rows = []
    for i in range(120):
        span = 40.0 + i  # 40..159 px
        k = 0.015 if span < 80 else 0.022
        rows.append({
            "split": "train",
            "bbox_content": [0.0, 0.0, span, span * 0.8],
            "truth_mm": k * span,
        })
    for _ in range(40):  # 非 train 行不参与拟合
        rows.append({
            "split": "val",
            "bbox_content": [0.0, 0.0, 50.0, 45.0],
            "truth_mm": 1.0,
        })
    return rows


class SpanCalibrationTests(unittest.TestCase):
    def test_fit_uses_train_rows_and_k_ref_matches_dominant_bin(self) -> None:
        cal = fit_span_calibration(make_rows())
        self.assertTrue(cal)
        self.assertAlmostEqual(cal["k_ref"], 0.022, places=6)
        self.assertLess(min(cal["k"][:3]), 0.018)  # 小尺寸档的 k 更低

    def test_correction_equalizes_k_across_sizes(self) -> None:
        cal = fit_span_calibration(make_rows())
        for span, k in ((50.0, 0.015), (120.0, 0.022)):
            s_prime = span * span_scale_factor(span, cal)
            k_prime = k * span / s_prime  # 修正后的 truth/S'
            self.assertAlmostEqual(k_prime, cal["k_ref"], places=3)

    def test_disabled_calibration_returns_identity(self) -> None:
        self.assertEqual(span_scale_factor(50.0, None), 1.0)
        self.assertEqual(span_scale_factor(50.0, {}), 1.0)


if __name__ == "__main__":
    unittest.main()
