"""zoom_content 的黄金测试：图与框同步缩放，S 跟随 α。"""

from __future__ import annotations

import unittest

import numpy as np

from core.data.datasets.v11_frames import zoom_content


class ZoomContentTests(unittest.TestCase):
    def test_scales_content_and_bbox_together(self) -> None:
        content = np.zeros((100, 200, 3), np.uint8)
        content[40:60, 80:120] = 255
        bbox = [80.0, 40.0, 120.0, 60.0]
        out, zb = zoom_content(content, bbox, 2.0)
        self.assertEqual(out.shape, (200, 400, 3))
        self.assertEqual(zb, [160.0, 80.0, 240.0, 120.0])
        span_after = max(zb[2] - zb[0], zb[3] - zb[1])
        self.assertAlmostEqual(span_after, 2.0 * 40.0, places=6)

    def test_shrink_uses_area_interpolation_and_positive_size(self) -> None:
        content = np.zeros((50, 50, 3), np.uint8)
        out, zb = zoom_content(content, [10.0, 10.0, 30.0, 30.0], 0.5)
        self.assertEqual(out.shape, (25, 25, 3))
        self.assertTrue(all(v > 0 for v in zb))

    def test_identity_alpha_returns_inputs(self) -> None:
        content = np.zeros((10, 10, 3), np.uint8)
        bbox = [1.0, 2.0, 3.0, 4.0]
        out, zb = zoom_content(content, bbox, 1.0)
        self.assertIs(out, content)
        self.assertIs(zb, bbox)


if __name__ == "__main__":
    unittest.main()
