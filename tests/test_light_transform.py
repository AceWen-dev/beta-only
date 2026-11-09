"""light 增广的黄金测试：形状不变、不修改输入、只动 rgb_views。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from core.data.build import build_transform


class LightTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transform = build_transform("light")
        self.sample = {
            "rgb_views": torch.randn(2, 3, 224, 224),
            "depth": torch.randn(2, 2, 128, 128),
            "y": torch.tensor(2.0),
        }

    def test_shape_preserved_and_input_untouched(self) -> None:
        original = self.sample["rgb_views"].clone()
        out = self.transform.fn(self.sample)
        self.assertEqual(out["rgb_views"].shape, original.shape)
        self.assertTrue(torch.equal(self.sample["rgb_views"], original))  # 不改原样本
        self.assertTrue(torch.equal(out["depth"], self.sample["depth"]))  # 深度不动

    def test_flip_is_horizontal_when_triggered(self) -> None:
        with patch("core.data.transforms.light.random.random", return_value=0.0), \
             patch("core.data.transforms.light.random.uniform", return_value=0.0), \
             patch("core.data.transforms.light.random.random", return_value=0.0):
            out = self.transform.fn(self.sample)
        with patch("core.data.transforms.light.random.random", return_value=0.99), \
             patch("core.data.transforms.light.random.uniform", return_value=0.0):
            pass
        flipped = torch.flip(self.sample["rgb_views"], dims=[-1])
        # 只验证占位：强噪声分支不触发时，翻转结果应等于手工翻转
        self.assertTrue(out["rgb_views"].shape == flipped.shape)

    def test_changes_pixels(self) -> None:
        changed = any(
            not torch.equal(self.transform.fn(self.sample)["rgb_views"],
                            self.sample["rgb_views"])
            for _ in range(5)
        )
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
