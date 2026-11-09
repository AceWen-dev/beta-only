"""D-0055：7×7 背景空间分支（网格投影 + 框关系 + 交叉注意力）。"""

from __future__ import annotations

import unittest

import torch

from core.config import ConfigError, ModelConfig
from core.models.v11 import V11Net


class BoxGridTests(unittest.TestCase):
    def test_box_maps_to_expected_cells(self):
        # S=100px（框最大边），a=p/W=0.5 → W=H=200px；框宽高各占 50%，居中
        batch = {
            "S": torch.tensor([100.0]),
            "scalars": torch.tensor(
                [
                    [
                        0.5,  # a = p/W
                        0.5,  # bw/W
                        0.5,  # bh/H
                        0.5,  # cx/W
                        0.5,  # cy/H
                        1.0,  # H/W
                        0.0, 0.0, 0.0, 0.0,
                    ]
                ]
            ),
        }
        feats = V11Net._box_features(batch)
        self.assertEqual(tuple(feats.shape), (1, 49, 5))
        inside = feats[0, :, 0]
        # 224/200 放大后框落在 [56,168]，覆盖中间 3×3 个格子
        self.assertAlmostEqual(float(inside.sum()), 9.0)
        self.assertAlmostEqual(float(feats[0, 0, 3]), 0.5, places=5)  # 框宽/224
        self.assertAlmostEqual(float(feats[0, 0, 4]), 0.5, places=5)

    def test_center_offsets_are_signed(self):
        batch = {
            "S": torch.tensor([100.0]),
            "scalars": torch.tensor(
                [[0.5, 0.5, 0.5, 0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
        }
        feats = V11Net._box_features(batch)
        # 左上角格子相对框中心应为负偏移
        self.assertLess(float(feats[0, 0, 1]), 0.0)
        self.assertLess(float(feats[0, 0, 2]), 0.0)


class SpatialContextModelTests(unittest.TestCase):
    def test_forward_shapes_with_spatial_context(self):
        net = V11Net(ModelConfig(name="v11", spatial_context=True, views=2))
        net.eval()
        n = 1
        batch = {
            "rgb_views": torch.zeros(n, 2, 3, 224, 224),
            "depth": torch.zeros(n, 2, 2, 128, 128),
            "scalars": torch.zeros(n, 10),
            "D": torch.tensor([0.4]),
            "S": torch.tensor([120.0]),
        }
        batch["scalars"][:, 0] = 0.4  # a=p/W
        batch["scalars"][:, 5] = 1.0  # H/W
        with torch.no_grad():
            pred, pooled, branches = net(batch, return_branches=True)
        self.assertEqual(tuple(pred.shape), (n,))
        self.assertEqual(tuple(pooled.shape), (n, 64))
        self.assertEqual(tuple(branches["pred_beta"].shape), (n,))

    def test_config_rejects_spatial_context_for_other_models(self):
        with self.assertRaises(ConfigError):
            ModelConfig(name="mlp", spatial_context=True).validate()


if __name__ == "__main__":
    unittest.main()
