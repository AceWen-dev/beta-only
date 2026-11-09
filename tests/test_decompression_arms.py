"""D-0067：解压缩两臂（gate_beta_floor / direct_scale_anchor）的形状与初值测试。

两臂都不新增参数，默认关闭；本文件锁定三条契约：
1. 关闭时与旧模型逐位一致（旧 checkpoint 可直接加载）；
2. gate_beta_floor 只按 wg = floor + (1−floor)·sigmoid(logit) 改变门控；
3. direct_scale_anchor 的初值与 β 分支相同（exp(log beta_ref)·D·S）。
"""

from __future__ import annotations

import unittest

import torch

from core.config import ConfigError, ModelConfig
from core.models.v11 import V11Net


def make_batch(n: int = 1) -> dict:
    batch = {
        "rgb_views": torch.zeros(n, 2, 3, 224, 224),
        "depth": torch.zeros(n, 2, 2, 128, 128),
        "scalars": torch.zeros(n, 10),
        "D": torch.full((n,), 0.4),
        "S": torch.full((n,), 120.0),
    }
    batch["scalars"][:, 0] = 0.4
    batch["scalars"][:, 5] = 1.0
    return batch


def build(**kw) -> V11Net:
    net = V11Net(ModelConfig(name="v11", views=2, **kw))
    return net.eval()


class GateFloorTests(unittest.TestCase):
    def test_no_new_parameters(self):
        base, arm = build(), build(gate_beta_floor=0.5)
        self.assertEqual(sorted(base.state_dict()), sorted(arm.state_dict()))
        arm.load_state_dict(base.state_dict())  # 旧 checkpoint 必须能直接加载

    def test_floor_zero_is_bitwise_identical(self):
        base, arm = build(), build(gate_beta_floor=0.0)
        arm.load_state_dict(base.state_dict())
        batch = make_batch()
        with torch.no_grad():
            p0, _, b0 = base(batch, return_branches=True)
            p1, _, b1 = arm(batch, return_branches=True)
        self.assertTrue(torch.equal(p0, p1))
        self.assertTrue(torch.equal(b0["gate"], b1["gate"]))

    def test_floor_transforms_gate_as_specified(self):
        base, arm = build(), build(gate_beta_floor=0.5)
        arm.load_state_dict(base.state_dict())
        batch = make_batch()
        with torch.no_grad():
            _, _, b0 = base(batch, return_branches=True)
            _, _, b1 = arm(batch, return_branches=True)
        wg0 = float(b0["gate"])
        wg1 = float(b1["gate"])
        self.assertGreater(wg0, 0.5)  # 初值 sigmoid(2.0)=0.88
        self.assertAlmostEqual(wg1, 0.5 + 0.5 * wg0, places=6)
        self.assertGreaterEqual(wg1, 0.5)


class DirectAnchorTests(unittest.TestCase):
    def test_no_new_parameters(self):
        base, arm = build(), build(direct_scale_anchor=True)
        self.assertEqual(sorted(base.state_dict()), sorted(arm.state_dict()))

    def test_anchored_init_matches_beta_branch(self):
        arm = build(direct_scale_anchor=True)
        batch = make_batch()
        with torch.no_grad():
            _, _, b = arm(batch, return_branches=True)
        expect = arm.beta_ref * 0.4 * 120.0
        self.assertAlmostEqual(float(b["pred_beta"]), expect, places=5)
        self.assertAlmostEqual(float(b["pred_direct"]), expect, places=5)

    def test_default_branch_is_softplus_truth_median(self):
        base = build()
        batch = make_batch()
        with torch.no_grad():
            _, _, b = base(batch, return_branches=True)
        self.assertAlmostEqual(float(b["pred_direct"]), base.truth_median, places=5)


class ConfigValidationTests(unittest.TestCase):
    def test_floor_range(self):
        for bad in (-0.1, 1.0, 2.0):
            with self.assertRaises(ConfigError, msg=f"floor={bad}"):
                ModelConfig(name="v11", gate_beta_floor=bad).validate()
        ModelConfig(name="v11", gate_beta_floor=0.5).validate()

    def test_arms_are_v11_only(self):
        with self.assertRaises(ConfigError):
            ModelConfig(name="mlp", gate_beta_floor=0.5).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(name="mlp", direct_scale_anchor=True).validate()


if __name__ == "__main__":
    unittest.main()
