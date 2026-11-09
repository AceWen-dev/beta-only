from __future__ import annotations

import math
import unittest

import torch

from core.config import ConfigError, ModelConfig, ProjectConfig
from core.models.v11 import (
    anchor_k,
    focal_beta,
    focal_v2_scale,
    metric_depth_tokens,
)


class MetricDepthTokenTests(unittest.TestCase):
    def test_reconstructs_mm_and_zeroes_invalid(self) -> None:
        depth = torch.zeros(1, 2, 2, 4, 4)
        depth[:, :, 0] = 1.0
        depth[:, :, 1] = 1.0
        out = metric_depth_tokens(
            depth, torch.tensor([20.0]), torch.tensor([0.5]), ref_mm=20.0
        )
        # global_q = 20 / 0.5 = 40 mm；token 1.0 → log(40/20)
        self.assertAlmostEqual(float(out[0, 0, 0, 0, 0]), math.log(2.0), places=5)
        # 无效像素的深度通道必须归零，不能变成 log(1e-3/20) 的强“近”信号
        depth[0, 0, 0, 0, 0] = 0.0
        depth[0, 0, 1, 0, 0] = 0.0
        out = metric_depth_tokens(
            depth, torch.tensor([20.0]), torch.tensor([0.5]), ref_mm=20.0
        )
        self.assertEqual(float(out[0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 0, 1, 0, 0]), 0.0)

    def test_clips_metric_range(self) -> None:
        depth = torch.ones(1, 1, 2, 1, 1)
        depth[:, :, 0] = 4.0
        out = metric_depth_tokens(
            depth, torch.tensor([20.0]), torch.tensor([1.0]), ref_mm=1.0
        )
        # log(80/1) ≈ 4.38 → clip 到 2.0
        self.assertEqual(float(out[0, 0, 0, 0, 0]), 2.0)


class MetricAnchorTests(unittest.TestCase):
    def test_zero_residual_starts_at_physical(self) -> None:
        k = anchor_k(
            torch.zeros(3), torch.zeros(3), 0.000769, 1.0, 0.3
        )
        self.assertTrue(torch.allclose(k, torch.full((3,), 0.000769)))

    def test_residual_and_physical_are_bounded(self) -> None:
        z = torch.tensor([100.0, -100.0])
        k = anchor_k(z, z, 0.000769, 1.0, 0.3)
        self.assertAlmostEqual(float(k[0]), 0.000769 * math.exp(1.3), places=8)
        self.assertAlmostEqual(float(k[1]), 0.000769 * math.exp(-1.3), places=8)


class MetricFocalHeadTests(unittest.TestCase):
    def test_zero_head_starts_at_physical(self) -> None:
        beta = focal_beta(torch.zeros(3), 0.000769, 0.15)
        self.assertTrue(torch.allclose(beta, torch.full((3,), 0.000769)))

    def test_focal_correction_is_bounded(self) -> None:
        z = torch.tensor([100.0, -100.0])
        beta = focal_beta(z, 0.000769, 0.15)
        self.assertAlmostEqual(
            float(beta[0]), 0.000769 * math.exp(-0.15), places=10
        )
        self.assertAlmostEqual(
            float(beta[1]), 0.000769 * math.exp(0.15), places=10
        )


class MetricFocalV2Tests(unittest.TestCase):
    def test_zero_heads_return_physical_baseline(self) -> None:
        scale = focal_v2_scale(torch.zeros(3), torch.zeros(3), 2.0, 0.3)
        self.assertTrue(torch.allclose(scale, torch.ones(3)))

    def test_geometry_and_appearance_are_bounded(self) -> None:
        z = torch.tensor([100.0, -100.0])
        zero = torch.zeros(2)
        geo = focal_v2_scale(z, zero, 2.0, 0.3)
        self.assertAlmostEqual(float(geo[0]), math.exp(-2.0), places=6)
        self.assertAlmostEqual(float(geo[1]), math.exp(2.0), places=6)
        both = focal_v2_scale(z, z, 2.0, 0.3)
        self.assertAlmostEqual(float(both[0]), math.exp(-2.0 + 0.3), places=6)
        self.assertAlmostEqual(float(both[1]), math.exp(2.0 - 0.3), places=6)


class MetricDepthConfigTests(unittest.TestCase):
    def test_requires_beta_only(self) -> None:
        ModelConfig(name="v11", fusion="none", metric_depth=True).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(name="v11", fusion="gate", metric_depth=True).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", image_only=True, metric_depth=True
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True, metric_ref_mm=0
            ).validate()

    def test_anchor_requires_metric_depth_and_beta_only(self) -> None:
        ModelConfig(
            name="v11", fusion="none", metric_depth=True, metric_anchor=True
        ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(name="v11", fusion="none", metric_anchor=True).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="gate", metric_depth=True, metric_anchor=True
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True, metric_anchor=True,
                metric_residual_bound=0,
            ).validate()

    def test_focal_head_requires_metric_beta_only(self) -> None:
        ModelConfig(
            name="v11", fusion="none", metric_depth=True,
            metric_focal_head=True,
        ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_focal_head=True
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="gate", metric_depth=True,
                metric_focal_head=True,
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True,
                metric_anchor=True, metric_focal_head=True,
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True,
                metric_focal_head=True, metric_focal_bound=0,
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True,
                metric_focal_head=True, metric_focal_bound=1.5,
            ).validate()

    def test_focal_v2_requires_metric_beta_only(self) -> None:
        ModelConfig(
            name="v11", fusion="none", metric_depth=True,
            metric_focal_v2=True,
        ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_focal_v2=True
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="gate", metric_depth=True,
                metric_focal_v2=True,
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True,
                metric_anchor=True, metric_focal_v2=True,
            ).validate()
        with self.assertRaises(ConfigError):
            ModelConfig(
                name="v11", fusion="none", metric_depth=True,
                metric_focal_head=True, metric_focal_v2=True,
            ).validate()
        for kwargs in (
            {"metric_focal_phys_bound": 0},
            {"metric_focal_phys_bound": 3.5},
            {"metric_focal_app_bound": 0},
            {"metric_focal_app_bound": 1.5},
            {"metric_focal_width_ref": 0},
            {"metric_focal_height_ref": 0},
        ):
            with self.assertRaises(ConfigError):
                ModelConfig(
                    name="v11", fusion="none", metric_depth=True,
                    metric_focal_v2=True, **kwargs,
                ).validate()

    def test_requires_endo_data_version(self) -> None:
        config = ProjectConfig()
        config.model.name = "v11"
        config.model.fusion = "none"
        config.model.metric_depth = True
        config.data.version = "company-v1"
        with self.assertRaises(ConfigError):
            config.validate()
        config.data.version = "company-v1-endo"
        config.validate()


if __name__ == "__main__":
    unittest.main()
