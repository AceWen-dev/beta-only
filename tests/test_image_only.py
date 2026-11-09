from __future__ import annotations

import unittest

try:
    import torch

    from core.config import ModelConfig
    from dlef.config import ConfigError
    from core.models.v11 import V11Net

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _config(**kwargs) -> ModelConfig:
    base = dict(
        name="v11", views=1, d_model=64, beta_ref=0.016903,
        truth_median=5.0, k_bound=2.06, fusion="none", image_only=True,
    )
    base.update(kwargs)
    return ModelConfig(**base)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ImageOnlyTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        _config().validate()
        for bad in (
            _config(fusion="gate"),
            _config(direct_scale_anchor=True),
            _config(size_input=True),
            _config(ordinal_head=True),
        ):
            with self.assertRaises(ConfigError):
                bad.validate()

    def test_forward_and_state(self) -> None:
        torch.manual_seed(0)
        net = V11Net(_config()).eval()
        state = net.state_dict()
        self.assertTrue(any(k.startswith("head_img.") for k in state))
        self.assertFalse(any(k.startswith("head_beta.") for k in state))
        self.assertFalse(any(k.startswith("head_dir.") for k in state))
        self.assertFalse(any(k.startswith("gate.") for k in state))
        batch = {"rgb_views": torch.randn(2, 1, 3, 224, 224), "D": torch.ones(2)}
        with torch.no_grad():
            pred, pooled = net(batch)
        self.assertEqual(tuple(pred.shape), (2,))
        self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
