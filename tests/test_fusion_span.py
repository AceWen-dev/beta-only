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
        name="v11",
        views=2,
        d_model=64,
        beta_ref=0.016903,
        truth_median=5.0,
        k_bound=2.06,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def _batch() -> dict:
    b, v = 2, 2
    return {
        "rgb_views": torch.randn(b, v, 3, 224, 224),
        "depth": torch.randn(b, v, 2, 224, 224),
        "scalars": torch.randn(b, 10),
        "D": torch.full((b,), 0.37),
        "S": torch.full((b,), 300.0),
        "S_norm": torch.full((b,), 0.25),
        "W": torch.full((b,), 1200.0),
    }


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class FusionSpanTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        _config(fusion="none").validate()
        _config(span_input=True).validate()
        for bad in (
            _config(fusion="nope"),
            _config(fusion="none", direct_scale_anchor=True),
            _config(fusion="none", ordinal_head=True),
            _config(span_input=True, size_input=True),
            _config(span_input=True, ordinal_head=True),
        ):
            with self.assertRaises(ConfigError):
                bad.validate()

    def test_beta_only_matches_beta_branch_at_init(self) -> None:
        torch.manual_seed(0)
        gate = V11Net(_config()).eval()
        beta = V11Net(_config(fusion="none")).eval()
        missing, unexpected = beta.load_state_dict(gate.state_dict(), strict=False)
        self.assertEqual(missing, [])
        self.assertTrue(any(k.startswith("head_dir.") for k in unexpected))
        self.assertTrue(any(k.startswith("gate.") for k in unexpected))
        batch = _batch()
        with torch.no_grad():
            out_beta, _ = beta(batch)
            _, _, aux = gate(batch, return_branches=True)
        self.assertTrue(torch.allclose(out_beta, aux["pred_beta"], atol=1e-6))

    def test_span_input_is_zero_init(self) -> None:
        torch.manual_seed(1)
        base = V11Net(_config()).eval()
        span = V11Net(_config(span_input=True)).eval()
        missing, unexpected = span.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(any(k.startswith("span_proj.") for k in missing))
        self.assertEqual(unexpected, [])
        batch = _batch()
        with torch.no_grad():
            out_base, _ = base(batch)
            out_span, _ = span(batch)
        self.assertTrue(torch.allclose(out_base, out_span, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
