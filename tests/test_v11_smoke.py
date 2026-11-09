from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from torch import nn

    from adapters.torch import TorchCheckpointCodec
    from core.config import ProjectConfig
    from core.data import make_loaders
    from core.runtime import build_policy, build_runner
    from core.task import build_task
    from dlef.config import load_config
    from dlef.eval import EvalSpec

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("DLEF_DATA_ROOT")
MANIFEST = (
    Path(DATA_ROOT) / "curated" / "v11_frames" / "single-v2" / "manifest.json"
    if DATA_ROOT
    else None
)


class StubBackbone(nn.Module):
    """Cheap 2048-d feature extractor; no ImageNet download in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Identity()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(1, 2, 3)).unsqueeze(1)
        return self.scale * pooled.repeat(1, 2048)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
@unittest.skipUnless(
    MANIFEST is not None and MANIFEST.is_file(),
    "set DLEF_DATA_ROOT with the v11_frames single-v2 manifest to run the smoke test",
)
class V11SmokeTests(unittest.TestCase):
    def _config(self, **overrides):
        base = ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
        values = {
            "seed": 123,
            "epochs": 1,
            "max_steps": 1,
            "log_every": 0,
            "checkpoint_every": 0,
            "early_stopping": {"enabled": False},
            "tensorboard": {"enabled": False},
            "data": {"max_samples": 2, "batch_size": 1},
            "model": {"init_checkpoint": "", "backbone": "full"},
        }
        values.update(overrides)
        return load_config(base, ProjectConfig, overrides=[values])

    def _run_one_step(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(Path, "home", return_value=Path(tmp) / "home"),
                patch("torchvision.models.resnet50", return_value=StubBackbone()),
            ):
                train_loader, val_loader = make_loaders(
                    config.schema.data, config.schema.seed
                )
                task = build_task(config.schema, device="cpu")
                policy = build_policy(config.schema)
                runner = build_runner(Path(tmp), policy)
                run = runner.fit(task, train_loader, val_loader, config)
                record = runner.evaluate(
                    task,
                    val_loader,
                    EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                    run,
                )

                self.assertEqual(run.manifest.status, "completed")
                self.assertTrue(run.checkpoint_path.is_file())
                self.assertIn("v11/mae", record.metrics)
                payload = TorchCheckpointCodec().load(run.checkpoint_path)
                self.assertIn("python", payload.rng_state)
                self.assertIn("torch", payload.rng_state)
                return run, record

    def test_baseline_one_step(self) -> None:
        self._run_one_step(self._config())

    def test_ranksim_one_step_uses_aux_features(self) -> None:
        config = self._config(
            data={"max_samples": 2, "batch_size": 2},
            model={"init_checkpoint": "", "backbone": "full", "proj": True},
            loss={"name": "mm_rank"},
        )
        self._run_one_step(config)


if __name__ == "__main__":
    unittest.main()
