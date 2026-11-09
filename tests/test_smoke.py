from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from dlef.config import resolve_config
from dlef.eval import EvalSpec

try:
    import torch  # noqa: F401

    from core.config import ProjectConfig
    from core.data import make_loaders
    from core.runtime import build_policy, build_runner
    from core.task import build_task

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorboard  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class SmokeTests(unittest.TestCase):
    def test_train_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = resolve_config(
                ProjectConfig,
                {
                    "seed": 1,
                    "epochs": 1,
                    "project": "smoke",
                    "experiment": "smoke",
                    "log_every": 0,
                    "data": {"n_train": 32, "n_val": 16, "batch_size": 8},
                    "model": {"hidden": 8},
                    "optim": {"lr": 0.02},
                    "amp": False,
                    "grad_accum_steps": 1,
                    "grad_clip": 1.0,
                },
            )
            policy = build_policy(config.schema)
            policy.setup()
            runner = build_runner(Path(tmp), policy)
            train_loader, val_loader = make_loaders(config.schema.data, config.schema.seed)
            task = build_task(config.schema, device=policy.device)
            run = runner.fit(
                task,
                train_loader,
                val_loader,
                config,
                tags={"baseline": "true"},
            )
            self.assertIsNotNone(run)
            self.assertEqual(run.manifest.status, "completed")
            self.assertEqual(run.manifest.tags["baseline"], "true")
            self.assertEqual(run.manifest.tags["framework"], "dlef")
            self.assertTrue(run.config_path.is_file())
            self.assertTrue(run.checkpoint_path.is_file())
            self.assertTrue(run.log_path.is_file())
            self.assertIn("Status: completed", run.log_path.read_text(encoding="utf-8"))
            if TENSORBOARD_AVAILABLE:
                events = list((run.dir / "tensorboard").glob("events.out.tfevents.*"))
                self.assertTrue(events, "TensorBoard event file was not written")

            record = runner.evaluate(
                task,
                val_loader,
                EvalSpec(checkpoint=str(run.checkpoint_path), split="val"),
                run,
            )
            self.assertIn("mae", record.metrics)
            self.assertTrue(math.isfinite(record.metrics["mae"]))


if __name__ == "__main__":
    unittest.main()
