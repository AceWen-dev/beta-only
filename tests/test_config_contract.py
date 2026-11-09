from __future__ import annotations

import unittest
from pathlib import Path

try:
    from dlef.config import ConfigError, load_config, resolve_config
    from core.config import ProjectConfig
    from core.data.build import DATASETS, TRANSFORMS
    from core.data.splits import SPLITTERS
    from core.losses.build import BUILDERS as LOSSES
    from core.metrics.build import BUILDERS as METRICS
    from core.models.build import BUILDERS as MODELS
    from core.optimizers.build import BUILDERS as OPTIMIZERS
    from core.runtime import resolve_output_root
    from core.schedulers.build import BUILDERS as SCHEDULERS

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class ConfigContractTests(unittest.TestCase):
    def test_every_selector_resolves_to_a_builder(self) -> None:
        config = load_config(str(ROOT / "input" / "configs" / "base.yaml"), ProjectConfig).schema

        self.assertIn(config.data.name, DATASETS)
        self.assertIn(config.data.split, SPLITTERS)
        self.assertIn(config.data.train_transform, TRANSFORMS)
        self.assertIn(config.data.eval_transform, TRANSFORMS)

        self.assertIn(config.model.name, MODELS)
        self.assertIn(config.loss.name, LOSSES)
        self.assertIn(config.optim.name, OPTIMIZERS)
        self.assertIn(config.scheduler.name, SCHEDULERS)
        for name in config.metrics:
            self.assertIn(name, METRICS)
            self.assertIn(name, config.benchmark.metric_versions)

    def test_v11_variants_resolve_to_builders(self) -> None:
        paths = sorted((ROOT / "input" / "configs" / "experiments" / "v11").glob("*.yaml"))
        self.assertTrue(paths, "no v11 experiment configs found")
        for path in paths:
            with self.subTest(config=path.name):
                config = load_config(path, ProjectConfig).schema
                self.assertEqual(config.data.name, "v11_frames")
                self.assertEqual(config.model.name, "v11")
                self.assertIn(config.data.split, SPLITTERS)
                self.assertIn(config.data.train_transform, TRANSFORMS)
                self.assertIn(config.data.eval_transform, TRANSFORMS)
                self.assertIn(config.model.name, MODELS)
                self.assertIn(config.loss.name, LOSSES)
                self.assertIn(config.optim.name, OPTIMIZERS)
                self.assertIn(config.scheduler.name, SCHEDULERS)
                for name in config.metrics:
                    self.assertIn(name, METRICS)
                    self.assertIn(name, config.benchmark.metric_versions)

    def test_ranksim_variants_start_from_imagenet(self) -> None:
        for name in ("ranksim-best", "proj-best"):
            with self.subTest(config=name):
                path = ROOT / "input" / "configs" / "experiments" / "v11" / f"{name}.yaml"
                config = load_config(path, ProjectConfig).schema
                self.assertEqual(config.model.init_checkpoint, "")
                self.assertEqual(config.loss.name, "mm_rank")

    def test_output_root_and_tensorboard_validate(self) -> None:
        with self.assertRaises(ConfigError):
            resolve_config(ProjectConfig, {"output_root": ""})
        with self.assertRaises(ConfigError):
            resolve_config(ProjectConfig, {"tensorboard": {"subdir": ""}})
        with self.assertRaises(ConfigError):
            resolve_config(ProjectConfig, {"tensorboard": {"flush_secs": 0}})

    def test_output_root_resolution(self) -> None:
        relative = resolve_config(
            ProjectConfig,
            {"output_root": "artifacts"},
        )
        self.assertEqual(
            resolve_output_root(relative.schema, "/tmp/project"),
            Path("/tmp/project/artifacts"),
        )
        absolute = resolve_config(
            ProjectConfig,
            {"output_root": "/tmp/absolute"},
        )
        self.assertEqual(
            resolve_output_root(absolute.schema, "/tmp/project"),
            Path("/tmp/absolute"),
        )


if __name__ == "__main__":
    unittest.main()
