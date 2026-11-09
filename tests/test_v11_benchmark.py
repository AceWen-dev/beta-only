from __future__ import annotations

import os
import unittest
from pathlib import Path

try:
    from dlef.benchmark import BenchmarkSpec
    from dlef.config import load_config
    from core.config import ProjectConfig
    from core.data import build_benchmark, build_dataset

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

BENCHMARKS = {
    "v11-b-test-v2.json": "b",
    "v11-d-test-v2.json": "d",
    "v11-r-test-v2.json": "r",
    "v11-clean-test-v2.json": "clean",
}


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
@unittest.skipUnless(
    MANIFEST is not None and MANIFEST.is_file(),
    "set DLEF_DATA_ROOT with the v11_frames single-v2 manifest to test benchmarks",
)
class V11BenchmarkTests(unittest.TestCase):
    def test_frozen_benchmarks_match_current_data_and_protocol(self) -> None:
        base = ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
        seen: set[str] = set()
        for filename, test_set in BENCHMARKS.items():
            path = ROOT / "input" / "benchmarks" / filename
            with self.subTest(benchmark=filename):
                config = load_config(
                    base,
                    ProjectConfig,
                    overrides=[
                        {
                            "data": {"test_set": test_set},
                            "benchmark": {"name": path.stem},
                        }
                    ],
                ).schema
                test_spec = build_dataset(
                    config.data, "test", config.data.eval_transform
                ).spec
                current = build_benchmark(config, test_spec)
                stored = BenchmarkSpec.load(path)

                self.assertEqual(current.test_set_id, stored.test_set_id)
                self.assertEqual(current.benchmark_id, stored.benchmark_id)
                self.assertNotIn(current.benchmark_id, seen)
                seen.add(current.benchmark_id)

    def test_membership_ignores_training_seed(self) -> None:
        base = ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
        first = load_config(base, ProjectConfig, overrides=[{"seed": 1}]).schema
        second = load_config(base, ProjectConfig, overrides=[{"seed": 2}]).schema
        first_spec = build_dataset(first.data, "test", first.data.eval_transform).spec
        second_spec = build_dataset(second.data, "test", second.data.eval_transform).spec
        self.assertEqual(first_spec.splits["test"], second_spec.splits["test"])


if __name__ == "__main__":
    unittest.main()
