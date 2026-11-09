from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from dlef.benchmark import BenchmarkSpec
    from core.config import BenchmarkConfig, DataConfig, ProjectConfig
    from core.data import build_benchmark, build_dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def make_config(
    *,
    n_train: int = 8,
    n_val: int = 4,
    n_test: int = 2,
    version: str = "synthetic",
    metric_version: str = "v1",
) -> ProjectConfig:
    return ProjectConfig(
        data=DataConfig(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            version=version,
        ),
        benchmark=BenchmarkConfig(metric_versions={"mae": metric_version}),
        metrics=["mae"],
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class BenchmarkTests(unittest.TestCase):
    def test_benchmark_ids_are_stable_and_separated(self) -> None:
        config = make_config()
        test_spec = build_dataset(config.data, "test", "identity").spec
        benchmark = build_benchmark(config, test_spec)

        self.assertTrue(benchmark.test_set_id.startswith("sha256:"))
        self.assertTrue(benchmark.benchmark_id.startswith("sha256:"))

        # Growing train does not change the frozen test set.
        bigger = make_config(n_train=16)
        bigger_test = build_dataset(bigger.data, "test", "identity").spec
        bigger_benchmark = build_benchmark(bigger, bigger_test)
        self.assertEqual(benchmark.test_set_id, bigger_benchmark.test_set_id)
        self.assertEqual(benchmark.benchmark_id, bigger_benchmark.benchmark_id)

        # Changing the test membership changes both ids.
        more_test = make_config(n_test=3)
        more_test_spec = build_dataset(more_test.data, "test", "identity").spec
        more_test_benchmark = build_benchmark(more_test, more_test_spec)
        self.assertNotEqual(benchmark.test_set_id, more_test_benchmark.test_set_id)
        self.assertNotEqual(benchmark.benchmark_id, more_test_benchmark.benchmark_id)

        # Changing only the metric protocol changes only the benchmark id.
        metric_v2 = make_config(metric_version="v2")
        metric_test_spec = build_dataset(metric_v2.data, "test", "identity").spec
        metric_benchmark = build_benchmark(metric_v2, metric_test_spec)
        self.assertEqual(benchmark.test_set_id, metric_benchmark.test_set_id)
        self.assertNotEqual(benchmark.benchmark_id, metric_benchmark.benchmark_id)

    def test_save_and_load(self) -> None:
        config = make_config()
        test_spec = build_dataset(config.data, "test", "identity").spec
        benchmark = build_benchmark(config, test_spec)

        with tempfile.TemporaryDirectory() as tmp:
            path = benchmark.save(Path(tmp) / "benchmark.json")
            loaded = BenchmarkSpec.load(path)
            self.assertEqual(loaded.benchmark_id, benchmark.benchmark_id)
            self.assertEqual(loaded.test_set_id, benchmark.test_set_id)
            self.assertEqual(loaded.name, "synthetic-test-v1")


if __name__ == "__main__":
    unittest.main()
