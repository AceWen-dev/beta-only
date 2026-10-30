"""Benchmark manifest construction from the frozen test split.

The test membership is part of ``DataSpec.splits``; the metric protocol comes
from ``ProjectConfig.benchmark``. Both are hashed into the benchmark id, so
changing the test set, eval transform, metric version or evaluator version
creates a new benchmark instead of silently changing an old one.
"""

from __future__ import annotations

from dlef.benchmark import BenchmarkSpec
from dlef.contracts import DataSpec

from ..config import ProjectConfig


def build_benchmark(config: ProjectConfig, test_spec: DataSpec) -> BenchmarkSpec:
    try:
        membership_digest = test_spec.splits["test"]
    except KeyError as exc:
        raise ValueError("DataSpec has no 'test' split; cannot build a benchmark") from exc

    return BenchmarkSpec(
        name=config.benchmark.name,
        dataset=config.data.name,
        dataset_version=config.data.version,
        split="test",
        membership_digest=membership_digest,
        label_version=config.benchmark.label_version,
        transform=config.data.eval_transform,
        metrics={
            name: config.benchmark.metric_versions[name] for name in config.metrics
        },
        aggregation=config.benchmark.aggregation,
        thresholds=dict(config.benchmark.thresholds),
        evaluator_version=config.benchmark.evaluator_version,
    )
