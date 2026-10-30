from __future__ import annotations

from typing import Callable, Sequence

from dlef.contracts import Metric
from .mae import build_mae
from .v11 import build_v11_metrics

BUILDERS: dict[str, Callable[[], Metric]] = {
    "mae": build_mae,
    "v11": build_v11_metrics,
}


def build_metrics(names: Sequence[str]) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    for name in names:
        try:
            builder = BUILDERS[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown metric: {name!r}; available: {sorted(BUILDERS)}"
            ) from exc
        metrics[name] = builder()
    return metrics
