from __future__ import annotations

from adapters.torch import TorchMetric
from dlef.contracts import Metric


def build_mae() -> Metric:
    return TorchMetric(
        lambda preds, targets: {"mae": (preds - targets).abs().mean()}
    )
