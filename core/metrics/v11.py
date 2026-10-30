"""v11 指标：总体/分层 MAE、10 mm 一致率、最差 1 mm 区间。"""

from __future__ import annotations

import torch

from adapters.torch import TorchMetric
from dlef.contracts import Metric


def build_v11_metrics() -> Metric:
    def compute(preds, targets):
        values = {}
        values["mae"] = (preds - targets).abs().mean()
        lt10 = targets < 10
        ge10 = ~lt10
        values["mae_lt10"] = (preds[lt10] - targets[lt10]).abs().mean() if lt10.any() else targets.new_tensor(float("nan"))
        values["mae_ge10"] = (preds[ge10] - targets[ge10]).abs().mean() if ge10.any() else targets.new_tensor(float("nan"))
        values["acc10"] = ((preds < 10) == lt10).float().mean()
        worst = None
        for k in range(2, 10):
            q = (targets >= k) & (targets < k + 1)
            if q.any():
                m = (preds[q] - targets[q]).abs().mean()
                values[f"mae_{k}_{k+1}"] = m
                worst = m if worst is None else torch.maximum(worst, m)
        values["worst_bin"] = worst if worst is not None else targets.new_tensor(float("nan"))
        return values

    return TorchMetric(compute)
