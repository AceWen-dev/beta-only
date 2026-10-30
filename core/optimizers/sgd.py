from __future__ import annotations

import torch

from adapters.torch import TorchModel, TorchOptimizer
from ..config import OptimConfig


def build_sgd(config: OptimConfig, model: TorchModel) -> TorchOptimizer:
    return TorchOptimizer(
        torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    )
