from __future__ import annotations

import torch

from adapters.torch import TorchModel, TorchOptimizer
from ..config import OptimConfig


def build_adamw(config: OptimConfig, model: TorchModel) -> TorchOptimizer:
    return TorchOptimizer(
        torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    )
