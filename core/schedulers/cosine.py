from __future__ import annotations

import torch

from adapters.torch import TorchOptimizer, TorchScheduler
from ..config import SchedulerConfig


def build_cosine(
    config: SchedulerConfig,
    optimizer: TorchOptimizer,
    epochs: int,
) -> TorchScheduler:
    t_max = config.t_max if config.t_max is not None else epochs
    return TorchScheduler(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer,
            T_max=t_max,
            eta_min=config.eta_min,
        ),
        step_unit="epoch",
    )
