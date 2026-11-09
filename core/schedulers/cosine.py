from __future__ import annotations

import math

import torch

from adapters.torch import TorchOptimizer, TorchScheduler
from ..config import SchedulerConfig


def build_cosine(
    config: SchedulerConfig,
    optimizer: TorchOptimizer,
    epochs: int,
) -> TorchScheduler:
    t_max = config.t_max if config.t_max is not None else epochs
    warmup = config.warmup_epochs
    if warmup > 0:
        # 与既有最优配方一致：前 warmup 轮线性升到 1，之后余弦降到 eta_min。
        # LambdaLR 的入参从 0 开始，所以第 1 轮对应 step=1。
        def factor(step: int) -> float:
            position = step + 1
            if position <= warmup:
                return position / warmup
            progress = (position - warmup) / max(1, t_max - warmup)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return TorchScheduler(
            torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=factor),
            step_unit="epoch",
        )
    return TorchScheduler(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer.optimizer,
            T_max=t_max,
            eta_min=config.eta_min,
        ),
        step_unit="epoch",
    )
