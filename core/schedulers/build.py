from __future__ import annotations

from typing import Callable

from adapters.torch import TorchOptimizer, TorchScheduler
from ..config import SchedulerConfig
from .constant import build_constant
from .cosine import build_cosine

BUILDERS: dict[str, Callable[[SchedulerConfig, TorchOptimizer, int], TorchScheduler | None]] = {
    "cosine": build_cosine,
    "constant": build_constant,
}


def build_scheduler(
    config: SchedulerConfig,
    optimizer: TorchOptimizer,
    *,
    epochs: int,
) -> TorchScheduler | None:
    try:
        builder = BUILDERS[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown scheduler.name: {config.name!r}; available: {sorted(BUILDERS)}"
        ) from exc
    return builder(config, optimizer, epochs)
