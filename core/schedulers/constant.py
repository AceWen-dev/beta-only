from __future__ import annotations

from adapters.torch import TorchOptimizer, TorchScheduler
from ..config import SchedulerConfig


def build_constant(
    config: SchedulerConfig,
    optimizer: TorchOptimizer,
    epochs: int,
) -> TorchScheduler | None:
    return None
