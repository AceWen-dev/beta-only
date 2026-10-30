from __future__ import annotations

from typing import Callable

from adapters.torch import TorchLoss
from ..config import LossConfig
from .huber import build_huber
from .mm_case_bin import build_mm_case_bin
from .mm_rank import build_mm_rank
from .mse import build_mse

BUILDERS: dict[str, Callable[[LossConfig], TorchLoss]] = {
    "mse": build_mse,
    "huber": build_huber,
    "mm_case_bin": build_mm_case_bin,
    "mm_rank": build_mm_rank,
}


def build_loss(config: LossConfig) -> TorchLoss:
    try:
        builder = BUILDERS[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown loss.name: {config.name!r}; available: {sorted(BUILDERS)}"
        ) from exc
    return builder(config)
