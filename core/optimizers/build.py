from __future__ import annotations

from typing import Callable

from adapters.torch import TorchModel, TorchOptimizer
from ..config import OptimConfig
from .adam import build_adam
from .adamw import build_adamw
from .sgd import build_sgd
from .v11_adamw import build_v11_adamw

BUILDERS: dict[str, Callable[[OptimConfig, TorchModel], TorchOptimizer]] = {
    "adamw": build_adamw,
    "adam": build_adam,
    "sgd": build_sgd,
    "v11_adamw": build_v11_adamw,
}


def build_optimizer(config: OptimConfig, model: TorchModel) -> TorchOptimizer:
    try:
        builder = BUILDERS[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown optim.name: {config.name!r}; available: {sorted(BUILDERS)}"
        ) from exc
    return builder(config, model)
