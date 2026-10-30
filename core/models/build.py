from __future__ import annotations

from typing import Any, Callable

from adapters.torch import TorchModel
from ..config import ModelConfig
from .linear import build_linear
from .mlp import build_mlp
from .v11 import build_v11

BUILDERS: dict[str, Callable[[ModelConfig, Any], TorchModel]] = {
    "mlp": build_mlp,
    "linear": build_linear,
    "v11": build_v11,
}


def build_model(config: ModelConfig, device: Any = "cpu") -> TorchModel:
    try:
        builder = BUILDERS[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model.name: {config.name!r}; available: {sorted(BUILDERS)}"
        ) from exc
    return builder(config, device)
