from __future__ import annotations

from typing import Any

from torch import nn

from adapters.torch import TorchModel
from ..config import ModelConfig


def build_linear(config: ModelConfig, device: Any = "cpu") -> TorchModel:
    return TorchModel(
        nn.Linear(1, 1),
        lambda module, batch: module(batch["x"]),
        device=device,
    )
