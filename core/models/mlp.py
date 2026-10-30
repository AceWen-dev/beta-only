from __future__ import annotations

from typing import Any

from torch import nn

from adapters.torch import TorchModel
from ..config import ModelConfig


class MLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_mlp(config: ModelConfig, device: Any = "cpu") -> TorchModel:
    return TorchModel(MLP(config.hidden), lambda module, batch: module(batch["x"]), device=device)
