"""AdamW with separate backbone/head learning rates (v11 recipe)."""

from __future__ import annotations

import torch

from adapters.torch import TorchModel, TorchOptimizer
from ..config import OptimConfig


def build_v11_adamw(config: OptimConfig, model: TorchModel) -> TorchOptimizer:
    backbone = [p for n, p in model.named_parameters()
                if n.startswith("backbone.") and p.requires_grad]
    head = [p for n, p in model.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad]
    groups = [{"params": head, "lr": config.lr}]
    if backbone:
        groups.append({"params": backbone, "lr": config.lr_backbone or config.lr})
    return TorchOptimizer(torch.optim.AdamW(groups, weight_decay=config.weight_decay))
