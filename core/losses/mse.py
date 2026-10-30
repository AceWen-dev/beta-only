from __future__ import annotations

import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput
from ..config import LossConfig


def build_mse(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        total = F.mse_loss(output.predictions, batch["y"])
        return LossOutput(total=total, terms={"mse": total})

    return TorchLoss(loss_fn)
