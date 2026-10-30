from __future__ import annotations

import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput
from ..config import LossConfig


def build_huber(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        total = F.smooth_l1_loss(
            output.predictions,
            batch["y"],
            beta=config.huber_beta,
        )
        return LossOutput(total=total, terms={"huber": total})

    return TorchLoss(loss_fn)
