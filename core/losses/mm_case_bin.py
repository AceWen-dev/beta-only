"""毫米域、病例等权、尺寸区间等权的 SmoothL1。"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput
from ..config import LossConfig


def bin_of(value: float) -> int:
    return 10 if value >= 10.0 else int(value)


def mm_case_bin(pred: torch.Tensor, truth: torch.Tensor, case_ids, delta: float) -> torch.Tensor:
    per = F.smooth_l1_loss(pred, truth, beta=delta, reduction="none")
    case_truth: dict[str, float] = {}
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for i, c in enumerate(case_ids):
        c = str(c)
        groups[c].append(per[i])
        case_truth[c] = float(truth[i].detach())
    bins: dict[int, list[torch.Tensor]] = defaultdict(list)
    for c, vals in groups.items():
        bins[bin_of(case_truth[c])].append(torch.stack(vals).mean())
    return torch.stack([torch.stack(v).mean() for v in bins.values()]).mean()


def build_mm_case_bin(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        total = mm_case_bin(output.predictions, batch["y"], batch["case_id"], config.delta)
        return LossOutput(total=total, terms={"mm_case_bin": total})

    return TorchLoss(loss_fn)
