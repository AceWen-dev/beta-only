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


def mm_case_bin(
    pred: torch.Tensor,
    truth: torch.Tensor,
    case_ids,
    delta: float,
    relative: bool = False,
    hybrid_rel_weight: float = 0.0,
    hybrid_rel_scale: float = 5.0,
    log_space: bool = False,
    log_beta: float = 0.1,
    log_weight: float = 0.0,
) -> torch.Tensor:
    if log_space:
        # 乘性误差：对 pred、truth 同步缩放不变，尺度压缩在对数域是加性偏移
        per = F.smooth_l1_loss(
            torch.log(pred.clamp_min(1e-3)),
            torch.log(truth.clamp_min(1e-3)),
            beta=log_beta,
            reduction="none",
        )
    elif relative:
        # 相对误差：小息肉按比例获得更大梯度（绝对 mm 误差下各档梯度相同）
        per = (pred - truth).abs() / truth.clamp_min(1.0)
        if log_weight > 0:
            # 叠加对数项：相对项管小档尺度、对数项管大档尺度（D-0034）
            per = per + log_weight * F.smooth_l1_loss(
                torch.log(pred.clamp_min(1e-3)),
                torch.log(truth.clamp_min(1e-3)),
                beta=log_beta,
                reduction="none",
            )
    elif hybrid_rel_weight > 0:
        # D-0022 对照 B：绝对为主 + 少量相对为辅——检验「相对损失饿死大档」假设。
        # per = (1-w)·|err| + w·m·|err|/d，m=训练折标签中位数（统一两项量级）；
        # w=0.25、2mm 病例:8mm 病例 的梯度比 ≈1.5（纯相对为 4，纯绝对为 1）。
        abs_l = (pred - truth).abs()
        rel_l = abs_l / truth.clamp_min(1.0)
        per = (1.0 - hybrid_rel_weight) * abs_l + hybrid_rel_weight * hybrid_rel_scale * rel_l
    else:
        per = F.smooth_l1_loss(pred, truth, beta=delta, reduction="none")
    return case_bin_mean(per, case_ids, [bin_of(float(t)) for t in truth.detach()])


def case_bin_mean(per: torch.Tensor, case_ids, case_bins) -> torch.Tensor:
    """病例等权 × 档位等权：帧→病例均值→（按该病例的档位）档均值→档间均值。

    `case_bins[i]` 是第 i 帧所属病例的档位标签（`bin_of` 口径）。同一病例出现多个
    档位标签时以最后一帧为准，与重构前 `mm_case_bin` 的行为一致。
    """
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    bin_of_case: dict[str, int] = {}
    for i, c in enumerate(case_ids):
        c = str(c)
        groups[c].append(per[i])
        bin_of_case[c] = int(case_bins[i])
    bins: dict[int, list[torch.Tensor]] = defaultdict(list)
    for c, vals in groups.items():
        bins[bin_of_case[c]].append(torch.stack(vals).mean())
    return torch.stack([torch.stack(v).mean() for v in bins.values()]).mean()


def build_mm_case_bin(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        total = mm_case_bin(
            output.predictions,
            batch["y"],
            batch["case_id"],
            config.delta,
            relative=config.relative,
            hybrid_rel_weight=config.hybrid_rel_weight,
            hybrid_rel_scale=config.hybrid_rel_scale,
            log_space=config.log_space,
            log_beta=config.log_beta,
            log_weight=config.log_weight,
        )
        return LossOutput(total=total, terms={"mm_case_bin": total})

    return TorchLoss(loss_fn)
