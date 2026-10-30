"""毫米主损失 + 官方 RankSim 排序约束（特征来自 ModelOutput.aux）。"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput
from ..config import LossConfig
from .mm_case_bin import mm_case_bin
from .ranking import TrueRanker, rank_normalised


def batchwise_ranking_regularizer(features: torch.Tensor, targets: torch.Tensor, interp: float) -> torch.Tensor:
    # 每个唯一标签只留一个样本，避免同标签多帧主导
    uniq = torch.unique(targets)
    if len(uniq) < len(targets):
        idx = []
        for t in uniq:
            idx.append(random.choice((targets == t).nonzero()[:, 0]).item())
        x = features[idx]
        y = targets[idx]
    else:
        x = features
        y = targets

    sim = F.normalize(x.view(x.size(0), -1), dim=1) @ F.normalize(
        x.view(x.size(0), -1), dim=1
    ).t()
    loss = features.new_zeros(())
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0, 1))
        feature_ranks = TrueRanker.apply(sim[i].unsqueeze(0), interp)
        loss = loss + F.mse_loss(feature_ranks, label_ranks)
    return loss


def build_mm_rank(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        main = mm_case_bin(output.predictions, batch["y"], batch["case_id"], config.delta)
        feats = output.aux.get("features")
        if feats is None:
            raise RuntimeError("mm_rank requires ModelOutput.aux['features']")
        rank = batchwise_ranking_regularizer(feats, batch["y"].unsqueeze(1), config.rank_interp)
        total = main + config.rank_gamma * rank
        return LossOutput(total=total, terms={"mm_case_bin": main, "rank": rank})

    return TorchLoss(loss_fn)
