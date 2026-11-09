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
        # 主损失可切换为相对误差 / 混合误差（D-0022 之后的最优配方用 relative）
        main = mm_case_bin(
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
        feats = output.aux.get("features")
        if feats is None:
            raise RuntimeError("mm_rank requires ModelOutput.aux['features']")
        rank = batchwise_ranking_regularizer(feats, batch["y"].unsqueeze(1), config.rank_interp)
        total = main + config.rank_gamma * rank
        terms = {"mm_case_bin": main, "rank": rank}
        # D-0045：分支独立监督——β/直接分支各自算同一口径的尺寸损失
        if config.branch_beta_weight > 0 or config.branch_direct_weight > 0:
            for key, weight, term_name in (
                ("pred_beta", config.branch_beta_weight, "loss_beta"),
                ("pred_direct", config.branch_direct_weight, "loss_direct"),
            ):
                branch_pred = output.aux.get(key)
                if branch_pred is None:
                    raise RuntimeError(f"mm_rank branch supervision needs aux[{key!r}]")
                if weight > 0:
                    branch_loss = mm_case_bin(
                        branch_pred,
                        batch["y"],
                        batch["case_id"],
                        config.delta,
                        relative=config.relative,
                        log_space=config.log_space,
                        log_beta=config.log_beta,
                    )
                    total = total + weight * branch_loss
                    terms[term_name] = branch_loss
        # D-0050：β 分支配对一致性。同一病例的远/近帧真实尺寸相同，故要求
        # β 分支输出相等；直接约束分支，模型无法只靠压低门控让最终输出变稳。
        if config.pair_beta_weight > 0:
            branch_pred = output.aux.get("pred_beta")
            if branch_pred is None:
                raise RuntimeError("mm_rank pair loss needs aux['pred_beta']")
            cases = batch.get("case_id")
            if cases is None:
                raise RuntimeError("mm_rank pair loss needs batch['case_id']")
            groups: dict[str, list[int]] = {}
            for i, case in enumerate(cases):
                groups.setdefault(str(case), []).append(i)
            pairs = [
                (a, b)
                for idx in groups.values()
                for a, b in zip(idx[0::2], idx[1::2])
            ]
            if pairs:
                log_pred = torch.log(branch_pred.clamp_min(1e-3))
                diffs = torch.stack([log_pred[a] - log_pred[b] for a, b in pairs])
                pair_loss = F.smooth_l1_loss(diffs, torch.zeros_like(diffs), beta=1.0)
                total = total + config.pair_beta_weight * pair_loss
                terms["loss_pair_beta"] = pair_loss
        return LossOutput(total=total, terms=terms)

    return TorchLoss(loss_fn)
