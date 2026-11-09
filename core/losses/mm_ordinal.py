"""分档交叉熵 + 档内偏移（D-0069）。

标量回归 + log 空间 SmoothL1 的二次核与病例/档位重加权让模型学「重加权后的条件
均值」，于是逐档偏差单调（company 2–3 +1.95 … 9–10 −1.68，见 D-0068）。分档交叉熵
的最优解是**条件分布本身**，读出取「档左端 + 档内偏移」，且分档正确时档内误差 <1mm
——与「每个 1mm 档 MAE <2mm」的判据直接对齐。

    L = CE(bin) + ordinal_offset_weight · SmoothL1(档内偏移) + rank_gamma · RankSim

CE 与偏移项都按 `mm_case_bin` 的病例/档位等权口径加权，保证与既有 run 可比。
档位定义与 `mm_case_bin.bin_of` 一致：0–9 各一档，>=10mm 归入第 11 档（无档内偏移，
模型输出 `10 + softplus(·)`）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from adapters.torch import TorchLoss
from dlef.contracts import LossOutput
from ..config import LossConfig
from .mm_case_bin import bin_of, case_bin_mean
from .mm_rank import batchwise_ranking_regularizer

TAIL_BIN = 10


def build_mm_ordinal(config: LossConfig) -> TorchLoss:
    def loss_fn(output, batch) -> LossOutput:
        logits = output.aux.get("ord_logits")
        raw_off = output.aux.get("ord_offset")
        if logits is None or raw_off is None:
            raise RuntimeError(
                "mm_ordinal requires ModelOutput.aux['ord_logits'] and aux['ord_offset'] "
                "（model.ordinal_head 必须为 true）"
            )
        truth = batch["y"]
        case_ids = batch.get("case_id")
        if case_ids is None:
            raise RuntimeError("mm_ordinal needs batch['case_id']（病例/档位等权）")

        bin_label = torch.tensor(
            [bin_of(float(t)) for t in truth.detach()],
            device=truth.device,
            dtype=torch.long,
        )
        ce_per = F.cross_entropy(logits, bin_label, reduction="none")
        ce = case_bin_mean(ce_per, case_ids, bin_label)

        # 档内偏移只对有界的 0–9 档定义；监督目标是「真值在自己档内的位置」。
        # 推理时它加在**预测档**的左端上，所以分档错时最多引入 1mm 的错配。
        offset = truth.new_zeros(())
        keep = bin_label < TAIL_BIN
        if bool(keep.any()) and config.ordinal_offset_weight > 0:
            target = truth[keep] - bin_label[keep].to(truth.dtype)
            off_per = F.smooth_l1_loss(
                torch.sigmoid(raw_off[keep]), target,
                beta=config.ordinal_offset_beta, reduction="none",
            )
            offset = case_bin_mean(off_per, [c for c, k in zip(case_ids, keep.tolist()) if k],
                                   bin_label[keep])

        feats = output.aux.get("features")
        if feats is None:
            raise RuntimeError("mm_ordinal requires ModelOutput.aux['features']")
        rank = batchwise_ranking_regularizer(feats, truth.unsqueeze(1), config.rank_interp)

        total = ce + config.ordinal_offset_weight * offset + config.rank_gamma * rank
        return LossOutput(
            total=total, terms={"ord_ce": ce, "ord_offset": offset, "rank": rank}
        )

    return TorchLoss(loss_fn)
