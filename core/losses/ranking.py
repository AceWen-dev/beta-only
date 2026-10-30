"""可微排序（RankSim 用）：rank_normalised + TrueRanker。

按 `sequence` 从大到小给出归一化名次；反向用插值排序的直通梯度。
"""

from __future__ import annotations

import torch


def rank(seq: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq: torch.Tensor) -> torch.Tensor:
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence: torch.Tensor, lambda_val: float):
        r = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, r)
        return r

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sequence, r = ctx.saved_tensors
        sequence_prime = sequence + ctx.lambda_val * grad_output
        r_prime = rank_normalised(sequence_prime)
        gradient = -(r - r_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None
