"""v11 模型：ResNet50 + 双/三视图 + 深度/几何 token + β/直接/门控。

`V11Net.forward` 返回 (mm, pooled)；`build_v11` 的 `_forward_v11` 把 pooled
放进 `ModelOutput.aux`，供 RankSim 损失读取。推理结构不依赖 RankSim。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adapters.torch import TorchModel
from dlef.contracts import ModelOutput
from ..config import ModelConfig
from ..data.paths import data_root
from ..losses.mm_case_bin import bin_of

# 档位数由 `bin_of` 的约定唯一决定：0–9 mm 各一档 + >=10 mm 一档（无档内偏移）
ORDINAL_BINS = 11
TAIL_BIN = ORDINAL_BINS - 1


def metric_depth_tokens(depth: torch.Tensor, D: torch.Tensor,
                        q_ratio: torch.Tensor, ref_mm: float = 20.0) -> torch.Tensor:
    """把 `depth/global_q` token 还原成 `log(depth_mm/ref)`（保留米制刻度）。

    depth: [B, views, 2, H, W]；通道 0=prepare_lesion 的 depth/global_q，
    通道 1=有效掩码。D=q（mm），q_ratio=scalars[7]=q/global_q → global_q=q/q_ratio。
    还原后的 mm 深度只受 prepare_lesion 的 `clip(.,0,4)` 上限影响（远处背景饱和）；
    D-0087 只在 EndoStreamDepth 的 mm 数据上启用。
    """
    valid = depth[:, :, 1:2]
    q = D.view(-1, 1, 1, 1, 1)
    ratio = q_ratio.view(-1, 1, 1, 1, 1).clamp_min(1e-6)
    global_q = q / ratio
    depth_mm = depth[:, :, 0:1] * global_q
    metric = torch.log(depth_mm.clamp_min(1e-3) / ref_mm).clamp(-2.0, 2.0)
    metric = torch.where(valid > 0.5, metric, torch.zeros_like(metric))
    return torch.cat([metric, valid], dim=2)


def anchor_k(z_phys: torch.Tensor, z_res: torch.Tensor, beta_ref: float,
             phys_bound: float, residual_bound: float) -> torch.Tensor:
    """物理锚定的 k：k_phys 只由几何决定，特征残差被 bound 硬性压小。"""
    k_phys = beta_ref * torch.exp(phys_bound * torch.tanh(z_phys / phys_bound))
    return k_phys * torch.exp(residual_bound * torch.tanh(z_res / residual_bound))


def focal_beta(z: torch.Tensor, beta_ref: float, bound: float) -> torch.Tensor:
    """米制焦距头：β = 1/f，f = (1/β_ref)·exp(bound·tanh(z/bound))。"""
    return beta_ref * torch.exp(-bound * torch.tanh(z / bound))


def focal_v2_scale(z_phys: torch.Tensor, z_app: torch.Tensor,
                   phys_bound: float, app_bound: float) -> torch.Tensor:
    """D-0093 双因子尺度乘数（相对 beta_ref·D·S）。

    等价于 log(f/W) = -log(beta_ref) - log(W) + phys_bound·tanh(z_phys/phys_bound)：
    W 在 D·(S/W) 中约掉，零输入时乘数=1，预测回到 beta_ref·D·S。
    """
    return torch.exp(
        app_bound * torch.tanh(z_app / app_bound)
        - phys_bound * torch.tanh(z_phys / phys_bound)
    )


class V11Net(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        imagenet = Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
        if imagenet.is_file():
            backbone = resnet50(weights=None)
            backbone.load_state_dict(torch.load(imagenet, map_location="cpu", weights_only=True))
        else:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        # 保留 7×7 空间网格：池化改在 forward 里显式做（mean == avgpool）
        backbone.avgpool = nn.Identity()
        self.backbone = backbone
        self.beta_ref = float(config.beta_ref)
        self.truth_median = float(config.truth_median)
        self.k_bound = float(config.k_bound)
        self.metric_depth = bool(getattr(config, "metric_depth", False))
        self.metric_ref_mm = float(getattr(config, "metric_ref_mm", 20.0))
        self.metric_anchor = bool(getattr(config, "metric_anchor", False))
        self.metric_residual_bound = float(getattr(config, "metric_residual_bound", 0.3))
        self.metric_phys_bound = float(getattr(config, "metric_phys_bound", 1.0))
        self.metric_span_ref_px = float(getattr(config, "metric_span_ref_px", 300.0))
        self.metric_focal_head = bool(getattr(config, "metric_focal_head", False))
        self.metric_focal_bound = float(getattr(config, "metric_focal_bound", 0.15))
        self.metric_focal_v2 = bool(getattr(config, "metric_focal_v2", False))
        self.metric_focal_phys_bound = float(
            getattr(config, "metric_focal_phys_bound", 2.0)
        )
        self.metric_focal_app_bound = float(
            getattr(config, "metric_focal_app_bound", 0.3)
        )
        self.metric_focal_width_ref = float(
            getattr(config, "metric_focal_width_ref", 1258.0)
        )
        self.metric_focal_height_ref = float(
            getattr(config, "metric_focal_height_ref", 1080.0)
        )
        self.views = int(config.views)
        d = int(config.d_model)

        self.rgb_proj = nn.Linear(2048, d)
        self.depth_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), 
        )
        self.depth_proj = nn.Linear(64, d)
        self.geom_proj = nn.Sequential(nn.Linear(10, d), nn.SiLU())
        self.view_emb = nn.Parameter(torch.zeros(self.views, d))
        self.modality = nn.Parameter(torch.zeros(2, d))
        self.attn = nn.MultiheadAttention(d, int(config.heads), batch_first=True)
        self.norm = nn.LayerNorm(d)
        # D-0069：分档头开启时不构造 β/direct/门控三个头（输出完全由分档头给出）
        self.ordinal_head = bool(getattr(config, "ordinal_head", False))
        # D-0077：beta-only 融合。`gate`（默认）= 原 β/direct/门控三头；
        # `none` = 只保留 head_beta，推理/训练都没有 direct 与 gate（初值与
        # diranchor 相同：βref·D·S）。
        self.fusion = str(getattr(config, "fusion", "gate"))
        # D-0083：纯图像回归对照；只用 RGB 两视图，直接回归 mm。
        self.image_only = bool(getattr(config, "image_only", False))
        if self.ordinal_head:
            self.head_ord = nn.Linear(2 * d, ORDINAL_BINS)
            self.head_off = nn.Linear(2 * d, 1)
        elif self.image_only:
            self.head_img = nn.Sequential(
                nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1)
            )
        else:
            self.head_beta = nn.Sequential(nn.Linear(2 * d, 64), nn.SiLU(), nn.Linear(64, 1))
            if self.fusion == "none":
                self.head_dir = None
                self.gate = None
            else:
                self.head_dir = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))
                self.gate = nn.Linear(2 * d, 1)
        if self.metric_anchor:
            # 零初始化：起点与普通 β-only 一致（k=beta_ref），随后只学几何曲线
            self.phys_net = nn.Sequential(
                nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1)
            )
            nn.init.zeros_(self.phys_net[-1].weight)
            nn.init.zeros_(self.phys_net[-1].bias)
        if self.metric_focal_v2:
            # 几何宽界头：D/S/W/H/S·W/S·H/框宽高比 + 深度统计 → log(f/W)
            self.focal_phys_net = nn.Sequential(
                nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 1)
            )
            nn.init.zeros_(self.focal_phys_net[-1].weight)
            nn.init.zeros_(self.focal_phys_net[-1].bias)
        self.inject = nn.Linear(d, d)
        self.proj = (
            nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 16))
            if config.proj
            else None
        )
        self.size_input = bool(getattr(config, "size_input", False))
        # D-0067 解压缩两臂；两者都不新增参数，关闭时与旧模型逐位一致
        self.gate_beta_floor = float(getattr(config, "gate_beta_floor", 0.0))
        self.direct_scale_anchor = bool(getattr(config, "direct_scale_anchor", False))
        if self.size_input:
            # 零初始化：训练开始等价于原模型，随后学习尺寸条件修正
            self.size_proj = nn.Linear(2, 2 * d)
            nn.init.zeros_(self.size_proj.weight)
            nn.init.zeros_(self.size_proj.bias)
        # D-0078：span 旁路。保留 S 的绝对尺度，把无量纲的 S/W 与内容宽度 W
        # 作为零初始化条件输入（与 D-0027 size_input 同一模式）；起点与无旁路
        # 模型逐位一致。它不替换 S，避免 D-0075 的 S→1258·S/W 交换。
        self.span_input = bool(getattr(config, "span_input", False))
        self.span_ratio_ref = float(getattr(config, "span_ratio_ref", 0.24629))
        self.span_width_ref = float(getattr(config, "span_width_ref", 1258.0))
        if self.span_input:
            self.span_proj = nn.Linear(2, 2 * d)
            nn.init.zeros_(self.span_proj.weight)
            nn.init.zeros_(self.span_proj.bias)
        # D-0055：7×7 背景空间分支（整图视图）
        self.spatial_context = bool(getattr(config, "spatial_context", False))
        if self.spatial_context:
            self.ctx_proj = nn.Conv2d(2048, d, kernel_size=1)
            self.ctx_pos = nn.Parameter(torch.zeros(1, 49, d))
            self.ctx_box = nn.Linear(5, d)
            self.ctx_attn = nn.MultiheadAttention(
                d, int(config.heads), batch_first=True
            )
            self.ctx_norm = nn.LayerNorm(d)
            self.ctx_inject = nn.Linear(d, d)
            nn.init.zeros_(self.ctx_inject.weight)
            nn.init.zeros_(self.ctx_inject.bias)
        self.backbone_strategy = config.backbone
        if config.backbone == "partial":
            self._frozen = [self.backbone.conv1, self.backbone.bn1,
                            self.backbone.layer1, self.backbone.layer2, self.backbone.layer3]
        elif config.backbone == "frozen":
            self._frozen = [self.backbone]
        else:
            self._frozen = []
        for module in self._frozen:
            for param in module.parameters():
                param.requires_grad = False

        if self.ordinal_head:
            # 起点＝训练集中位数所在档的档中心（与旧 head_dir 的 bias 初值同一思路）：
            # 权重全零 → logits 只剩 bias，argmax 落在 truth_median 所在档；
            # 偏移 bias 0 → sigmoid=0.5 → pred = 档左端 + 0.5。
            nn.init.zeros_(self.head_ord.weight)
            nn.init.zeros_(self.head_ord.bias)
            with torch.no_grad():
                self.head_ord.bias[bin_of(float(config.truth_median))] = 2.0
            nn.init.zeros_(self.head_off.weight)
            nn.init.zeros_(self.head_off.bias)
        elif self.image_only:
            nn.init.zeros_(self.head_img[-1].weight)
            nn.init.constant_(
                self.head_img[-1].bias,
                float(np.log(np.expm1(config.truth_median))),
            )
        else:
            nn.init.zeros_(self.head_beta[-1].weight)
            nn.init.zeros_(self.head_beta[-1].bias)
            if self.fusion != "none":
                nn.init.zeros_(self.head_dir[-1].weight)
                if self.direct_scale_anchor:
                    # 锚定版初值：exp(log beta_ref)·D·S == β 分支初值，两分支起点一致
                    nn.init.constant_(self.head_dir[-1].bias, float(np.log(self.beta_ref)))
                else:
                    nn.init.constant_(self.head_dir[-1].bias, float(np.log(np.expm1(config.truth_median))))
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 2.0)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self._frozen:
                module.eval()
        return self

    def forward(
        self, batch: dict, feats: torch.Tensor | None = None, return_branches: bool = False
    ):
        n = batch["D"].shape[0]
        raw = None
        if feats is None:
            raw = self.backbone(batch["rgb_views"].flatten(0, 1))
            # resnet 的 flatten(1) 把 (B,2048,7,7) 摊平；这里还原回网格
            if raw.dim() == 2 and raw.shape[1] % 2048 == 0:
                side = int(round((raw.shape[1] // 2048) ** 0.5))
                if side * side * 2048 == raw.shape[1]:
                    raw = raw.reshape(raw.shape[0], 2048, side, side)
            if raw.dim() == 4:
                # avgpool 已替换为 Identity：显式求均值，与原结构数值等价
                feats = raw.mean(dim=(2, 3)).reshape(n, self.views, 2048)
            else:
                feats = raw.reshape(n, self.views, 2048)
        elif self.spatial_context:
            raise ValueError("spatial_context needs the backbone feature map (raw)")
        rgb = self.rgb_proj(feats) + self.view_emb + self.modality[0]
        if self.image_only:
            # D-0083：纯图像回归只用 RGB 两视图；深度/几何完全不进网络。
            out, _ = self.attn(rgb, rgb, rgb)
            tok = self.norm(rgb + out)
            pooled = tok.mean(dim=1)
            pred = torch.nn.functional.softplus(
                self.head_img(pooled).squeeze(1)
            )
            if return_branches:
                return pred, pooled, {"pred_direct": pred}
            return pred, pooled

        depth_in = batch["depth"]
        if self.metric_depth:
            depth_in = metric_depth_tokens(
                depth_in, batch["D"], batch["scalars"][:, 7], self.metric_ref_mm
            )
        b2, n_views, c, hd, wd = depth_in.shape
        dep = self.depth_net(depth_in.reshape(b2 * n_views, c, hd, wd)).reshape(b2, n_views, 64)
        dep_tok = self.depth_proj(dep) + self.view_emb[:n_views] + self.modality[1]
        geom = self.geom_proj(batch["scalars"])
        tok = torch.cat([rgb, dep_tok, geom.unsqueeze(1)], dim=1)
        out, _ = self.attn(tok, tok, tok)
        tok = self.norm(tok + out)
        pooled = tok.mean(dim=1)

        depth_guide = self.inject(dep_tok.mean(dim=1))
        feat_beta = torch.cat([pooled, geom], dim=1) + torch.cat(
            [depth_guide, torch.zeros_like(geom)], dim=1
        )
        if self.spatial_context:
            ctx_vec = self._context_vector(raw, rgb, batch)
            feat_beta = feat_beta + torch.cat(
                [self.ctx_inject(ctx_vec), torch.zeros_like(geom)], dim=1
            )
        if self.size_input:
            # 固定标准化（log S ≈ 5、log D ≈ -1）后经零初始化投影加入 β 头输入，
            # 让头能显式感知框大小与深度尺度（D-0027）。
            size = torch.stack(
                [
                    torch.log(batch["S"].clamp_min(1.0)) - 5.0,
                    torch.log(batch["D"].clamp_min(1e-3)) + 1.0,
                ],
                dim=1,
            )
            feat_beta = feat_beta + self.size_proj(size)
        if self.span_input:
            # D-0078：S 保持原样，另加 [log(S/W), log W] 的零初始化旁路。
            ratio = batch["S_norm"].clamp_min(1e-8)
            width = batch["W"].clamp_min(1.0)
            span = torch.stack(
                [
                    torch.log(ratio) - math.log(self.span_ratio_ref),
                    torch.log(width) - math.log(self.span_width_ref),
                ],
                dim=1,
            )
            feat_beta = feat_beta + self.span_proj(span)
        if self.ordinal_head:
            # D-0069：输出 = 档左端 + 档内偏移。档宽 1mm 且从 0 开始，故档左端就是档序号。
            # argmax 不可导，logits 由 CE 训练、偏移由 SmoothL1 训练（见 losses/mm_ordinal）。
            logits = self.head_ord(feat_beta)
            raw_off = self.head_off(feat_beta).squeeze(1)
            bin_idx = logits.argmax(dim=1)
            left = bin_idx.to(raw_off.dtype)
            pred = torch.where(
                bin_idx == TAIL_BIN,
                left + torch.nn.functional.softplus(raw_off),
                left + torch.sigmoid(raw_off),
            )
            if return_branches:
                return pred, pooled, {
                    "ord_logits": logits,
                    "ord_offset": raw_off,
                    "ord_bin": bin_idx,
                }
            return pred, pooled
        if self.metric_anchor:
            d = batch["D"].clamp_min(1e-3)
            s = batch["S"].clamp_min(1.0)
            phys = torch.stack(
                [
                    torch.log(d) - math.log(self.metric_ref_mm),
                    torch.log(s) - math.log(self.metric_span_ref_px),
                ],
                dim=1,
            ).clamp(-2.0, 2.0)
            z_phys = self.phys_net(phys).squeeze(1)
            z_res = self.head_beta(feat_beta).squeeze(1)
            k = anchor_k(
                z_phys, z_res, self.beta_ref,
                self.metric_phys_bound, self.metric_residual_bound,
            )
            pred_beta = k * batch["D"] * batch["S"]
        else:
            z = self.head_beta(feat_beta).squeeze(1)
            if self.metric_focal_v2:
                # D-0093：几何宽界定相机尺度，外观窄残差只修框/分割误差
                z_phys = self.focal_phys_net(
                    self._focal_phys_inputs(batch)
                ).squeeze(1)
                scale = focal_v2_scale(
                    z_phys, z,
                    self.metric_focal_phys_bound,
                    self.metric_focal_app_bound,
                )
                pred_beta = self.beta_ref * batch["D"] * batch["S"] * scale
            elif self.metric_focal_head:
                # D-0091：只让外观决定有限范围的焦距修正，β=1/f 的物理刻度由
                # beta_ref 与米制 D 承担；紧界阻止 B 上的尺寸相关压缩。
                beta = focal_beta(z, self.beta_ref, self.metric_focal_bound)
                pred_beta = beta * batch["D"] * batch["S"]
            else:
                beta = self.beta_ref * torch.exp(
                    self.k_bound * torch.tanh(z / self.k_bound)
                )
                pred_beta = beta * batch["D"] * batch["S"]
        if self.fusion == "none":
            if return_branches:
                return pred_beta, pooled, {"pred_beta": pred_beta}
            return pred_beta, pooled
        dir_in = tok[:, : self.views].mean(dim=1)
        if self.direct_scale_anchor:
            # 对数空间的无量纲乘性修正：pred = exp(k_dir)·D·S（clamp 只防溢出）
            k_dir = torch.exp(self.head_dir(dir_in).squeeze(1).clamp(-15.0, 5.0))
            pred_dir = k_dir * batch["D"] * batch["S"]
        else:
            pred_dir = torch.nn.functional.softplus(self.head_dir(dir_in).squeeze(1))
        wg = torch.sigmoid(self.gate(torch.cat([pooled, geom], dim=1)).squeeze(1))
        if self.gate_beta_floor > 0.0:
            wg = self.gate_beta_floor + (1.0 - self.gate_beta_floor) * wg
        fused = wg * pred_beta + (1 - wg) * pred_dir
        if return_branches:
            # D-0045：分支独立监督需要两个分支的原始输出
            return fused, pooled, {"pred_beta": pred_beta, "pred_direct": pred_dir, "gate": wg}
        return fused, pooled

    def _focal_phys_inputs(self, batch: dict) -> torch.Tensor:
        """D-0093 几何尺度头输入：米制几何 + 分辨率 + 深度统计（10 维）。"""
        sc = batch["scalars"].float()
        d = batch["D"].clamp_min(1e-3)
        s = batch["S"].clamp_min(1.0)
        w = batch["W"].clamp_min(1.0)
        hw = sc[:, 5].clamp_min(1e-3)  # H/W（与 _box_features 同一约定）
        h = (w * hw).clamp_min(1.0)
        bw_n, bh_n = sc[:, 1], sc[:, 2]  # bw/W, bh/H
        bbox_ar = (bw_n.clamp_min(1e-3) / bh_n.clamp_min(1e-3)) / hw
        dep = batch["depth"][:, :, 0].float()
        val = batch["depth"][:, :, 1].float()
        vsum = val.sum(dim=(1, 2, 3)).clamp_min(1.0)
        mean = (dep * val).sum(dim=(1, 2, 3)) / vsum
        var = (((dep - mean[:, None, None, None]) ** 2) * val).sum(
            dim=(1, 2, 3)
        ) / vsum
        std = var.clamp_min(0).sqrt()
        valid = val.mean(dim=(1, 2, 3))
        return torch.stack(
            [
                torch.log(d) - math.log(self.metric_ref_mm),
                torch.log(s) - math.log(self.metric_span_ref_px),
                torch.log(w) - math.log(self.metric_focal_width_ref),
                torch.log(h) - math.log(self.metric_focal_height_ref),
                torch.log(s / w) - math.log(self.span_ratio_ref),
                torch.log(s / h) - math.log(0.30),
                torch.log(bbox_ar.clamp_min(1e-3)),
                mean,
                std,
                valid,
            ],
            dim=1,
        )

    @staticmethod
    def _box_features(batch: dict, grid: int = 7, size: float = 224.0) -> torch.Tensor:
        """把病灶框映射到 7×7 网格：每格的 [框内, Δx, Δy, 框宽, 框高]。

        只用 batch 里的几何标量（bw/W、bh/H、cx/W、cy/H、H/W）与宽高比，
        不读图、也不依赖 S（因此与 span_calibration 无关）。
        """
        sc = batch["scalars"].float()
        n = sc.shape[0]
        bw_n, bh_n, cx, cy, hw = sc[:, 1], sc[:, 2], sc[:, 3], sc[:, 4], sc[:, 5]
        # letterbox 后内容区在 224 画布上的宽度只取决于 H/W（宽高比）：
        #   wpx = W·(224/max(H,W)) = 224 / max(H/W, 1)
        # 这样不依赖 S（也就不受 span_calibration 影响）。
        aspect = torch.maximum(hw, torch.ones_like(hw))
        wpx = size / aspect
        hpx = wpx * hw
        left = (size - wpx) / 2
        top = (size - hpx) / 2
        x1 = (cx - bw_n / 2) * wpx + left
        x2 = (cx + bw_n / 2) * wpx + left
        y1 = (cy - bh_n / 2) * hpx + top
        y2 = (cy + bh_n / 2) * hpx + top
        centers = (torch.arange(grid, device=sc.device, dtype=sc.dtype) + 0.5) * (
            size / grid
        )
        gx = centers.view(1, 1, grid)  # 列
        gy = centers.view(1, grid, 1)  # 行
        x1, x2, y1, y2 = (v.view(n, 1, 1) for v in (x1, x2, y1, y2))
        inside = ((gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2)).to(sc.dtype)
        gxn, gyn = gx / size, gy / size
        cxv = (x1 + x2) / (2 * size)
        cyv = (y1 + y2) / (2 * size)
        bwv = (x2 - x1) / size
        bhv = (y2 - y1) / size
        feats = torch.stack(
            [
                inside,
                (gxn - cxv).expand(n, grid, grid),
                (gyn - cyv).expand(n, grid, grid),
                bwv.expand(n, grid, grid),
                bhv.expand(n, grid, grid),
            ],
            dim=-1,
        )
        return feats.reshape(n, grid * grid, 5)

    def _context_vector(self, raw, rgb, batch) -> torch.Tensor:
        """整图 7×7 网格 → 交叉注意力（query = ROI 视图 token）→ 上下文向量。"""
        n = rgb.shape[0]
        g = raw.shape[-1]
        if g * g != self.ctx_pos.shape[1]:
            raise ValueError(
                f"spatial_context expects {int(self.ctx_pos.shape[1] ** 0.5)}×"
                f"{int(self.ctx_pos.shape[1] ** 0.5)} grid, got {g}×{raw.shape[-2]}"
            )
        grid = raw.reshape(n, self.views, 2048, raw.shape[-2], g)[:, 0]
        tokens = self.ctx_proj(grid).flatten(2).transpose(1, 2)  # (n, 49, d)
        tokens = tokens + self.ctx_pos + self.ctx_box(self._box_features(batch, grid=g))
        q = rgb[:, -1:, :]  # 1.25× ROI 视图
        attn, _ = self.ctx_attn(q, tokens, tokens)
        return self.ctx_norm(q + attn).squeeze(1)


def _forward_v11(module: V11Net, batch: dict) -> ModelOutput:
    pred, pooled, branches = module(batch, return_branches=True)
    features = module.proj(pooled) if module.proj is not None else pooled
    aux = {"features": features, **branches}
    return ModelOutput(predictions=pred, aux=aux)


def build_v11(config: ModelConfig, device: Any = "cpu") -> TorchModel:
    net = V11Net(config)
    if config.init_checkpoint:
        path = data_root() / config.init_checkpoint
        ck = torch.load(path, map_location="cpu", weights_only=False)
        # 兼容两种 checkpoint：DLEF Checkpoint 对象（.model）与旧 dict（["model"]）
        raw_state = ck.model if hasattr(ck, "model") else ck["model"]
        state = dict(raw_state)
        old = state.pop("view_emb", None)
        net.load_state_dict(state, strict=False)
        if old is not None:
            with torch.no_grad():
                k = min(old.shape[0], net.view_emb.shape[0])
                net.view_emb[:k].copy_(old[:k])
                if net.view_emb.shape[0] > k:
                    net.view_emb[k:].copy_(old.mean(dim=0))
    return TorchModel(net, _forward_v11, device=device)
