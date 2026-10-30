"""v11 模型：ResNet50 + 双/三视图 + 深度/几何 token + β/直接/门控。

`V11Net.forward` 返回 (mm, pooled)；`build_v11` 的 `_forward_v11` 把 pooled
放进 `ModelOutput.aux`，供 RankSim 损失读取。推理结构不依赖 RankSim。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adapters.torch import TorchModel
from dlef.contracts import ModelOutput
from ..config import ModelConfig
from ..data.paths import data_root


class V11Net(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        from torchvision.models import resnet50

        imagenet = Path.home() / ".cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
        if imagenet.is_file():
            backbone = resnet50(weights=None)
            backbone.load_state_dict(torch.load(imagenet, map_location="cpu", weights_only=True))
        else:
            backbone = resnet50(weights="IMAGENET1K_V1")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.beta_ref = float(config.beta_ref)
        self.truth_median = float(config.truth_median)
        self.k_bound = float(config.k_bound)
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
        self.head_beta = nn.Sequential(nn.Linear(2 * d, 64), nn.SiLU(), nn.Linear(64, 1))
        self.head_dir = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, 1))
        self.gate = nn.Linear(2 * d, 1)
        self.inject = nn.Linear(d, d)
        self.proj = (
            nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 16))
            if config.proj
            else None
        )
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

        nn.init.zeros_(self.head_beta[-1].weight)
        nn.init.zeros_(self.head_beta[-1].bias)
        nn.init.zeros_(self.head_dir[-1].weight)
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

    def forward(self, batch: dict, feats: torch.Tensor | None = None):
        n = batch["D"].shape[0]
        if feats is None:
            feats = self.backbone(batch["rgb_views"].flatten(0, 1)).reshape(n, self.views, 2048)
        rgb = self.rgb_proj(feats) + self.view_emb + self.modality[0]

        b2, n_views, c, hd, wd = batch["depth"].shape
        dep = self.depth_net(batch["depth"].reshape(b2 * n_views, c, hd, wd)).reshape(b2, n_views, 64)
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
        z = self.head_beta(feat_beta).squeeze(1)
        beta = self.beta_ref * torch.exp(self.k_bound * torch.tanh(z / self.k_bound))
        pred_beta = beta * batch["D"] * batch["S"]
        pred_dir = torch.nn.functional.softplus(
            self.head_dir(tok[:, : self.views].mean(dim=1)).squeeze(1)
        )
        wg = torch.sigmoid(self.gate(torch.cat([pooled, geom], dim=1)).squeeze(1))
        return wg * pred_beta + (1 - wg) * pred_dir, pooled


def _forward_v11(module: V11Net, batch: dict) -> ModelOutput:
    pred, pooled = module(batch)
    features = module.proj(pooled) if module.proj is not None else pooled
    return ModelOutput(predictions=pred, aux={"features": features})


def build_v11(config: ModelConfig, device: Any = "cpu") -> TorchModel:
    net = V11Net(config)
    if config.init_checkpoint:
        path = data_root() / config.init_checkpoint
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(ck["model"])
        old = state.pop("view_emb", None)
        net.load_state_dict(state, strict=False)
        if old is not None:
            with torch.no_grad():
                k = min(old.shape[0], net.view_emb.shape[0])
                net.view_emb[:k].copy_(old[:k])
                if net.view_emb.shape[0] > k:
                    net.view_emb[k:].copy_(old.mean(dim=0))
    return TorchModel(net, _forward_v11, device=device)
