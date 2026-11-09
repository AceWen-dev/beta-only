#!/usr/bin/env python3
"""对齐训练探针：PM 与 DLEF 两个实现在「相同初始化 + 相同批序列 + 相同损失
代码」下逐步训练，找第一个分歧点。

设计：
- 初始化：PM 的 V10Feat 新建后，把 state_dict 灌入 DLEF 的 V11Net（结构同构）。
- 批序列：PM 的 WeightedRandomSampler（固定 generator seed）采出的索引，
  两个模型用**同一批张量**（PM LiveDS 产出一次，两模型共享）。
- 损失：两边都用 PM 的 mm_case_bin_loss + 官方 rank 项（共享实现，隔离模型差异）；
  python random 在每侧损失前同步，保证 dedup 抽取一致。
- 优化器：两侧各自 AdamW（同参数分组），每步后对比若干权重张量。
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

PM = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep")
N_STEPS = 24
STEPS_PER_EPOCH = 128


def main() -> int:
    os.chdir(PM)
    sys.path.insert(0, "/tmp/opencode")
    sys.path.insert(0, str(PM / "runs/size/v11-rank2-official-20261030/code"))
    sys.path.insert(0, str(ROOT))

    from beta_image_v10r50 import (DEVICE, LiveDS, build_bbox_maps, collate as pm_collate,
                                   load_resnet50)
    from beta_image_v10_feat import V10Feat
    from v11_rank2_train import make_weights, mm_case_bin_loss
    from ranksim_official import batchwise_ranking_regularizer
    from core.data.datasets.v11_frames import assign_dev_folds  # noqa: F401 (确认模块可导入)
    from core.config import ModelConfig
    from core.models.v11 import V11Net

    import torch.nn.functional as F

    seed = 20261020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---------- 数据与批序列（PM 采样器，固定 generator） ----------
    rows = json.loads((PM / "outputs/training/v11-mmrank-dev-20261030/da/train.json").read_text())["rows"]
    for_row = build_bbox_maps()
    weights, _, _, _ = make_weights(rows)
    g = torch.Generator()
    g.manual_seed(123456)  # 探针专用固定流
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=N_STEPS * 16, replacement=True, generator=g)
    idx = list(sampler)
    ds_pm = LiveDS(rows, for_row)

    # ---------- 两个模型：PM 新建初始化，DLEF 灌入同一 state ----------
    pm_net = V10Feat(0.07543606728391815, 5.0, load_resnet50()).to(DEVICE)
    state0 = {k: v.detach().clone() for k, v in pm_net.state_dict().items()}
    from core.config import ModelConfig
    from core.models.v11 import V11Net
    resolved = json.loads((ROOT / "output/runs/beta-regression/v11-ranksim-align/20261031-004742_c0f027ae_s20261020_8d14ba/config.resolved.json").read_text())
    dl_net = V11Net(ModelConfig.from_dict(resolved["model"]))
    dl_net.load_state_dict(state0, strict=True)
    dl_net = dl_net.to(DEVICE)
    pm_net.train()
    dl_net.train()

    head = [p for n, p in pm_net.named_parameters() if not n.startswith("backbone.")]
    bb = [p for n, p in pm_net.named_parameters() if n.startswith("backbone.")]
    opt_pm = torch.optim.AdamW([{"params": bb, "lr": 2e-5}, {"params": head, "lr": 2e-4}],
                               weight_decay=1e-4)
    dl_head = [p for n, p in dl_net.named_parameters() if not n.startswith("backbone.")]
    dl_bb = [p for n, p in dl_net.named_parameters() if n.startswith("backbone.")]
    opt_dl = torch.optim.AdamW([{"params": dl_bb, "lr": 2e-5}, {"params": dl_head, "lr": 2e-4}],
                               weight_decay=1e-4)

    def pm_loss(pred, truth_t, case_ids, feats_pooled):
        main = mm_case_bin_loss(pred, truth_t, case_ids, 1.0)
        rank = batchwise_ranking_regularizer(feats_pooled, truth_t.unsqueeze(1), 1.0)
        return main + 1.0 * rank, main, rank

    def dl_loss(pred, truth_t, case_ids, feats_pooled):
        # 与 PM 完全相同的逐样本→病例→档位聚合 + 官方 rank 项
        per = torch.nn.functional.smooth_l1_loss(pred, truth_t, beta=1.0, reduction="none")
        case_truth = {}
        groups = {}
        for i, c in enumerate(case_ids):
            c = str(c)
            groups.setdefault(c, []).append(per[i])
            case_truth[c] = float(truth_t[i].detach())
        bins = {}
        for c, vals in groups.items():
            t = case_truth[c]
            b = 10 if t >= 10 else int(t)
            bins.setdefault(b, []).append(torch.stack(vals).mean())
        main = torch.stack([torch.stack(v).mean() for v in bins.values()]).mean()
        rank = batchwise_ranking_regularizer(feats_pooled, truth_t.unsqueeze(1), 1.0)
        return main + 1.0 * rank, main, rank

    print(f"对齐探针：{N_STEPS} 步（batch=16，同批同权重同损失代码）")
    first_diff = None
    for step in range(N_STEPS):
        batch_idx = idx[step * 16:(step + 1) * 16]
        b = pm_collate([ds_pm[i] for i in batch_idx])
        b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
        random.seed(seed * 1000 + step)  # 同步 dedup 抽取

        feats = pm_net.backbone(b["rgb_views"].flatten(0, 1)).reshape(16, 2, 2048)
        pred_pm, pooled_pm = pm_net.forward_feat(b, feats)
        loss_pm, main_pm, rank_pm = pm_loss(pred_pm, b["truth"], b["case_id"], pooled_pm)
        opt_pm.zero_grad(set_to_none=True)
        loss_pm.backward()
        torch.nn.utils.clip_grad_norm_(pm_net.parameters(), 1.0)
        gnorm_pm = float(sum(p.grad.norm()**2 for p in pm_net.parameters() if p.grad is not None) ** 0.5)
        opt_pm.step()

        random.seed(seed * 1000 + step)
        pred_dl, pooled_dl = dl_net(b)
        loss_dl, main_dl, rank_dl = dl_loss(pred_dl, b["truth"], b["case_id"], pooled_dl)
        opt_dl.zero_grad(set_to_none=True)
        loss_dl.backward()
        torch.nn.utils.clip_grad_norm_(dl_net.parameters(), 1.0)
        opt_dl.step()

        if step % 4 == 0 or first_diff is not None:
            pass
        dl = float((loss_pm - loss_dl).abs())
        pd_ = float((pred_pm - pred_dl).abs().max())
        pdl = float((pooled_pm - pooled_dl).abs().max())
        if step % 4 == 0:
            print(f"  step{step:>3}: loss {float(loss_pm):.5f}/{float(loss_dl):.5f} "
                  f"(d={dl:.2e})  main {float(main_pm):.4f}/{float(main_dl):.4f}  "
                  f"rank {float(rank_pm):.4f}/{float(rank_dl):.4f}  "
                  f"pred_d={pd_:.2e} pooled_d={pdl:.2e}", flush=True)
        if dl > 1e-6 and first_diff is None:
            first_diff = step
            print(f"  >>> 首个分歧 step={step}: loss_pm={float(loss_pm):.6f} loss_dl={float(loss_dl):.6f}")
    print(f"首个分歧步: {first_diff if first_diff is not None else '无（24 步内完全一致）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
