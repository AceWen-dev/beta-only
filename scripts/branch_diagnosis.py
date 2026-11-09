#!/usr/bin/env python3
"""分支诊断（D-0044）：β 分支 / 直接分支 / 融合输出，在「训练集内」与「测试集」上的逐档误差。

回答两个问题：
1. 2–3mm 的高估是「训练没学会」（训练集内也错）还是「留出病例泛化不好」（训练集内准确）；
2. 融合（门控）是否把小息肉预测抬高——即 β/直接两支是否比融合更准。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig, ProjectConfig  # noqa: E402
from core.data import make_loader  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402
from dlef.config import load_config  # noqa: E402

TEST_SPLITS = {"B": "test_b", "D": "test_d"}


def branches(net, batch):
    n = batch["D"].shape[0]
    feats = net.backbone(batch["rgb_views"].flatten(0, 1)).reshape(n, net.views, 2048)
    rgb = net.rgb_proj(feats) + net.view_emb + net.modality[0]
    b2, nv, c, hd, wd = batch["depth"].shape
    dep = net.depth_net(batch["depth"].reshape(b2 * nv, c, hd, wd)).reshape(b2, nv, 64)
    dep_tok = net.depth_proj(dep) + net.view_emb[:nv] + net.modality[1]
    geom = net.geom_proj(batch["scalars"])
    tok = torch.cat([rgb, dep_tok, geom.unsqueeze(1)], dim=1)
    out, _ = net.attn(tok, tok, tok)
    tok = net.norm(tok + out)
    pooled = tok.mean(dim=1)
    dg = net.inject(dep_tok.mean(dim=1))
    fb = torch.cat([pooled, geom], dim=1) + torch.cat([dg, torch.zeros_like(geom)], dim=1)
    z = net.head_beta(fb).squeeze(1)
    beta = net.beta_ref * torch.exp(net.k_bound * torch.tanh(z / net.k_bound))
    pred_beta = beta * batch["D"] * batch["S"]
    pred_dir = F.softplus(net.head_dir(tok[:, : net.views].mean(dim=1)).squeeze(1))
    wg = torch.sigmoid(net.gate(torch.cat([pooled, geom], dim=1)).squeeze(1))
    return pred_beta, pred_dir, wg


def measure_loader(net, loader_iter):
    P, PD, W, T = [], [], [], []
    from torch.utils.data import DataLoader as DL

    with torch.inference_mode():
        for b in loader_iter:
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            pb, pd, wg = branches(net, b)
            P.extend(pb.cpu().tolist())
            PD.extend(pd.cpu().tolist())
            W.extend(wg.cpu().tolist())
            T.extend(b["y"].cpu().tolist())
    return map(np.asarray, (P, PD, W, T))


def per_bin(truth, pred):
    return {
        k: (int(((truth >= k) & (truth < k + 1)).sum()),
            float(np.mean(np.abs(pred[(truth >= k) & (truth < k + 1)]
                                 - truth[(truth >= k) & (truth < k + 1)]))))
        for k in range(10) if ((truth >= k) & (truth < k + 1)).any()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to("cuda").eval()
    data_cfg = load_config(run / "config.resolved.json", ProjectConfig).schema.data
    seed = json.loads((run / "manifest.json").read_text())["seed"]

    out = {}
    # 训练集内（该 run 的实际训练成员，identity 变换）
    train_loader = make_loader(data_cfg, "train", seed, shuffle=False).loader
    P, PD, W, T = measure_loader(net, train_loader)
    FU = W * P + (1 - W) * PD
    out["train"] = {"n": int(len(T)), "beta": per_bin(T, P), "direct": per_bin(T, PD),
                    "fused": per_bin(T, FU), "gate_mean": float(np.mean(W)),
                    "gate_by_bin": {k: float(np.mean(W[(T >= k) & (T < k + 1)]))
                                    for k in range(10) if ((T >= k) & (T < k + 1)).any()}}
    # 测试
    rows_all = json.loads((DATA / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
    for tag, split in TEST_SPLITS.items():
        rows = [r for r in rows_all if r["split"] == split]
        ds = V11FramesDataset(rows, DATA, 2, None)

        def collate(items):
            return {
                **{k: torch.stack([x[k] for x in items]) for k in
                   ("rgb_views", "depth", "scalars", "D", "S", "y")},
                "case_id": [x["case_id"] for x in items],
            }

        loader = DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate)
        P, PD, W, T = measure_loader(net, loader)
        FU = W * P + (1 - W) * PD
        out[tag] = {"n": int(len(T)), "beta": per_bin(T, P), "direct": per_bin(T, PD),
                    "fused": per_bin(T, FU), "gate_mean": float(np.mean(W)),
                    "gate_by_bin": {k: float(np.mean(W[(T >= k) & (T < k + 1)]))
                                    for k in range(10) if ((T >= k) & (T < k + 1)).any()}}

    print(f"run={run.name} ckpt={args.checkpoint}")
    for part in ("train", "B", "D"):
        d = out[part]
        print(f"\n=== {part} (n={d['n']}, gate均值={d['gate_mean']:.2f})")
        print(f"{'档':>6} {'n':>4} {'β分支':>8} {'直接分支':>8} {'实际融合':>9} {'门控w':>7}")
        for k in sorted(d["fused"], key=int):
            nb, mb = d["beta"][k]
            nd, md = d["direct"][k]
            nf, mf = d["fused"][k]
            print(f"{k}-{k+1}: {nf:>4} {mb:>8.2f} {md:>8.2f} {mf:>9.2f} {d['gate_by_bin'][k]:>7.2f}")
    rep = ROOT / "output/reports" / f"branch_diag_{run.name[:15]}.json"
    rep.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
