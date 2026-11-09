#!/usr/bin/env python3
"""对全部 pre-fix 固定管线实验做 BN 重估 + B/D/R 重测（D-0023 后续）。

每个实验取最新 completed run 的 best.pt：
1. 污染口径（checkpoint 现状 BN stats，eval 模式）推理 B/D/R；
2. 用训练数据重估 BN running stats（momentum=None 累计平均，1 epoch）；
3. 重估后再次推理 B/D/R；
输出前后对照表。不写 eval 记录（诊断性质）。
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
ROOT = Path(".").resolve()
from core.config import ModelConfig
from core.data.datasets.v11_frames import V11FramesDataset
from core.models.v11 import V11Net

EXPERIMENTS = (
    "v11-baseline-shuffle",
    "v11-baseline-relloss",
    "v11-baseline-best",
    "v11-backbone-partial-best",
    "v11-backbone-frozen-best",
    "v11-proj-best",
)


def build(run: Path):
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    return net.to("cuda")


def infer(net, ds):
    def collate(items):
        return {**{k: torch.stack([x[k] for x in items]) for k in ("rgb_views", "depth", "scalars", "D", "S", "y")},
                "case_id": [x["case_id"] for x in items]}
    ps, ts = [], []
    was = net.training
    net.eval()
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=16, num_workers=2, collate_fn=collate):
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            p, _ = net(b)
            ps.extend(p.cpu().tolist())
            ts.extend(b["y"].cpu().tolist())
    net.train(was)
    return np.array(ps), np.array(ts)


def per_bin(ts, ps):
    out = {}
    for k in range(10):
        m = (ts >= k) & (ts < k + 1)
        if m.any():
            out[k] = float(np.mean(np.abs(ps[m] - ts[m])))
    return out


def recalibrate(net, tr_ds):
    for mod in net.modules():
        if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            mod.reset_running_stats()
            mod.momentum = None
    net.train()

    def collate(items):
        return {**{k: torch.stack([x[k] for x in items]) for k in ("rgb_views", "depth", "scalars", "D", "S", "y")},
                "case_id": [x["case_id"] for x in items]}

    with torch.no_grad():
        for b in DataLoader(tr_ds, batch_size=16, num_workers=2, collate_fn=collate):
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            net(b)
    net.eval()


def main() -> int:
    man = json.loads((DATA / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
    tr_rows = [r for r in man if r["split"] == "train"]
    tr_ds = V11FramesDataset(tr_rows, DATA, 2, None)
    test_rows = {c: [r for r in man if r["split"] == c] for c in ("test_b", "test_d", "test_r")}
    summary = {}
    for exp in EXPERIMENTS:
        runs = sorted(Path(p) for p in glob.glob(f"output/runs/beta-regression/{exp}/2*"))
        runs = [r for r in runs if json.loads((r / "manifest.json").read_text()).get("status") == "completed"]
        if not runs:
            print(f"{exp}: 无 completed run，跳过")
            continue
        run = runs[-1]
        seed = json.loads((run / "manifest.json").read_text())["seed"]
        net = build(run)
        res = {}
        for stage in ("污染", "重估"):
            if stage == "重估":
                recalibrate(net, tr_ds)
            for c, rows in test_rows.items():
                ps, ts = infer(net, V11FramesDataset(rows, DATA, 2, None))
                pb = per_bin(ts, ps)
                lt10 = float(np.mean(np.abs(ps[ts < 10] - ts[ts < 10])))
                res.setdefault(c, {})[stage] = {
                    "lt10": round(lt10, 3),
                    "worst": round(max(pb.values()), 3),
                    "fail": {f"{k}-{k+1}": round(v, 2) for k, v in pb.items() if v >= 2},
                }
        summary[exp] = {"seed": seed, "run": run.name, "res": res}
        for c in ("test_b", "test_d", "test_r"):
            a, b = res[c]["污染"], res[c]["重估"]
            print(f"{exp} seed{seed} {c}: lt10 {a['lt10']}→{b['lt10']}  worst {a['worst']}→{b['worst']}  "
                  f"超标 {a['fail']} → {b['fail']}", flush=True)
        del net
        torch.cuda.empty_cache()
    out = ROOT / "output/reports/bn_recal_all.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
