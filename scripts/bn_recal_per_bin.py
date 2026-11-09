#!/usr/bin/env python3
"""全量 BN 重估 + 逐档捕获（D-0023 后续）：为所有 pre-fix 固定管线实验与
align 三 seed 生成「BN 重估口径」的逐档 MAE 数组，存为权威修正数据。

- pre-fix 固定管线实验：最新 completed run 的 best.pt（污染 BN）→
  训练数据重估 BN → 重测 B/D/R 逐档；
- align 三 seed：同上（best.pt 与 best_ema.pt 各测）；
- valmerge（验证集=训练域，无污染）：直接用记录口径，标注 clean；
- 旧管线运行（ranksim-best/repro）：不处理（历史参照，标注旧口径）。

输出：output/reports/bn_recal_per_bin.json
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

FIXED = (
    "v11-baseline-shuffle",
    "v11-baseline-relloss",
    "v11-baseline-best",
    "v11-backbone-partial-best",
    "v11-backbone-frozen-best",
    "v11-proj-best",
)
SEED_STUDY = {"v11-baseline-valmerge", "v11-ranksim-align"}


def build(run: Path):
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    return net.to("cuda"), ck.epoch


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


def latest_completed(exp: str) -> list[Path]:
    runs = [Path(p) for p in glob.glob(f"output/runs/beta-regression/{exp}/2*")]
    runs = [r for r in runs if json.loads((r / "manifest.json").read_text()).get("status") == "completed"]
    return sorted(runs, key=lambda p: p.name)


def main() -> int:
    man = json.loads((DATA / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
    tr_rows = [r for r in man if r["split"] == "train"]
    tr_ds = V11FramesDataset(tr_rows, DATA, 2, None)
    test_rows = {c: [r for r in man if r["split"] == c] for c in ("test_b", "test_d", "test_r", "test_clean")}

    out: dict = {"_meta": {
        "口径说明": "pre-fix 运行的 best.pt 带被验证集污染的 BN running stats；"
                   "本文件 bn_recal 口径 = 以训练数据重估 BN 统计后的 eval 模式推理，"
                   "为这些实验的权威修正数字。valmerge 的验证集来自训练域，无污染，"
                   "recorded 即权威。旧管线运行（ranksim-best/repro）不在此列，仅作历史参照。"
    }}

    jobs = []
    for exp in FIXED:
        runs = latest_completed(exp)
        if runs:
            jobs.append((exp, runs[-1], ("best",), "bn_recal"))
    for exp in ("v11-ranksim-align", "v11-baseline-valmerge"):
        for run in latest_completed(exp):
            jobs.append((exp, run, ("best", "best_ema") if exp == "v11-ranksim-align" else ("best",),
                         "bn_recal" if exp == "v11-ranksim-align" else "recorded"))

    for exp, run, ckpts, mode in jobs:
        seed = json.loads((run / "manifest.json").read_text())["seed"]
        net, best_epoch = build(run)
        entry = {"seed": seed, "run": run.name, "best_epoch": best_epoch, "口径": mode, "cohorts": {}}
        for c, rows in test_rows.items():
            ps, ts = infer(net, V11FramesDataset(rows, DATA, 2, None))
            entry["cohorts"][c] = {"per_bin": per_bin(ts, ps),
                                   "lt10": float(np.mean(np.abs(ps[ts < 10] - ts[ts < 10]))),
                                   "mae": float(np.mean(np.abs(ps - ts)))}
        if mode == "bn_recal":
            recalibrate(net, tr_ds)
            for c, rows in test_rows.items():
                ps, ts = infer(net, V11FramesDataset(rows, DATA, 2, None))
                entry["cohorts"][c]["bn_recal"] = {
                    "per_bin": per_bin(ts, ps),
                    "lt10": float(np.mean(np.abs(ps[ts < 10] - ts[ts < 10]))),
                    "mae": float(np.mean(np.abs(ps - ts))),
                }
        out.setdefault(exp, []).append({"run": run.name, "seed": seed, **entry})
        print(f"{exp} seed{seed}: "
              + " | ".join(
                  f"{c} lt10={entry['cohorts'][c]['lt10']:.3f}"
                  + (f"→{entry['cohorts'][c]['bn_recal']['lt10']:.3f}" if "bn_recal" in entry["cohorts"][c] else "")
                  for c in entry["cohorts"]), flush=True)
        del net
        torch.cuda.empty_cache()

    out_path = ROOT / "output/reports/bn_recal_per_bin.json"
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
