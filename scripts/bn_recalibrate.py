#!/usr/bin/env python3
"""BN 统计重估对照（D-0043）。

背景：修复 eval 模式（D-0041）之前，验证跑在 train 模式，验证批把 BN 的
running mean/var 也更新了（实测 last.pt 的 num_batches_tracked = 5,640 =
5,200 训练批 + 440 验证批）。本脚本固定**该 run 的实际训练成员**（含
val_from_train_cases 合并后的成员），用 identity 变换重新统计 BN，
比较重估前后的逐档误差。

用法：
    python scripts/bn_recalibrate.py --run <run_dir> [--checkpoint final_ema.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig, ProjectConfig  # noqa: E402
from core.data import make_loader  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402
from dlef.config import load_config  # noqa: E402

SPLITS = {"d": "test_d", "b": "test_b", "r": "test_r", "clean": "test_clean"}


def build_net(run: Path, ckpt: str):
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    return net.to("cuda").eval(), ck


def reestimate_bn(net, data_cfg, seed: int) -> int:
    """用该 run 的实际训练成员重估 BN；返回 BN 计数增量。"""
    loader = make_loader(data_cfg, "train", seed, shuffle=False).loader
    before = None
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # 累积平均 = 精确统计
            before = 0
    net.train()
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            net(batch)
            n += 1
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.1
    return n


def evaluate(net, cohorts):
    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    out = {}
    for cohort in cohorts:
        rows = [r for r in rows_all if r["split"] == SPLITS[cohort]]
        ds = V11FramesDataset(rows, DATA, 2, None)

        def collate(items):
            return {
                **{k: torch.stack([x[k] for x in items]) for k in
                   ("rgb_views", "depth", "scalars", "D", "S", "y")},
                "case_id": [x["case_id"] for x in items],
            }

        from torch.utils.data import DataLoader

        ps, ts = [], []
        with torch.inference_mode():
            for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
                b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
                p, _ = net(b)
                ps.extend(p.cpu().tolist())
                ts.extend(b["y"].tolist())
        ps, ts = np.array(ps), np.array(ts)
        bins = {
            k: float(np.mean(np.abs(ps[(ts >= k) & (ts < k + 1)] - ts[(ts >= k) & (ts < k + 1)])))
            for k in range(10) if ((ts >= k) & (ts < k + 1)).any()
        }
        out[cohort] = {
            "per_bin": bins,
            "lt10": float(np.mean(np.abs(ps[ts < 10] - ts[ts < 10]))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--cohorts", default="b,d,r,clean")
    args = ap.parse_args()
    run = Path(args.run).resolve()
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    seed = json.loads((run / "manifest.json").read_text())["seed"]
    resolved = json.loads((run / "config.resolved.json").read_text())
    data_cfg = load_config(run / "config.resolved.json", ProjectConfig).schema.data

    net, ck = build_net(run, args.checkpoint)
    before = evaluate(net, cohorts)
    n = reestimate_bn(net, data_cfg, seed)
    after = evaluate(net, cohorts)

    print(f"run={run.name} ckpt={args.checkpoint} | BN 重估用了 {n} 个训练批")
    print(f"{'cohort':>6} {'档':>6} {'重估前':>8} {'重估后':>8} {'Δ':>7}")
    for c in cohorts:
        for k in sorted(before[c]["per_bin"], key=int):
            a = before[c]["per_bin"][k]
            b = after[c]["per_bin"].get(k, float("nan"))
            print(f"{c.upper():>6} {k}-{int(k)+1}: {a:>8.2f} {b:>8.2f} {b-a:>+7.2f}")
        print(f"{c.upper():>6} {'<10':>6} {before[c]['lt10']:>8.3f} {after[c]['lt10']:>8.3f} "
              f"{after[c]['lt10']-before[c]['lt10']:>+7.3f}")
    out = ROOT / "output/reports" / f"bn_recal_{run.parent.name}_{run.name[:15]}.json"
    out.write_text(json.dumps({"run": run.name, "checkpoint": args.checkpoint,
                               "train_batches": n, "before": before, "after": after},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
