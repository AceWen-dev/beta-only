#!/usr/bin/env python3
"""病例级误差审计（D-0047）：dev 折 / 测试集按病例拆解，并区分来源。

输出：
- CSV：每个病例一行（split, source, case_id, n, truth, pred_mean, signed_err,
  within_case_std, abs_err）
- 汇总：按 (split, source, bin) 的帧级 MAE，以及按病例等权的 MAE
- 复核清单：|signed_err| ≥ 阈值 的小档病例

不修改任何记录；只读 checkpoint + 数据集。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig, ProjectConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset, assign_dev_folds  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402
from dlef.config import load_config  # noqa: E402


def source_of(case_id: str) -> str:
    c = str(case_id)
    for p in ("company", "newdata", "phantom", "virtual"):
        if c.startswith(p):
            return p
    return c.split("-")[0]


def predict(net, rows):
    ds = V11FramesDataset(rows, DATA, 2, None)

    def collate(items):
        return {
            **{k: torch.stack([x[k] for x in items]) for k in
               ("rgb_views", "depth", "scalars", "D", "S", "y")},
            "case_id": [x["case_id"] for x in items],
            "sample_id": [x["sample_id"] for x in items],
        }

    ps, ts, cs, ids = [], [], [], []
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            p, _ = net(b)
            ps.extend(p.cpu().tolist())
            ts.extend(b["y"].tolist())
            cs.extend(b["case_id"])
            ids.extend(b["sample_id"])
    return np.array(ps), np.array(ts), cs, ids


def audit(rows, pred, truth, cases, ids, split):
    by = defaultdict(lambda: {"p": [], "t": [], "ids": []})
    for p, t, c, sid in zip(pred, truth, cases, ids):
        by[str(c)]["p"].append(float(p))
        by[str(c)]["t"].append(float(t))
        by[str(c)]["ids"].append(sid)
    out = []
    for c, d in by.items():
        ts = np.array(d["t"])
        ps = np.array(d["p"])
        out.append({
            "split": split,
            "source": source_of(c),
            "case_id": c,
            "n": len(ts),
            "truth_med": float(np.median(ts)),
            "truth_min": float(ts.min()),
            "truth_max": float(ts.max()),
            "label_span": float(ts.max() - ts.min()),
            "pred_mean": float(ps.mean()),
            "signed_err": float(ps.mean() - np.median(ts)),
            "within_std": float(ps.std()),
            "abs_err": float(np.mean(np.abs(ps - ts))),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--bins", default="2", help="逗号分隔的档位，用于复核清单")
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()
    run = Path(args.run).resolve()
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to("cuda").eval()
    data_cfg = load_config(run / "config.resolved.json", ProjectConfig).schema.data

    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    dev = [r for r in rows_all if r["split"] in ("train", "val")]
    fold = assign_dev_folds(dev, data_cfg.dev_folds or 3)
    val_rows = [r for r in dev if fold[str(r["case_id"])] == args.fold]

    records = []
    p, t, c, i = predict(net, val_rows)
    records += audit(val_rows, p, t, c, i, f"dev_fold{args.fold}")
    for tag, split in (("D", "test_d"), ("B", "test_b")):
        rows = [r for r in rows_all if r["split"] == split]
        p, t, c, i = predict(net, rows)
        records += audit(rows, p, t, c, i, tag)

    # CSV
    out = ROOT / "output/reports" / f"case_audit_{run.name[:15]}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["split"], x["source"], -abs(x["signed_err"]))):
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"CSV: {out}")

    # 汇总：按 split × source × 档（帧级 + 病例等权）
    print(f"\n{'split':10s} {'source':9s} {'档':>4} {'帧数':>4} {'病例':>4} "
          f"{'帧MAE':>7} {'病例MAE':>8} {'平均偏差':>9}")
    agg = defaultdict(lambda: {"frames": 0, "cases": [], "errs": [], "biases": []})
    for r in records:
        k = int(r["truth_med"])
        key = (r["split"], r["source"], k)
        agg[key]["frames"] += r["n"]
        agg[key]["cases"].append(r["case_id"])
        agg[key]["errs"].append(r["abs_err"])
        agg[key]["biases"].append(r["signed_err"])
    for key in sorted(agg):
        a = agg[key]
        print(f"{key[0]:10s} {key[1]:9s} {key[2]:>4} {a['frames']:>4} {len(a['cases']):>4} "
              f"{'':>7} {np.mean(a['errs']):>8.2f} {np.mean(a['biases']):>+9.2f}")

    # 复核清单
    bins = {int(x) for x in args.bins.split(",") if x.strip()}
    print(f"\n复核清单（档∈{sorted(bins)}，|有符号误差| ≥ {args.threshold}）：")
    print(f"{'split':10s} {'source':9s} {'case_id':24s} {'n':>3} {'truth':>6} "
          f"{'pred':>6} {'偏差':>6} {'帧内std':>7}")
    flags = [r for r in records if int(r["truth_med"]) in bins
             and abs(r["signed_err"]) >= args.threshold]
    for r in sorted(flags, key=lambda x: -abs(x["signed_err"])):
        print(f"{r['split']:10s} {r['source']:9s} {r['case_id'][:24]:24s} {r['n']:>3} "
              f"{r['truth_med']:>6.1f} {r['pred_mean']:>6.2f} {r['signed_err']:>+6.2f} "
              f"{r['within_std']:>7.2f}")
    print(f"\n共 {len(flags)} 个病例进入复核清单")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
