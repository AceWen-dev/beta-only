#!/usr/bin/env python3
"""dev 三折汇总（D-0051 第 2 步）：每个病例只由「没训练过它」的模型预测。

fold0/1/2 的三次训练各自用另外两折训练、用本折验证；把三折的病例级预测并起来，
就得到覆盖全部 dev 池的「out-of-fold」预测。输出：

- 逐帧、病例等权、按来源拆分的逐档 MAE
- CSV：每条 (case, fold) 一行

注意：只读 dev 折（train+val 池），不碰任何测试集。
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

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import (  # noqa: E402
    V11FramesDataset,
    assign_dev_folds,
)
from core.models.v11 import V11Net  # noqa: E402

RUNS = ROOT / "output/runs/beta-regression"


def collate(items):
    return {
        **{
            k: torch.stack([x[k] for x in items])
            for k in ("rgb_views", "depth", "scalars", "D", "S", "y")
        },
        "case_id": [x["case_id"] for x in items],
        "sample_id": [x["sample_id"] for x in items],
    }


def find_run(experiment: str, fold: int) -> Path | None:
    best = None
    for manifest in (RUNS / experiment).glob("*/manifest.json"):
        payload = json.loads(manifest.read_text())
        if payload.get("status") != "completed":
            continue
        resolved = json.loads((manifest.parent / "config.resolved.json").read_text())
        if resolved["data"]["dev_fold"] != fold:
            continue
        if best is None or manifest.parent.name > best.name:
            best = manifest.parent
    return best


def predict(run: Path, rows: list[dict], device: str) -> list[dict]:
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints/final_ema.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to(device).eval()
    ds = V11FramesDataset(rows, DATA, 2, None)
    out = []
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            p, _ = net(b)
            for i, sid in enumerate(b["sample_id"]):
                out.append(
                    {
                        "case_id": str(b["case_id"][i]),
                        "sample_id": str(sid),
                        "truth": float(b["y"][i]),
                        "pred": float(p[i]),
                    }
                )
    del net
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def source_of(case: str) -> str:
    for p in ("company", "newdata", "phantom", "virtual"):
        if case.startswith(p):
            return p
    return case.split("-")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="v11-dev3-g20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/reports/dev3_folds")
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    dev = [r for r in rows_all if r["split"] in ("train", "val")]
    fold_of = assign_dev_folds(dev, 3)

    records = []
    for fold in (0, 1, 2):
        run = find_run(args.experiment, fold)
        if run is None:
            print(f"!! fold{fold}: 没有 completed run")
            return 1
        rows = [r for r in dev if fold_of[str(r["case_id"])] == fold]
        preds = predict(run, rows, device)
        for rec in preds:
            rec["fold"] = fold
        records.extend(preds)
        print(f"fold{fold}: {run.name} | {len(preds)} 帧 / "
              f"{len({p['case_id'] for p in preds})} 病例", flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "oof_frames.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    # 病例级
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_case[r["case_id"]].append(r)
    case_rows = []
    for case, rs in by_case.items():
        truth = float(np.median([r["truth"] for r in rs]))
        case_rows.append(
            {
                "case_id": case,
                "fold": rs[0]["fold"],
                "source": source_of(case),
                "n": len(rs),
                "truth": truth,
                "pred_mean": float(np.mean([r["pred"] for r in rs])),
                "abs_err": float(np.mean([abs(r["pred"] - r["truth"]) for r in rs])),
            }
        )

    def bucket(t: float) -> int:
        return 10 if t >= 10 else int(t)

    print(f"\n全 dev 池 out-of-fold 汇总（{len(case_rows)} 病例 / {len(records)} 帧）")
    print(f"{'档':>5} {'病例':>4} {'帧':>5} {'帧MAE':>7} {'病例MAE':>8}")
    per_bin: dict[int, dict[str, list]] = defaultdict(lambda: {"cases": [], "frames": []})
    for c in case_rows:
        per_bin[bucket(c["truth"])]["cases"].append(c["abs_err"])
        for r in by_case[c["case_id"]]:
            per_bin[bucket(c["truth"])]["frames"].append(abs(r["pred"] - r["truth"]))
    for b in sorted(per_bin):
        d = per_bin[b]
        print(f"{b}-{b+1}: {len(d['cases']):>4} {len(d['frames']):>5} "
              f"{np.mean(d['frames']):>7.2f} {np.mean(d['cases']):>8.2f}")

    lt10 = [c for c in case_rows if c["truth"] < 10]
    print(f"\n<10mm 病例等权 MAE: {np.mean([c['abs_err'] for c in lt10]):.3f}")
    print("\n按来源（病例等权）：")
    for source in sorted({c["source"] for c in case_rows}):
        rows = [c for c in case_rows if c["source"] == source]
        print(f"  {source:10s} n={len(rows):>3}  <10mm MAE="
              f"{np.mean([c['abs_err'] for c in rows if c['truth'] < 10]):.3f}")

    with (out / "oof_cases.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(case_rows[0]))
        w.writeheader()
        for c in case_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    summary = {
        "n_cases": len(case_rows),
        "n_frames": len(records),
        "lt10_case_mae": float(np.mean([c["abs_err"] for c in lt10])),
        "per_bin_case_mae": {
            str(b): float(np.mean(per_bin[b]["cases"])) for b in sorted(per_bin)
        },
        "per_bin_frame_mae": {
            str(b): float(np.mean(per_bin[b]["frames"])) for b in sorted(per_bin)
        },
    }
    (out / "oof_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
