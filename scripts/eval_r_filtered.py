#!/usr/bin/env python3
"""Evaluate one run on REAL-colon R with truth>=2mm (benchmark2 D-0076 filter).

Runs with the code on PYTHONPATH: main repo for gate/diranchor/base/ordinal,
the span-normalization worktree for the span-norm run (it has span_reference_width).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402


def collate(items):
    keys = ["rgb_views", "depth", "scalars", "D", "S", "y"]
    for key in ("S_norm", "W"):
        if key in items[0]:
            keys.append(key)
    return {
        **{k: torch.stack([x[k] for x in items]) for k in keys},
        "case_id": [x["case_id"] for x in items],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--min-truth", type=float, default=2.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run)
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).eval()

    manifest = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )
    rows = [
        r for r in manifest["rows"]
        if r["split"] == "test_r" and float(r["truth_mm"]) >= args.min_truth
    ]
    ds = V11FramesDataset(
        rows, DATA, int(resolved.get("data", {}).get("views", 2)), None,
        input_size=int(resolved.get("data", {}).get("input_size", 224)),
    )
    preds, targets = [], []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, _ = net(batch)
            preds.extend(pred.cpu().tolist())
            targets.extend(batch["y"].cpu().tolist())
    preds, targets = np.array(preds), np.array(targets)
    bins = {
        f"{k}-{k+1}": float(np.mean(np.abs(preds[(targets >= k) & (targets < k+1)]
                                           - targets[(targets >= k) & (targets < k+1)])))
        for k in range(10) if ((targets >= k) & (targets < k+1)).any()
    }
    result = {
        "run": run.name,
        "checkpoint": args.checkpoint,
        "min_truth_mm": args.min_truth,
        "n_frames": len(rows),
        "per_bin": bins,
        "lt10": float(np.mean(np.abs(preds[targets < 10] - targets[targets < 10]))),
        "mae": float(np.mean(np.abs(preds - targets))),
        "worst_bin": max(bins.values()),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
