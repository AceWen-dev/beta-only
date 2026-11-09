#!/usr/bin/env python3
"""DIAGNOSTIC — SWA（权重平均）能否压下 B/D 8-9 档的轻微超标。

把 baseline-shuffle 已有的 epoch-0030/0035/0040 权重做算术平均，在内存里
推理 B/D（不写 checkpoint、不写 eval 记录），逐档对比单个 best.pt。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
RUN = ROOT / "output/runs/beta-regression/v11-baseline-shuffle/20261031-020209_4147d3b9_s20261020_cbdbaf"
SPLITS = {"B": "test_b", "D": "test_d"}
EPOCHS = ("epoch-0030", "epoch-0035", "epoch-0040")


def build_model(device: str):
    import sys

    sys.path.insert(0, str(ROOT))
    from core.config import ModelConfig
    from core.models.v11 import V11Net

    resolved = json.loads((RUN / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    states = []
    for name in EPOCHS:
        ck = torch.load(RUN / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
        states.append(dict(ck.model))
    avg = {}
    for key in states[0]:
        vals = [s[key].float() for s in states]
        avg[key] = torch.mean(torch.stack(vals), dim=0).to(states[0][key].dtype)
    net.load_state_dict(avg)
    return net.to(device).eval()


def infer(net, split: str, device: str):
    from core.data.datasets.v11_frames import V11FramesDataset

    manifest = json.loads(
        (DATA_ROOT / "curated/v11_frames/single-v2/manifest.json").read_text()
    )
    rows = [r for r in manifest["rows"] if r["split"] == SPLITS[split]]
    ds = V11FramesDataset(rows, DATA_ROOT, 2, None)

    def collate(items):
        return {
            **{k: torch.stack([x[k] for x in items]) for k in
               ("rgb_views", "depth", "scalars", "D", "S", "y")},
            "case_id": [x["case_id"] for x in items],
        }

    loader = DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate)
    preds, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, _ = net(batch)
            preds.extend(pred.cpu().tolist())
            truths.extend(batch["y"].cpu().tolist())
    return np.asarray(preds), np.asarray(truths)


def per_bin(truth: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    out = {}
    for k in range(10):
        m = (truth >= k) & (truth < k + 1)
        if m.any():
            out[k] = float(np.mean(np.abs(pred[m] - truth[m])))
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(device)
    print(f"SWA over {EPOCHS}（权重算术平均，未训练）")
    for split in ("B", "D"):
        pred, truth = infer(net, split, device)
        bins = per_bin(truth, pred)
        worst = max(bins.values())
        fails = {f"{k}-{k+1}": round(v, 3) for k, v in bins.items() if v >= 2}
        lt10 = float(np.mean(np.abs(pred[truth < 10] - truth[truth < 10])))
        print(f"{split}: worst={worst:.3f}  lt10={lt10:.3f}  未达标档: {fails}")
    print("对照 best.pt（单点）: B worst=3.124 lt10=1.316 未达标{2-3:2.93,7-8:3.12}; "
          "D worst=2.521 lt10=1.266 未达标{2-3:2.46,8-9:2.17}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
