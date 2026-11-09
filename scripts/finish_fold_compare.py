#!/usr/bin/env python3
"""补算 D-0022 三折对照判定（runner 挂掉后从 run 目录离线恢复）。

每臂取每折**最新**的 completed run，eval 模式推理该折 val，
帧平均 + 病例等权并列，按预注册条件出判定。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
RUNS = ROOT / "output/runs/beta-regression"
FOLDS = (0, 1, 2)
ARMS = {
    "A_relloss": "v11-dev3-relloss",
    "B_hybrid": "v11-dev3-hybrid",
}


def latest_completed(experiment: str, fold: int) -> Path | None:
    out = []
    for manifest in (RUNS / experiment).glob("*/manifest.json"):
        payload = json.loads(manifest.read_text())
        if payload.get("status") != "completed":
            continue
        resolved = json.loads((manifest.parent / "config.resolved.json").read_text())
        if resolved["data"]["dev_fold"] == fold:
            out.append((manifest.parent, payload.get("started_at", "")))
    return max(out)[0] if out else None


def infer(net, rows):
    from core.data.datasets.v11_frames import V11FramesDataset

    ds = V11FramesDataset(rows, DATA_ROOT, 2, None)

    def collate(items):
        return {
            **{k: torch.stack([x[k] for x in items]) for k in
               ("rgb_views", "depth", "scalars", "D", "S", "y")},
            "case_id": [x["case_id"] for x in items],
        }

    loader = DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate)
    device = next(net.parameters()).device
    preds, truths, cases = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, _ = net(batch)
            preds.extend(pred.cpu().tolist())
            truths.extend(batch["y"].cpu().tolist())
            cases.extend(batch["case_id"])
    return np.asarray(preds), np.asarray(truths), cases


def per_bin(truth, pred):
    out = {}
    for k in range(10):
        m = (truth >= k) & (truth < k + 1)
        if m.any():
            out[k] = float(np.mean(np.abs(pred[m] - truth[m])))
    return out


def case_level(truth, pred, cases):
    by = {}
    ct = {}
    for t, p, c in zip(truth, pred, cases):
        by.setdefault(c, []).append(abs(p - t))
        ct[c] = t
    agg = {}
    for c, errs in by.items():
        b = 10 if ct[c] >= 10 else int(ct[c])
        agg.setdefault(b, []).append(float(np.mean(errs)))
    return {b: float(np.mean(v)) for b, v in agg.items()}


def worst_tail(bins):
    tail = [v for k, v in bins.items() if 7 <= k < 10]
    return max(tail) if tail else float("nan")


def main() -> int:
    import cv2  # noqa: F401  (确保 cv2 已加载)
    from core.config import ModelConfig
    from core.data.datasets.v11_frames import V11FramesDataset, assign_dev_folds
    from core.models.v11 import V11Net

    rows_all = json.loads(
        (DATA_ROOT / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    dev = [r for r in rows_all if r["split"] in ("train", "val")]
    fold_of = assign_dev_folds(dev, 3)

    results = {}
    for arm, experiment in ARMS.items():
        for fold in FOLDS:
            run_dir = latest_completed(experiment, fold)
            if run_dir is None:
                print(f"!! {arm} fold{fold}: 无 completed run")
                continue
            resolved = json.loads((run_dir / "config.resolved.json").read_text())
            net = V11Net(ModelConfig.from_dict(resolved["model"]))
            ck = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu",
                            weights_only=False)
            net.load_state_dict(ck.model)
            net = net.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            rows = [r for r in dev if fold_of[str(r["case_id"])] == fold]
            pred, truth, cases = infer(net, rows)
            fb = per_bin(truth, pred)
            cb = case_level(truth, pred, cases)
            key = f"{arm}_fold{fold}"
            results[key] = {
                "run": run_dir.name,
                "frame_bins": fb,
                "case_bins": cb,
                "frame_worst_tail": worst_tail(fb),
                "case_worst_tail": worst_tail(cb),
                "frame_23": fb.get(2),
                "case_23": cb.get(2),
            }
            print(f"[{key}] {run_dir.name}  frame: tail={results[key]['frame_worst_tail']:.3f} "
                  f"2-3={results[key]['frame_23']:.3f} | case: tail={results[key]['case_worst_tail']:.3f} "
                  f"2-3={results[key]['case_23']:.3f}", flush=True)
            del net
            torch.cuda.empty_cache()

    summary = {}
    for arm in ARMS:
        folds = [results[f"{arm}_fold{f}"] for f in FOLDS if f"{arm}_fold{f}" in results]
        summary[arm] = {
            "frame_worst_tail_mean": float(np.mean([f["frame_worst_tail"] for f in folds])),
            "case_worst_tail_mean": float(np.mean([f["case_worst_tail"] for f in folds])),
            "frame_23_mean": float(np.mean([f["frame_23"] for f in folds])),
            "case_23_mean": float(np.mean([f["case_23"] for f in folds])),
        }
    a, b = summary["A_relloss"], summary["B_hybrid"]
    tail_gain = a["frame_worst_tail_mean"] - b["frame_worst_tail_mean"]
    reg_23 = b["frame_23_mean"] - a["frame_23_mean"]
    verdict = {
        "tail_gain_frame_mean": tail_gain,
        "regression_23_frame_mean": reg_23,
        "pass_tail_gain_ge_0.5": tail_gain >= 0.5,
        "pass_23_regression_le_0.2": reg_23 <= 0.2,
        "advance": tail_gain >= 0.5 and reg_23 <= 0.2,
    }
    out = ROOT / "output/reports/fold_compare_relloss_vs_hybrid.json"
    out.write_text(json.dumps({"per_run": results, "summary": summary,
                               "verdict": verdict}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
