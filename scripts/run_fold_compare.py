#!/usr/bin/env python3
"""D-0022 预注册对照：纯相对损失(A) vs 混合损失(B)，dev 三折病例级验证。

六次训练（2 损失 × 3 折），seed 固定 20261020；每折用同一病例切分，A/B 直接可比。
不触碰任何测试集：判定指标是每折 val（病例级 + 帧级）的逐档 MAE。

预注册推进条件（写入 STATE §4.1）：
- B 相对 A：dev 7–10mm 最差档 MAE 改善 ≥ 0.5mm（三折平均）；
- B 的 2–3mm 退化 ≤ 0.2mm；
- 报告全部档位，不允许牺牲其他档换达标。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
RUNS = ROOT / "output/runs/beta-regression"
ARMS = (
    ("A_relloss", "input/configs/experiments/v11/dev3-relloss.yaml"),
    ("B_hybrid", "input/configs/experiments/v11/dev3-hybrid.yaml"),
)
FOLDS = (0, 1, 2)
SEED = 20261020


def run_training(arm: str, config: str, fold: int) -> Path:
    before = {p.name for p in RUNS.glob(f"{exp_name(arm)}/2*")}
    cmd = [
        sys.executable, "core/commands/train.py", "--config", config,
        "--tag", "dev3", "--tag", arm,
        "--set", f"data.dev_fold={fold}", "--set", f"seed={SEED}",
    ]
    print(f"[train] {arm} fold{fold}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    new = [p for p in RUNS.glob(f"{exp_name(arm)}/2*") if p.name not in before]
    if len(new) != 1:
        raise RuntimeError(f"expected 1 new run, got {new}")
    return new[0]


def exp_name(arm: str) -> str:
    return f"v11-dev3-{arm.split('_')[1]}"


def infer_fold_val(run_dir: Path, config_file: str, fold: int):
    sys.path.insert(0, str(ROOT))
    from core.config import ModelConfig
    from core.data.datasets.v11_frames import V11FramesDataset
    from core.models.v11 import V11Net

    resolved = json.loads((run_dir / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = [r for r in json.loads((DATA_ROOT / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
           if r["split"] in ("train", "val")]
    from core.data.datasets.v11_frames import assign_dev_folds

    fold_of = assign_dev_folds(dev, resolved["data"]["dev_folds"])
    rows = [r for r in dev if fold_of[str(r["case_id"])] == fold]
    ds = V11FramesDataset(rows, DATA_ROOT, resolved["data"]["views"], None)

    def collate(items):
        return {
            **{k: torch.stack([x[k] for x in items]) for k in
               ("rgb_views", "depth", "scalars", "D", "S", "y")},
            "case_id": [x["case_id"] for x in items],
        }

    device = next(net.parameters()).device
    preds, truths, cases = [], [], []
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, _ = net(batch)
            preds.extend(pred.cpu().tolist())
            truths.extend(batch["y"].cpu().tolist())
            cases.extend(batch["case_id"])
    return np.asarray(preds), np.asarray(truths), cases


def per_bin(truth: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    out = {}
    for k in range(10):
        m = (truth >= k) & (truth < k + 1)
        if m.any():
            out[k] = float(np.mean(np.abs(pred[m] - truth[m])))
    return out


def case_level(truth: np.ndarray, pred: np.ndarray, cases: list[str]) -> dict[int, float]:
    """病例等权：先病例内帧误差均值，再病例间均值（按病例真值分档）。"""
    by_case: dict[str, list[float]] = {}
    case_truth: dict[str, float] = {}
    for t, p, c in zip(truth, pred, cases):
        by_case.setdefault(c, []).append(abs(p - t))
        case_truth[c] = t
    out: dict[int, list[float]] = {}
    for c, errs in by_case.items():
        b = 10 if case_truth[c] >= 10 else int(case_truth[c])
        out.setdefault(b, []).append(float(np.mean(errs)))
    return {b: float(np.mean(v)) for b, v in out.items()}


def worst_tail(bins: dict[int, float]) -> float:
    tail = [v for k, v in bins.items() if 7 <= k < 10]
    return max(tail) if tail else float("nan")


def main() -> int:
    results = {}
    for arm, config in ARMS:
        for fold in FOLDS:
            run_dir = run_training(arm, config, fold)
            pred, truth, cases = infer_fold_val(run_dir, config, fold)
            fb = per_bin(truth, pred)
            cb = case_level(truth, pred, cases)
            results[f"{arm}_fold{fold}"] = {
                "run": run_dir.name,
                "frame_bins": fb,
                "case_bins": cb,
                "frame_worst_tail": worst_tail(fb),
                "case_worst_tail": worst_tail(cb),
                "frame_23": fb.get(2),
                "case_23": cb.get(2),
            }
            print(f"[result] {arm} fold{fold}: frame 7-10 worst={results[f'{arm}_fold{fold}']['frame_worst_tail']:.3f} "
                  f"2-3={results[f'{arm}_fold{fold}']['frame_23']:.3f} | "
                  f"case 7-10 worst={results[f'{arm}_fold{fold}']['case_worst_tail']:.3f} "
                  f"2-3={results[f'{arm}_fold{fold}']['case_23']:.3f}", flush=True)
            torch.cuda.empty_cache()

    # 三折平均 + 预注册条件判定
    summary = {}
    for arm, _ in ARMS:
        folds = [results[f"{arm}_fold{f}"] for f in FOLDS]
        summary[arm] = {
            "frame_worst_tail_mean": float(np.mean([f["frame_worst_tail"] for f in folds])),
            "case_worst_tail_mean": float(np.mean([f["case_worst_tail"] for f in folds])),
            "frame_23_mean": float(np.mean([f["frame_23"] for f in folds if f["frame_23"] is not None])),
            "case_23_mean": float(np.mean([f["case_23"] for f in folds if f["case_23"] is not None])),
        }
    a, b = summary["A_relloss"], summary["B_hybrid"]
    tail_gain_frame = a["frame_worst_tail_mean"] - b["frame_worst_tail_mean"]
    reg_23_frame = b["frame_23_mean"] - a["frame_23_mean"]
    verdict = {
        "tail_gain_frame_mean": tail_gain_frame,
        "regression_23_frame_mean": reg_23_frame,
        "pass_tail_gain_ge_0.5": tail_gain_frame >= 0.5,
        "pass_23_regression_le_0.2": reg_23_frame <= 0.2,
        "advance": tail_gain_frame >= 0.5 and reg_23_frame <= 0.2,
    }
    out = ROOT / "output/reports/fold_compare_relloss_vs_hybrid.json"
    out.write_text(json.dumps({"per_run": results, "summary": summary, "verdict": verdict},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
