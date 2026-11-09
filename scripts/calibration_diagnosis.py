#!/usr/bin/env python3
"""DIAGNOSTIC — 标定能否把 B/D 的超标档压进 2mm。

在 **val** 上拟合 isotonic 修正（pred→truth 的单调映射），应用到 B/D test，
逐档对比标定前后 MAE。不训练、不写 eval 记录；结果只进诊断报告。

顺带输出 B 7-8 档（n=5，系统性低估）的逐样本明细，供个案检查。
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
SPLITS = {"val": "val", "B": "test_b", "D": "test_d", "R": "test_r", "clean": "test_clean"}


def pava(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Isotonic（递增）回归的 PAVA 实现；按 x 排序后拟合 y 的单调阶梯。"""
    order = np.argsort(x)
    x, y = x[order], y[order]
    # 每块： (weight, value)；违反递增就合并
    blocks = [[1.0, v] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1] + 1e-12:
            w = blocks[i][0] + blocks[i + 1][0]
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w
            blocks[i:i + 2] = [[w, v]]
            i = max(0, i - 1)
        else:
            i += 1
    fitted, idx = [], 0
    for w, v in blocks:
        n = int(round(w))
        fitted.extend([v] * n)
        idx += n
    return np.asarray(fitted)


class Calibrator:
    """val 上拟合的 pred→truth 单调映射；区间外 clip。"""

    def fit(self, pred: np.ndarray, truth: np.ndarray) -> "Calibrator":
        order = np.argsort(pred)
        self.x = pred[order]
        self.y = pava(self.x, truth[order])
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        return np.interp(pred, self.x, self.y, left=self.y[0], right=self.y[-1])


def build_model(device: str):
    import sys

    sys.path.insert(0, str(ROOT))
    from core.config import ModelConfig
    from core.models.v11 import V11Net

    resolved = json.loads((RUN / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(RUN / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
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
            "image": [x["sample_id"] for x in items],
        }

    loader = DataLoader(ds, batch_size=16, num_workers=2, collate_fn=collate)
    preds, truths, spans = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, _ = net(batch)
            preds.extend(pred.cpu().tolist())
            truths.extend(batch["y"].cpu().tolist())
            spans.extend(batch["S"].cpu().tolist())
    return np.asarray(preds), np.asarray(truths), np.asarray(spans), [r["sample_id"] for r in rows]


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

    val_p, val_t, _, _ = infer(net, "val", device)
    cal = Calibrator().fit(val_p, val_t)
    print(f"val n={len(val_p)}  标定点范围 [{cal.x[0]:.2f}, {cal.x[-1]:.2f}]")

    report = ["# DIAGNOSTIC — isotonic 标定（val 拟合）对 B/D 的作用\n",
              f"checkpoint: {RUN.name} best.pt（未写 eval 记录，仅供分析）\n"]
    detail = {}
    for split in ("B", "D", "R", "clean"):
        raw_p, truth, spans, ids = infer(net, split, device)
        cal_p = cal.predict(raw_p)
        before, after = per_bin(truth, raw_p), per_bin(truth, cal_p)
        report.append(f"\n## {split}（n={len(truth)}）\n")
        report.append("| bin | n | MAE raw | MAE cal | Δ |")
        report.append("|---|---|---:|---:|---:|")
        for k in range(10):
            if k not in before:
                continue
            report.append(
                f"| {k}-{k+1} | {int(((truth>=k)&(truth<k+1)).sum())} "
                f"| {before[k]:.3f} | {after[k]:.3f} | {after[k]-before[k]:+.3f} |"
            )
        bf = max(before.values()); af = max(after.values())
        report.append(f"\nworst_bin: raw **{bf:.3f}** → cal **{af:.3f}**"
                      f"  |  lt10: raw {np.mean(np.abs(raw_p[truth<10]-truth[truth<10])):.3f}"
                      f" → cal {np.mean(np.abs(cal_p[truth<10]-truth[truth<10])):.3f}\n")
        detail[split] = (truth, raw_p, cal_p, spans, ids, before, after)

    # B 7-8 个案（n=5，系统性低估）
    truth, raw_p, cal_p, spans, ids, _, _ = detail["B"]
    m = (truth >= 7) & (truth < 8)
    report.append("\n## B 7-8 档逐样本（n=5，标定前系统性低估）\n")
    report.append("| sample | truth | pred_raw | pred_cal | span_px |")
    report.append("|---|---:|---:|---:|---:|")
    for i in np.where(m)[0]:
        report.append(f"| {ids[i]} | {truth[i]:.2f} | {raw_p[i]:.2f} | {cal_p[i]:.2f} | {spans[i]:.1f} |")

    out = ROOT / "output/reports/calibration_diagnosis.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
