#!/usr/bin/env python3
"""benchmark2 全量可视复核：β-only 推理 + 原图检测框 + GT/Pred 标注。

目录结构（output/review/visioncheck/）：
    <cohort>/<k>-<k+1>mm/mae_lt2/0001__e0.31__t4.0_p4.3__<case>__<stem>.jpg
    <cohort>/<k>-<k+1>mm/mae_ge2/...

- cohort = benchmark2 的 company / b / r，成员过滤与 benchmark2.json 一致
  （B 排除 7–8mm，R 排除 1–2mm）。
- k-(k+1)mm 表示 truth ∈ [k, k+1)；每档内按单帧 MAE <2 / >=2 分子目录。
- 文件名前缀为档内按误差从大到小的序号，便于先看最差的。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import (  # noqa: E402
    V11FramesDataset,
    detect_image_area,
)
from core.models.v11 import V11Net  # noqa: E402

RUN = (ROOT / "output/runs/beta-regression/v11-company-diranchor40-betaonly"
       / "20261106-075001_f83b8b9b_s20261021_48542a")
SUITE = ROOT / "input/benchmarks/benchmark2.json"
OUT = ROOT / "output/review/visioncheck"
KEYS = ("rgb_views", "depth", "scalars", "D", "S", "S_norm", "W", "y")
SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def data_root() -> Path:
    root = Path(os.environ.get("DLEF_DATA_ROOT", ""))
    if not root.is_dir():
        raise SystemExit("set DLEF_DATA_ROOT")
    return root


def load_rows(root: Path, spec: dict) -> list[dict]:
    rows = json.loads(
        (root / f"curated/v11_frames/{spec['version']}/manifest.json").read_text()
    )["rows"]
    excluded = {int(b) for b in spec.get("exclude_bins", [])}
    min_truth = float(spec.get("min_truth_mm", 0.0))
    return [
        r for r in rows
        if r["split"] == spec["split"]
        and float(r["truth_mm"]) >= min_truth
        and int(float(r["truth_mm"])) not in excluded
    ]


def build_net(resolved: dict, ckpt: Path, device: torch.device) -> V11Net:
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    return net.to(device).eval()


def collate(items: list[dict]) -> dict:
    out = {k: torch.stack([x[k] for x in items]) for k in KEYS}
    out["sample_id"] = [x["sample_id"] for x in items]
    return out


def infer(net, rows, resolved, root, device, batch_size=16, workers=4):
    views = int(resolved["data"]["views"])
    size = int(resolved["data"].get("input_size", 224))
    ds = V11FramesDataset(rows, root, views, None, input_size=size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        collate_fn=collate)
    preds = []
    with torch.inference_mode():
        for i, b in enumerate(loader):
            b = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in b.items()}
            p, _ = net(b)
            preds.extend(p.detach().cpu().tolist())
            if (i + 1) % 20 == 0:
                print(f"    batch {i + 1}/{len(loader)}", flush=True)
    if len(preds) != len(rows):
        raise RuntimeError(f"prediction count {len(preds)} != rows {len(rows)}")
    return preds


def draw(root: Path, row: dict, pred: float, out_path: Path) -> None:
    im = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(root / row["image"])
    crop = row.get("crop_box")
    if crop:
        x0, y0 = int(crop[0]), int(crop[1])
    else:
        # 与 V11FramesDataset 完全一致：它把 RGB 数组传给 detect_image_area
        x0, y0, _, _ = detect_image_area(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = (float(v) for v in row["bbox_content"])
    bx1, by1 = int(round(x0 + x1)), int(round(y0 + y1))
    bx2, by2 = int(round(x0 + x2)), int(round(y0 + y2))
    truth = float(row["truth_mm"])
    err = abs(pred - truth)
    h, w = im.shape[:2]
    bx1, bx2 = max(0, min(bx1, w - 1)), max(0, min(bx2, w - 1))
    by1, by2 = max(0, min(by1, h - 1)), max(0, min(by2, h - 1))
    color = (70, 180, 70) if err < 2 else (50, 50, 220)
    thick = max(2, int(round(min(h, w) / 350)))
    cv2.rectangle(im, (bx1, by1), (bx2, by2), color, thick)

    label = f"GT {truth:.1f}mm | Pred {pred:.2f}mm | Err {err:.2f}mm"
    scale = max(0.6, min(h, w) / 900.0)
    tthick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, tthick)
    tx = min(max(0, bx1), max(0, w - tw - 10))
    ty = by1 - 8 - th
    if ty < th + 4:
        ty = by2 + 8
    ty = min(ty, h - 6)
    cv2.rectangle(im, (tx - 3, ty - 3), (tx + tw + 5, ty + th + 5),
                  (0, 0, 0), -1)
    cv2.putText(im, label, (tx, ty + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), tthick, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--cohorts", default="company,b,r")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="每队列最多 N 帧（冒烟）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = data_root()
    run = Path(args.run)
    resolved = json.loads((run / "config.resolved.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net(resolved, run / "checkpoints" / args.checkpoint, device)
    suite = json.loads(SUITE.read_text())["members"]
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    rows_out, summary = [], {}
    for cohort in cohorts:
        rows = load_rows(root, suite[cohort])
        if args.limit:
            rows = rows[: args.limit]
        row_by_id = {r["sample_id"]: r for r in rows}
        print(f"[{cohort}] {len(rows)} frames", flush=True)
        preds = infer(net, rows, resolved, root, device,
                      args.batch_size, args.workers)
        per_folder = {}
        for row, pred in zip(rows, preds):
            truth = float(row["truth_mm"])
            err = abs(pred - truth)
            k = int(truth)
            group = "mae_lt2" if err < 2 else "mae_ge2"
            rec = {
                "cohort": cohort,
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "truth_mm": truth,
                "pred_mm": float(pred),
                "abs_err_mm": err,
                "size_bin": f"{k}-{k + 1}mm",
                "mae_group": group,
                "image": row["image"],
            }
            per_folder.setdefault((k, group), []).append(rec)
            rows_out.append(rec)
        for k in sorted({k for k, _ in per_folder}):
            for group in ("mae_lt2", "mae_ge2"):
                (out_root / cohort / f"{k}-{k + 1}mm" / group).mkdir(
                    parents=True, exist_ok=True
                )
        for (k, group), recs in per_folder.items():
            recs.sort(key=lambda r: r["abs_err_mm"], reverse=True)
            for i, rec in enumerate(recs, 1):
                stem = Path(rec["image"]).stem
                name = (f"{i:04d}__e{rec['abs_err_mm']:.2f}"
                        f"__t{rec['truth_mm']:.1f}_p{rec['pred_mm']:.2f}"
                        f"__{SAFE.sub('_', rec['case_id'])}"
                        f"__{SAFE.sub('_', stem)}.jpg")
                rel = Path(cohort) / rec["size_bin"] / group / name
                row = row_by_id[rec["sample_id"]]
                draw(root, row, rec["pred_mm"], out_root / rel)
                rec["annotated"] = str(rel)
        cohort_rows = [r for r in rows_out if r["cohort"] == cohort]
        lt10 = float(np.mean([r["abs_err_mm"] for r in cohort_rows
                              if r["truth_mm"] < 10]))
        summary[cohort] = {
            "frames": len(rows), "lt10": lt10,
            "mae": float(np.mean([r["abs_err_mm"] for r in cohort_rows])),
        }
        print(f"[{cohort}] lt10={lt10:.4f} mae={summary[cohort]['mae']:.4f}",
              flush=True)

    with (out_root / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)

    lines = [
        "# visioncheck — benchmark2 可视复核",
        "",
        f"- run: `{run}`",
        f"- checkpoint: `{args.checkpoint}` (final EMA)",
        "- 结构：`<cohort>/<k>-<k+1>mm/mae_lt2|mae_ge2/`；"
        "文件按档内误差降序编号。",
        "- 判定：单帧 MAE <2mm 入 `mae_lt2`，≥2mm 入 `mae_ge2`（2.0 边界归后者）。",
        "- benchmark2 口径：B 排除 7–8mm；R 排除 1–2mm。",
        "",
    ]
    for cohort in cohorts:
        s = summary[cohort]
        lines += [f"## {cohort}", "",
                  f"- frames: {s['frames']}，<10mm MAE: **{s['lt10']:.4f}**，"
                  f"全帧 MAE: {s['mae']:.4f}", "",
                  "| bin | n | MAE | <2 | >=2 |", "|---|---:|---:|---:|---:|"]
        for k in sorted({int(r["truth_mm"]) for r in rows_out
                         if r["cohort"] == cohort}):
            recs = [r for r in rows_out
                    if r["cohort"] == cohort and int(r["truth_mm"]) == k]
            n_lt = sum(1 for r in recs if r["mae_group"] == "mae_lt2")
            lines.append(
                f"| {k}-{k + 1}mm | {len(recs)} | "
                f"{np.mean([r['abs_err_mm'] for r in recs]):.3f} | "
                f"{n_lt} | {len(recs) - n_lt} |"
            )
        lines.append("")
    (out_root / "summary.md").write_text("\n".join(lines))

    (out_root / "README.md").write_text(
        "# visioncheck\n\n"
        "beta-only（`v11-company-diranchor40-betaonly` seed 20261021，"
        "final_ema.pt）对 benchmark2 三个队列的逐帧推理。\n\n"
        "每个队列按真实大小分档：`k-(k+1)mm` 表示 truth ∈ [k, k+1)；"
        "每档再按单帧 MAE 分成 `mae_lt2`（<2mm）与 `mae_ge2`（≥2mm）。"
        "文件按档内误差从大到小排序，前缀越大越差。\n\n"
        "图上绿框/红框 = MAE <2 / ≥2；框上方标注 GT、Pred、Err。"
        "原图坐标由 manifest 的 `crop_box`（company-v1）或"
        "`detect_image_area`（B/R）映射。\n"
    )
    print("wrote", out_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
