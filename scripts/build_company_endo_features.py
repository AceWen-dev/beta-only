#!/usr/bin/env python3
"""用 EndoStreamDepth 生成 v11 特征 + 深度可视化（D-0062）。

一次遍历同时做两件事：
1. `visioncheck/<stem>_depth_endo.jpg` —— metric depth（mm）伪彩色 + 框 + 框中位深度；
2. `curated/v11_frames/company-v1-endo/features/<sid>.npz` —— 与 DA 特征同 schema
   （depth 2 通道视图 + 10 个几何标量，q 为毫米中位深度）。

数据划分、框、标签与 company-v1 完全一致，仅深度来源不同。

用法（PolypMeasure 根目录）：
    .venv/bin/python <this script> --out /.../curated/v11_frames/company-v1-endo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

POLYP = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
PROJECT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep")
for p in (str(POLYP), str(POLYP / "scripts"), str(PROJECT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_company_features import depth_vis, robust_content_box  # noqa: E402
from build_endostream_depth_images import ENDO_ROOT, CHECKPOINT, infer_mm  # noqa: E402
from evaluate_endostreamdepth_k import _load_model  # noqa: E402
from polyp_measurement.algorithms.measurement.lesion_features import (  # noqa: E402
    align_relative_depth,
    prepare_lesion,
)
from polyp_measurement.datasets.internal_samples import load_frame  # noqa: E402
from polyp_measurement.utils.imaging import as_rgb  # noqa: E402

SPLITS = {"train": "train", "val": "val", "test": "test_company"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/twj/桌面/息肉数据集/Data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.json"

    cases = {}
    with (data / "manifest.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            cases[r["case_folder"]] = {
                "case_id": r["case_id"], "truth_mm": float(r["label_mm"]),
                "split": SPLITS[r["split"]],
            }
    boxes = {}
    with (data / "visioncheck_index.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            boxes[(r["case_folder"], r["image"])] = (
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            )
    print(f"cases={len(cases)} boxes={len(boxes)}", flush=True)

    done = {}
    if rows_path.is_file():
        done = {r["sample_id"]: r for r in json.loads(rows_path.read_text())["rows"]}
        print(f"resume: {len(done)} rows", flush=True)

    model = _load_model(ENDO_ROOT, CHECKPOINT)
    model.eval()
    started = time.monotonic()
    rows, failures, seen = [], [], 0
    for split_dir in ("train", "val", "test"):
        for case_dir in sorted(p for p in (data / split_dir).iterdir() if p.is_dir()):
            meta = cases.get(case_dir.name)
            if meta is None:
                continue
            img_dir, vis_dir = case_dir / "images", case_dir / "visioncheck"
            if not img_dir.is_dir() or not vis_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                if args.limit and seen >= args.limit:
                    break
                seen += 1
                sid = f"{meta['case_id']}::{img.stem}"
                if sid in done:
                    rows.append(done[sid])
                    continue
                box = boxes.get((case_dir.name, img.name))
                if box is None:
                    failures.append({"sample_id": sid, "reason": "no box"})
                    continue
                try:
                    raw_bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    if raw_bgr is None:
                        raise FileNotFoundError(img)
                    crop = robust_content_box(raw_bgr)
                    packet, (x0, y0, w, h) = load_frame(
                        img, frame_id=sid, crop=True, crop_box=crop
                    )
                    depth = infer_mm(model, packet.image)
                    depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
                    x1, y1, x2, y2 = box
                    bbox = [
                        min(max(x1 - x0, 0.0), w - 1), min(max(y1 - y0, 0.0), h - 1),
                        min(max(x2 - x0, 0.0), w - 1), min(max(y2 - y0, 0.0), h - 1),
                    ]
                    item = prepare_lesion(
                        as_rgb(packet.image, packet.image_color), depth_full,
                        (bbox[0], bbox[1], bbox[2], bbox[3]),
                    )
                    feature_path = feat_dir / f"{sid.replace('::', '_')}.npz"
                    np.savez_compressed(
                        feature_path,
                        depth=item["depth"][:, :2],
                        scalars=item["scalars"],
                        base_norm=np.float32(item["base_norm"]),
                    )
                    vis = depth_vis(depth_full, bbox, float(item["q"]))
                    cv2.putText(vis, "EndoStreamDepth", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imwrite(str(vis_dir / f"{img.stem}_depth_endo.jpg"), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])
                    rows.append({
                        "sample_id": sid, "case_id": meta["case_id"],
                        "split": meta["split"], "image": str(img),
                        "feature_path": str(feature_path), "truth_mm": meta["truth_mm"],
                        "bbox_content": bbox,
                        "content_size": [int(h), int(w)],
                        "span_px": float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])),
                        "q_raw": float(item["q"]),
                    })
                except Exception as exc:  # noqa: BLE001
                    failures.append({"sample_id": sid, "reason": str(exc)})
                if seen % args.save_every == 0:
                    rows_path.write_text(
                        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False),
                        encoding="utf-8")
                    rate = seen / max(time.monotonic() - started, 1e-6)
                    print(f"[{seen}] ok={len(rows)} fail={len(failures)} {rate:.2f}/s", flush=True)
            if args.limit and seen >= args.limit:
                break
        if args.limit and seen >= args.limit:
            break
    rows_path.write_text(
        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False), encoding="utf-8")
    print(f"done rows={len(rows)} failures={len(failures)} "
          f"elapsed={(time.monotonic()-started)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
