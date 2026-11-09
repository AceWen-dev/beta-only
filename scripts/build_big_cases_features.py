#!/usr/bin/env python3
"""为用户新增的 10–15mm 病例生成 DA 特征（D-0064）。

只处理 manifest.csv 中、未出现在 company-v1 清单里的病例（默认即新增大息肉），
沿用 build_company_features 的裁剪/深度/框口径。
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
from polyp_measurement.algorithms.measurement.lesion_features import (  # noqa: E402
    align_relative_depth,
    prepare_lesion,
)
from polyp_measurement.datasets.internal_samples import load_frame  # noqa: E402
from polyp_measurement.runtime.builder import create_pipeline  # noqa: E402
from polyp_measurement.runtime.config import load_config  # noqa: E402
from polyp_measurement.utils.imaging import as_rgb  # noqa: E402

DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
SPLITS = {"train": "train", "val": "val", "test": "test_company"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/twj/桌面/息肉数据集/Data")
    ap.add_argument("--base-version", default="company-v1")
    ap.add_argument("--out", default=str(DATA_ROOT / "curated/v11_frames/company-v1-big"))
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out)
    feat = out / "features"
    feat.mkdir(parents=True, exist_ok=True)

    have = {r["case_id"].removeprefix("company-") for r in json.loads(
        (DATA_ROOT / f"curated/v11_frames/{args.base_version}/manifest.json").read_text()
    )["rows"]}
    cases = [r for r in csv.DictReader(open(data / "manifest.csv", encoding="utf-8-sig"))
             if r["case_id"] not in have]
    boxes = {(r["case_folder"], r["image"]): r for r in
             csv.DictReader(open(data / "visioncheck_index.csv", encoding="utf-8-sig"))}
    print(f"new cases={len(cases)}", flush=True)

    pipeline = create_pipeline(load_config(
        POLYP / "polyp_measurement/configs/polyp_size_delivery_dav2.yaml"))
    pipeline.depth_estimator.initialize()
    rows, failures, started = [], [], time.monotonic()
    for case in cases:
        case_dir = data / case["split"] / case["case_folder"]
        for img in sorted((case_dir / "images").glob("*.jpg")):
            sid = f"{case['case_id']}::{img.stem}"
            box = boxes.get((case["case_folder"], img.name))
            if box is None:
                failures.append({"sample_id": sid, "reason": "no box"})
                continue
            try:
                raw_bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                crop = robust_content_box(raw_bgr)
                packet, (x0, y0, w, h) = load_frame(img, frame_id=sid, crop=True, crop_box=crop)
                x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
                bbox = [min(max(x1 - x0, 0.0), w - 1), min(max(y1 - y0, 0.0), h - 1),
                        min(max(x2 - x0, 0.0), w - 1), min(max(y2 - y0, 0.0), h - 1)]
                with torch.inference_mode():
                    pred = pipeline.depth_estimator.predict(packet)
                    if pred.data is None:
                        raise ValueError("深度失败")
                    raw = align_relative_depth(packet, pred.data)
                    item = prepare_lesion(as_rgb(packet.image, packet.image_color), raw,
                                          (bbox[0], bbox[1], bbox[2], bbox[3]))
                fpath = feat / f"{sid.replace('::', '_')}.npz"
                np.savez_compressed(fpath, depth=item["depth"][:, :2],
                                    scalars=item["scalars"],
                                    base_norm=np.float32(item["base_norm"]))
                vis = depth_vis(raw, bbox, float(item["q"]))
                cv2.imwrite(str(case_dir / "visioncheck" / f"{img.stem}_depth_da.jpg"),
                            vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                rows.append({
                    "sample_id": sid, "case_id": case["case_id"],
                    "split": SPLITS[case["split"]], "image": str(img),
                    "feature_path": str(fpath), "truth_mm": float(case["label_mm"]),
                    "bbox_content": bbox, "content_size": [int(h), int(w)],
                    "span_px": float(max(bbox[2] - bbox[0], bbox[3] - bbox[1])),
                    "q_raw": float(item["q"]),
                })
            except Exception as exc:  # noqa: BLE001
                failures.append({"sample_id": sid, "reason": str(exc)})
        print(f"  {case['case_folder']}: rows={len(rows)} fail={len(failures)}", flush=True)
    (out / "rows.json").write_text(json.dumps({"rows": rows, "failures": failures},
                                              ensure_ascii=False), encoding="utf-8")
    print(f"done rows={len(rows)} failures={len(failures)} "
          f"{(time.monotonic()-started)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
