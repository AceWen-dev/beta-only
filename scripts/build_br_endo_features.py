#!/usr/bin/env python3
"""为 B/R 测试帧生成 EndoStreamDepth 特征（D-0062 配套）。

用途：Endo 深度训练的模型必须在**同源深度**上评测；B/R 的旧特征是 DA 相对深度，
q 的尺度不同（mm vs 相对），不能混用。这里沿用数据集对 single-v2 的裁剪口径
（`detect_image_area` + 清单里的 bbox），只把深度换成 Endo metric mm。

输出：`curated/v11_frames/b-r-endo/{manifest.json, features/*.npz}`
      rows 保留原 sample_id / split / image / bbox_content，仅替换 feature_path。
"""

from __future__ import annotations

import argparse
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

from build_endostream_depth_images import CHECKPOINT, ENDO_ROOT, infer_mm  # noqa: E402
from evaluate_endostreamdepth_k import _load_model  # noqa: E402
from polyp_measurement.datasets.internal_samples import detect_image_area  # noqa: E402
from polyp_measurement.algorithms.measurement.lesion_features import prepare_lesion  # noqa: E402

DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="test_b,test_r")
    ap.add_argument("--out", default=str(DATA_ROOT / "curated/v11_frames/b-r-endo"))
    args = ap.parse_args()

    out = Path(args.out)
    feat_dir = out / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    splits = {s.strip() for s in args.splits.split(",") if s.strip()}

    rows_all = json.loads(
        (DATA_ROOT / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    rows = [r for r in rows_all if r["split"] in splits]
    print(f"frames={len(rows)} splits={sorted(splits)}", flush=True)

    model = _load_model(ENDO_ROOT, CHECKPOINT)
    model.eval()
    out_rows, failures = [], []
    started = time.monotonic()
    for i, r in enumerate(rows, 1):
        try:
            image = cv2.imread(str(DATA_ROOT / r["image"]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(r["image"])
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x0, y0, w, h = detect_image_area(rgb)
            content = rgb[y0:y0 + h, x0:x0 + w]
            depth = infer_mm(model, content)
            depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
            bbox = [float(v) for v in r["bbox_content"]]
            item = prepare_lesion(content, depth_full,
                                  (bbox[0], bbox[1], bbox[2], bbox[3]))
            feat = feat_dir / f"{r['sample_id'].replace('::', '_')}.npz"
            np.savez_compressed(feat, depth=item["depth"][:, :2],
                                scalars=item["scalars"],
                                base_norm=np.float32(item["base_norm"]))
            entry = dict(r)
            entry["feature_path"] = str(feat.relative_to(DATA_ROOT))
            out_rows.append(entry)
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": r["sample_id"], "reason": str(exc)})
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] ok={len(out_rows)} fail={len(failures)} "
                  f"{(i / max(time.monotonic() - started, 1e-6)):.2f}/s", flush=True)

    manifest = {"name": "v11_frames", "version": "b-r-endo", "rows": out_rows}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"done rows={len(out_rows)} failures={len(failures)}", flush=True)
    for f in failures[:5]:
        print("  fail:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
