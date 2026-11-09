#!/usr/bin/env python3
"""把 build_company_features.py 的 rows.json 变成 v11_frames 清单（D-0057）。

- 图像路径改为相对 `$DLEF_DATA_ROOT`（`raw/company-v1/...`，指向用户数据集的软链）
- case_id 加 `company-` 前缀（与既有工具/来源统计一致）
- 输出 `curated/v11_frames/company-v1/manifest.json`
- 顺带算新训练集的 β_ref = median(truth / (q·span))
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
SRC_ROOT = Path("/home/twj/桌面/息肉数据集/Data")


def _robust_crop(path: Path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_company_features import robust_content_box

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return list(robust_content_box(image))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(DATA_ROOT / "curated/v11_frames/company-v1/rows.json"))
    ap.add_argument("--out", default=str(DATA_ROOT / "curated/v11_frames/company-v1/manifest.json"))
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    payload = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    rows = payload["rows"]
    failures = payload.get("failures", [])
    expected = 12894
    if args.require_complete and len(rows) < expected:
        raise SystemExit(f"rows {len(rows)} < {expected}，先补齐特征再建清单")

    out_rows = []
    for r in rows:
        image = Path(r["image"])
        try:
            rel_img = image.relative_to(SRC_ROOT)
        except ValueError:
            rel_img = image
        feat = Path(r["feature_path"])
        try:
            rel_feat = feat.relative_to(DATA_ROOT)
        except ValueError:
            rel_feat = feat
        crop = _robust_crop(SRC_ROOT / rel_img) if not Path(r["image"]).is_absolute() else _robust_crop(Path(r["image"]))
        if crop is None:
            raise SystemExit(f"读图失败：{r['image']}")
        out_rows.append({
            "sample_id": f"company-{r['sample_id']}",
            "case_id": f"company-{r['case_id']}",
            "split": r["split"],
            "image": str(Path("raw/company-v1") / rel_img),
            "feature_path": str(rel_feat),
            "truth_mm": r["truth_mm"],
            "bbox_content": [round(float(v), 2) for v in r["bbox_content"]],
            "content_size": r["content_size"],
            # D-0058：裁剪框随清单固化（饱和度法去 OSD）。数据集优先用它，
            # 否则会退回旧灰度裁剪，导致框与裁剪不一致、ROI 为空。
            "crop_box": [int(v) for v in crop],
        })

    manifest = {"name": "v11_frames", "version": "company-v1", "rows": out_rows}
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    per_split = Counter(r["split"] for r in out_rows)
    per_case = {s: len({r["case_id"] for r in out_rows if r["split"] == s}) for s in per_split}
    print(f"manifest: {args.out}")
    print(f"rows={len(out_rows)} failures={len(failures)} splits={dict(per_split)} cases={per_case}")

    train = [r for r in out_rows if r["split"] == "train"]
    betas = []
    for r in train:
        feat = np.load(DATA_ROOT / r["feature_path"], allow_pickle=False)
        q = float(np.exp(float(feat["scalars"][6])))
        span = max(r["bbox_content"][2] - r["bbox_content"][0],
                   r["bbox_content"][3] - r["bbox_content"][1])
        if q > 0 and span > 0:
            betas.append(float(r["truth_mm"]) / (q * span))
    if betas:
        print(f"train beta_ref median = {np.median(betas):.6f} "
              f"(IQR {np.percentile(betas,25):.4f}-{np.percentile(betas,75):.4f})")
    bins = Counter(int(r["truth_mm"]) if r["truth_mm"] < 10 else 10 for r in train)
    print("train per-bin cases:", dict(sorted(bins.items())))
    if failures:
        print(f"failures sample: {failures[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
