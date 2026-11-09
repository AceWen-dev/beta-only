#!/usr/bin/env python3
"""合并 company-v1 清单 + 新增 10–15mm 病例，生成 company-v1-big（D-0064）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
SRC_ROOT = Path("/home/twj/桌面/息肉数据集/Data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DATA_ROOT / "curated/v11_frames/company-v1/manifest.json"))
    ap.add_argument("--extra-rows", default=str(DATA_ROOT / "curated/v11_frames/company-v1-big/rows.json"))
    ap.add_argument("--out", default=str(DATA_ROOT / "curated/v11_frames/company-v1-big/manifest.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_company_features import robust_content_box

    base = json.loads(Path(args.base).read_text())["rows"]
    extra = json.loads(Path(args.extra_rows).read_text())["rows"]
    print(f"base rows={len(base)} extra rows={len(extra)}")

    new_rows = []
    for r in extra:
        image = Path(r["image"])
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"读图失败 {image}")
        crop = [int(v) for v in robust_content_box(img)]
        try:
            rel_img = image.relative_to(SRC_ROOT)
        except ValueError:
            rel_img = image
        feat = Path(r["feature_path"])
        try:
            rel_feat = feat.relative_to(DATA_ROOT)
        except ValueError:
            rel_feat = feat
        new_rows.append({
            "sample_id": f"company-{r['sample_id']}",
            "case_id": f"company-{r['case_id']}",
            "split": r["split"],
            "image": str(Path("raw/company-v1") / rel_img),
            "feature_path": str(rel_feat),
            "truth_mm": r["truth_mm"],
            "bbox_content": [round(float(v), 2) for v in r["bbox_content"]],
            "content_size": r["content_size"],
            "crop_box": crop,
        })
    rows = base + new_rows
    Path(args.out).write_text(json.dumps({"name": "v11_frames",
                                          "version": "company-v1-big",
                                          "rows": rows}, ensure_ascii=False),
                              encoding="utf-8")
    from collections import Counter
    print("splits:", dict(Counter(r["split"] for r in rows)))
    print("train labels:", dict(sorted(Counter(int(r["truth_mm"]) if r["truth_mm"] < 10 else 10
                                               for r in rows if r["split"] == "train").items())))
    print("manifest:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
