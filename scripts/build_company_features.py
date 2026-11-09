#!/usr/bin/env python3
"""为公司新划分数据集生成 v11 特征（D-0057）。

输入：`~/桌面/息肉数据集/Data`（用户病例级重划分，train/val/test，≤10mm）
输出：`$DLEF_DATA_ROOT/curated/v11_frames/company-v1/`
      ├── features/<sample>.npz   depth views + scalars（与 single-v2 同 schema）
      └── rows.json               逐帧清单（train / val / test_company）

框来源：该数据集自带的 `visioncheck_index.csv`（检测器在原图上的框），
换算到内容裁剪坐标；**不重新检测**，避免检测器在两套预处理下不一致（实测约 20% 帧
在 PolypMeasure 的裁剪图上重检失败）。深度用现役 `polyp_size_delivery_dav2.yaml`
（DA-V2 metric-SfM 全网络微调），与老单一版本特征同源。

用法（在 PolypMeasure 根目录跑，路径用绝对路径）：
    cd <PolypMeasure> && .venv/bin/python <this script> \
        --data-dir /home/twj/桌面/息肉数据集/Data \
        --out /.../Betaregression_data/curated/v11_frames/company-v1
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
sys.path.insert(0, str(POLYP))

from polyp_measurement.algorithms.measurement.lesion_features import (  # noqa: E402
    align_relative_depth,
    prepare_lesion,
)
from polyp_measurement.datasets.internal_samples import (  # noqa: E402
    detect_image_area,
    load_frame,
)
from polyp_measurement.runtime.builder import create_pipeline  # noqa: E402
from polyp_measurement.runtime.config import load_config  # noqa: E402
from polyp_measurement.utils.imaging import as_rgb  # noqa: E402

SPLITS = {"train": "train", "val": "val", "test": "test_company"}


def robust_content_box(image_bgr: np.ndarray, min_frac: float = 0.05):
    """找内镜画面区域：优先用**饱和度**，失败再退回亮度连通域。

    原 `detect_image_area` 用 gray>12 的最大连通域；当 OSD 面板背景不是纯黑
    （灰度 ~20–40）时整幅图会连成一个组件，裁剪退化为整图，**OSD 被带进 RGB 与
    深度输入**（实测新数据约 6% 帧、旧 single-v2 公司帧约 10%）。内镜画面是彩色
    （饱和度高），OSD 是灰/黑/白（饱和度低），用饱和度分割能稳定切掉 OSD。
    """
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 40).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        i = 1 + int(np.argmax(areas))
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area >= min_frac * h * w:
            return x, y, bw, bh
    x, y, bw, bh = detect_image_area(image_bgr)
    return x, y, bw, bh


def depth_vis(depth: np.ndarray, bbox, q: float, long_edge: int = 1024) -> np.ndarray:
    """把对齐到内容区的深度图染成伪彩色，画上病灶框，供人工核对。

    颜色语义：暖色=读数大（相对距离更远）、冷色=读数小（更近）；
    涡轮色图按 2–98 分位拉伸，只表示相对深度，没有毫米含义。
    """
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((long_edge, long_edge, 3), np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    h, w = color.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1.0:
        color = cv2.resize(color, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = (int(round(float(v) * scale)) for v in bbox)
    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(color, f"depth q={q:.3f} (relative)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return color


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/twj/桌面/息肉数据集/Data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(POLYP / "polyp_measurement/configs/polyp_size_delivery_dav2.yaml"))
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
                "case_id": r["case_id"],
                "truth_mm": float(r["label_mm"]),
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
        old = json.loads(rows_path.read_text(encoding="utf-8"))
        done = {r["sample_id"]: r for r in old["rows"]}
        print(f"resume: {len(done)} rows", flush=True)

    config = load_config(args.config)
    pipeline = create_pipeline(config)
    pipeline.depth_estimator.initialize()
    started = time.monotonic()

    rows, failures, seen = [], [], 0
    for split_dir in ("train", "val", "test"):
        split_root = data / split_dir
        for case_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            meta = cases.get(case_dir.name)
            if meta is None:
                continue
            img_dir = case_dir / "images"
            if not img_dir.is_dir():
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
                    failures.append({"sample_id": sid, "reason": "no box in visioncheck_index"})
                    continue
                try:
                    raw_bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    if raw_bgr is None:
                        raise FileNotFoundError(img)
                    crop_box = robust_content_box(raw_bgr)
                    packet, (x0, y0, w, h) = load_frame(
                        img, frame_id=sid, crop=True, crop_box=crop_box
                    )
                    x1, y1, x2, y2 = box
                    bbox = [
                        min(max(x1 - x0, 0.0), w - 1), min(max(y1 - y0, 0.0), h - 1),
                        min(max(x2 - x0, 0.0), w - 1), min(max(y2 - y0, 0.0), h - 1),
                    ]
                    with torch.inference_mode():
                        pred = pipeline.depth_estimator.predict(packet)
                        if pred.data is None:
                            raise ValueError(f"深度失败：{pred.message}")
                        raw = align_relative_depth(packet, pred.data)
                        item = prepare_lesion(
                            as_rgb(packet.image, packet.image_color), raw,
                            (bbox[0], bbox[1], bbox[2], bbox[3]),
                        )
                    feature_path = feat_dir / f"{sid.replace('::', '_')}.npz"
                    np.savez_compressed(
                        feature_path,
                        # v11 schema 只要 2 通道（归一化深度 + 有效掩码），
                        # 第 3 通道是病灶框掩码，不属于模型输入
                        depth=item["depth"][:, :2],
                        scalars=item["scalars"],
                        base_norm=np.float32(item["base_norm"]),
                    )
                    vis = depth_vis(raw, bbox, float(item["q"]))
                    cv2.imwrite(
                        str(case_dir / "visioncheck" / f"{img.stem}_depth_da.jpg"),
                        vis, [cv2.IMWRITE_JPEG_QUALITY, 85],
                    )
                    rows.append({
                        "sample_id": sid,
                        "case_id": meta["case_id"],
                        "split": meta["split"],
                        "image": str(img),
                        "feature_path": str(feature_path),
                        "truth_mm": meta["truth_mm"],
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
                        encoding="utf-8",
                    )
                    rate = seen / max(time.monotonic() - started, 1e-6)
                    print(f"[{seen}] ok={len(rows)} fail={len(failures)} {rate:.2f}/s", flush=True)
            if args.limit and seen >= args.limit:
                break
        if args.limit and seen >= args.limit:
            break

    rows_path.write_text(
        json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False), encoding="utf-8"
    )
    pipeline.close()
    print(f"done rows={len(rows)} failures={len(failures)} "
          f"elapsed={(time.monotonic()-started)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
