#!/usr/bin/env python3
"""用 EndoStreamDepth 生成第二套深度可视化（D-0058）。

与 `build_company_features.py` 的 DA-V2 深度图同格式、同目录（病例的 `visioncheck/`），
文件名后缀区分模型：

    <stem>_depth_da.jpg     DA-V2（metric-SfM 微调，v11 特征用）
    <stem>_depth_endo.jpg   EndoStreamDepth（metric mm）

Endo 输出本身就是毫米，图例标注框内中位深度；色带按每帧 2–98 分位拉伸，
紫=近、红=远。裁剪用新的饱和度法（去 OSD），与特征重建保持同一口径。

用法：
    cd <PolypMeasure> && .venv/bin/python <this script> \
        --data-dir /home/twj/桌面/息肉数据集/Data [--limit 20]
"""

from __future__ import annotations

import argparse
import csv
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

from build_company_features import robust_content_box  # noqa: E402
from evaluate_endostreamdepth_k import _load_model  # noqa: E402
from polyp_measurement.datasets.internal_samples import load_frame  # noqa: E402

ENDO_ROOT = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/endostreamdepth")
CHECKPOINT = ENDO_ROOT / "checkpoints/iter_15001 (1).pth"


def infer_mm(model, crop_rgb: np.ndarray) -> np.ndarray:
    """内容区 RGB → 518×518 毫米深度图（沿用 run_visioncheck_endostreamdepth）。"""
    resized = cv2.resize(crop_rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
    resized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = torch.from_numpy((resized - mean) / std).permute(2, 0, 1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        model.mamba.start_new_sequence()
        dpt = model.get_dpt_features(tensor, input_shape=(1, 3, 518, 518))
        depth = model.multi_level_final_head(dpt)[0][0]
    return depth.detach().float().cpu().numpy()


def colorize(depth_mm: np.ndarray, bbox, long_edge: int = 1024) -> np.ndarray:
    values = np.asarray(depth_mm, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low, high = ((float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                 if finite.size else (0.0, 1.0))
    norm = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    h, w = colored.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale < 1.0:
        colored = cv2.resize(colored, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sx = colored.shape[1] / w
    sy = colored.shape[0] / h
    x1, y1, x2, y2 = (int(round(float(v) * (sx if i % 2 == 0 else sy)))
                      for i, v in enumerate(bbox))
    cv2.rectangle(colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return colored


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/twj/桌面/息肉数据集/Data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    data = Path(args.data_dir)
    cases = {}
    with (data / "manifest.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            cases[r["case_folder"]] = r["split"]
    boxes = {}
    with (data / "visioncheck_index.csv").open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            boxes[(r["case_folder"], r["image"])] = (
                float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            )
    print(f"cases={len(cases)} boxes={len(boxes)}", flush=True)

    model = _load_model(ENDO_ROOT, CHECKPOINT)
    model.eval()
    print("EndoStreamDepth loaded", flush=True)

    started = time.monotonic()
    seen = done = skipped = failed = 0
    for split_dir in ("train", "val", "test"):
        for case_dir in sorted(p for p in (data / split_dir).iterdir() if p.is_dir()):
            if case_dir.name not in cases:
                continue
            img_dir = case_dir / "images"
            vis_dir = case_dir / "visioncheck"
            if not img_dir.is_dir() or not vis_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                if args.limit and seen >= args.limit:
                    break
                seen += 1
                out = vis_dir / f"{img.stem}_depth_endo.jpg"
                if out.is_file():
                    skipped += 1
                    continue
                box = boxes.get((case_dir.name, img.name))
                if box is None:
                    failed += 1
                    continue
                try:
                    raw_bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    if raw_bgr is None:
                        raise FileNotFoundError(img)
                    crop_box = robust_content_box(raw_bgr)
                    packet, (x0, y0, w, h) = load_frame(
                        img, frame_id=0, crop=True, crop_box=crop_box
                    )
                    depth = infer_mm(model, packet.image)  # 518×518 mm
                    depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
                    x1, y1, x2, y2 = box
                    bbox = [
                        min(max(x1 - x0, 0.0), w - 1), min(max(y1 - y0, 0.0), h - 1),
                        min(max(x2 - x0, 0.0), w - 1), min(max(y2 - y0, 0.0), h - 1),
                    ]
                    bx1, by1, bx2, by2 = (int(round(v)) for v in bbox)
                    region = depth_full[by1:by2, bx1:bx2]
                    valid = np.isfinite(region)
                    med = float(np.median(region[valid])) if valid.any() else float("nan")
                    vis = colorize(depth_full, bbox)
                    cv2.putText(vis, f"EndoStreamDepth {med:.1f} mm",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    if failed <= 5:
                        print(f"[失败] {img}: {exc}", flush=True)
                if seen % args.save_every == 0:
                    rate = seen / max(time.monotonic() - started, 1e-6)
                    print(f"[{seen}] done={done} skip={skipped} fail={failed} {rate:.2f}/s",
                          flush=True)
            if args.limit and seen >= args.limit:
                break
        if args.limit and seen >= args.limit:
            break

    print(f"done={done} skipped={skipped} failed={failed} seen={seen} "
          f"elapsed={(time.monotonic()-started)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
