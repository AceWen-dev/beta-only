#!/usr/bin/env python3
"""Visualize polyp-size-04 (B 7-8mm) stored box vs company detector box."""
from __future__ import annotations

import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

POLYP = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
PROJ = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep")
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
for path in (str(POLYP), str(PROJ)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(POLYP)

from polyp_measurement.datasets.internal_samples import load_frame  # noqa: E402
from polyp_measurement.runtime.builder import create_detector  # noqa: E402
from polyp_measurement.runtime.config import component_config, load_config  # noqa: E402

OUT = PROJ / "output/review/polyp_size_04_detector"
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = POLYP / "polyp_measurement/configs/polyp_size_delivery_dav2.yaml"


def draw_box(img, box, color, label, thick=2):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 4, max(th, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main() -> int:
    manifest = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )
    rows = [
        r for r in manifest["rows"]
        if str(r["case_id"]) == "polyp-size-04" and r["split"] == "test_b"
    ]
    if not rows:
        raise SystemExit("no polyp-size-04 test_b rows")

    cfg = load_config(CONFIG)
    detector = create_detector(component_config(cfg, "detector"))
    detector.initialize()

    summary = []
    tiles = []
    for idx, row in enumerate(rows, start=1):
        path = DATA / row["image"]
        packet, (x0, y0, w, h) = load_frame(path, frame_id=0, crop=True)
        content_bgr = cv2.cvtColor(packet.image, cv2.COLOR_RGB2BGR)
        truth = float(row["truth_mm"])
        stored = [float(v) for v in row["bbox_content"]]
        stored = [min(max(stored[0], 0), w - 1), min(max(stored[1], 0), h - 1),
                  min(max(stored[2], 0), w - 1), min(max(stored[3], 0), h - 1)]
        stored_bw, stored_bh = stored[2] - stored[0], stored[3] - stored[1]
        stored_span = max(stored_bw, stored_bh)

        result = detector.detect(packet)
        dets = result.data.detections if result.data is not None else []
        best = max(dets, key=lambda d: d.bbox.width * d.bbox.height) if dets else None
        det_box = None
        if best is not None:
            det_box = [float(best.bbox.x1), float(best.bbox.y1),
                       float(best.bbox.x2), float(best.bbox.y2)]
            det_bw, det_bh = best.bbox.width, best.bbox.height
            det_span = max(det_bw, det_bh)
            det_conf = float(best.confidence)
            draw_box(content_bgr, det_box, (0, 255, 0),
                     f"detector conf={det_conf:.2f} span={det_span:.1f}px", 3)
        else:
            det_bw = det_bh = det_span = det_conf = None
        draw_box(content_bgr, stored, (255, 255, 0),
                 f"stored span={stored_span:.1f}px", 2)

        # full original frame with the same boxes shifted by the content crop
        full = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if det_box is not None:
            draw_box(full, [x0 + det_box[0], y0 + det_box[1],
                            x0 + det_box[2], y0 + det_box[3]],
                     (0, 255, 0), f"detector {det_conf:.2f}", 3)
        draw_box(full, [x0 + stored[0], y0 + stored[1],
                        x0 + stored[2], y0 + stored[3]],
                 (255, 255, 0), "stored", 2)

        stem = row["sample_id"].split("::")[-1]
        cv2.imwrite(str(OUT / f"{idx:02d}_{stem}_content.jpg"), content_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(OUT / f"{idx:02d}_{stem}_full.jpg"), full,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        scale = 520 / content_bgr.shape[1]
        tile = cv2.resize(content_bgr, None, fx=scale, fy=scale)
        tiles.append(tile)
        summary.append({
            "frame": stem, "truth_mm": truth, "content_w": w, "content_h": h,
            "stored_bbox": [round(v, 1) for v in stored],
            "stored_span_px": round(stored_span, 1),
            "stored_S_over_W": round(stored_span / w, 4),
            "detector_bbox": None if det_box is None else [round(v, 1) for v in det_box],
            "detector_span_px": None if det_span is None else round(det_span, 1),
            "detector_S_over_W": None if det_span is None else round(det_span / w, 4),
            "detector_conf": None if det_conf is None else round(det_conf, 4),
        })
    detector.close()

    if tiles:
        height = max(t.shape[0] for t in tiles)
        padded = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
                  for t in tiles]
        cv2.imwrite(str(OUT / "contact_sheet.jpg"), cv2.hconcat(padded),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        for item in summary:
            writer.writerow(item)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
