#!/usr/bin/env python3
"""Run the latest beta-only v11 checkpoint on a video and draw per-frame size.

Pipeline matches the training features:
  content crop -> detector bbox -> DA-V2 relative depth -> prepare_lesion
  -> beta-only V11 -> mm.

Output: <video_dir>/<stem>_betaonly_<seed>.mp4 and a preview contact sheet.
"""
from __future__ import annotations

import os

# Must be set before transformers/polyp_measurement imports.
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import torch

POLYP = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
PROJ = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_ep")
for path in (str(POLYP), str(PROJ / "scripts"), str(PROJ)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(POLYP)  # config's weights/delivery/*.pt are relative to PolypMeasure

from build_company_features import robust_content_box  # noqa: E402
from polyp_measurement.algorithms.measurement.lesion_features import (  # noqa: E402
    align_relative_depth,
    prepare_lesion,
)
from polyp_measurement.domain.models import CoordinateSpace, FramePacket  # noqa: E402
from polyp_measurement.runtime.builder import create_detector, create_pipeline  # noqa: E402
from polyp_measurement.runtime.config import component_config, load_config  # noqa: E402

from core.config import ModelConfig  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402

RUN_DEFAULT = (
    PROJ
    / "output/runs/beta-regression/v11-company-diranchor40-betaonly/"
    "20261106-075001_f83b8b9b_s20261021_48542a"
)
MODEL_CONFIG = POLYP / "polyp_measurement/configs/polyp_size_delivery_dav2.yaml"


def draw_label(frame, box, text, color, sub=""):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.95, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    ty = max(th + 12, y1 - 10)
    cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 12, ty + base), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 6, ty), font, scale, color, thick, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (x1 + 6, ty + 22), font, 0.55, color, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--run", default=str(RUN_DEFAULT))
    ap.add_argument("--out", default="")
    ap.add_argument("--stride", type=int, default=2,
                    help="process every Nth frame; duplicate overlay for skipped frames")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--hold", type=int, default=24,
                    help="max processed frames to hold the last bbox")
    ap.add_argument("--smooth", type=int, default=15,
                    help="median window over processed frames")
    args = ap.parse_args()

    video = Path(args.video)
    run = Path(args.run)
    seed = "s21" if "s20261021" in run.name else ("s20" if "s20261020" in run.name else "s?")
    out = Path(args.out) if args.out else video.with_name(
        f"{video.stem}_betaonly_{seed}.mp4"
    )
    preview_dir = video.with_name(f"{video.stem}_betaonly_{seed}_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints/final_ema.pt", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck.model)
    net = net.cuda().eval()
    print(f"model: {run}  epoch={getattr(ck, 'epoch', '?')}", flush=True)

    cfg = load_config(MODEL_CONFIG)
    pipe = create_pipeline(cfg)
    pipe.depth_estimator.initialize()
    detector = create_detector(component_config(cfg, "detector"))
    detector.initialize()

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    print(f"video: {video}  {width}x{height}  fps={fps:.3f}  frames={total}", flush=True)

    history: deque[float] = deque(maxlen=args.smooth)
    last_box = None
    last_mm = None
    held = 0
    last_vis = None
    n_det = n_pred = 0
    previews = []
    started = time.time()

    for idx in range(total):
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            writer.write(last_vis if last_vis is not None else bgr)
            continue

        crop = robust_content_box(bgr)
        x0, y0, w, h = crop
        rgb = cv2.cvtColor(bgr[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2RGB)
        packet = FramePacket(
            image=rgb, frame_id=idx, source_id=video.name,
            coordinate_space=CoordinateSpace.PROCESSED_IMAGE,
            image_color="rgb",
            metadata={"crop_box_xywh": crop,
                      "original_size_wh": (bgr.shape[1], bgr.shape[0])},
        )
        detections = detector.detect(packet)
        dets = detections.data.detections if detections.data is not None else []
        dets = [d for d in dets if float(d.confidence) >= args.conf]
        best = max(dets, key=lambda d: d.bbox.width * d.bbox.height) if dets else None
        box = None
        low_conf = False
        if best is not None:
            box = (float(best.bbox.x1), float(best.bbox.y1),
                   float(best.bbox.x2), float(best.bbox.y2))
            last_box = box
            held = 0
            n_det += 1
        elif last_box is not None and held < args.hold:
            box = last_box
            held += 1
            low_conf = True

        mm = None
        if box is not None:
            try:
                with torch.inference_mode():
                    pred = pipe.depth_estimator.predict(packet)
                    if pred.data is None:
                        raise ValueError(pred.message or "depth failed")
                    raw = align_relative_depth(packet, pred.data)
                    item = prepare_lesion(rgb, raw, box)
                    batch = {
                        "rgb_views": torch.from_numpy(item["rgb"]).unsqueeze(0).cuda(),
                        "depth": torch.from_numpy(item["depth"][:, :2]).unsqueeze(0).cuda(),
                        "scalars": torch.from_numpy(item["scalars"]).unsqueeze(0).cuda(),
                        "D": torch.tensor([float(np.exp(item["scalars"][6]))],
                                          dtype=torch.float32, device="cuda"),
                        "S": torch.tensor([float(item["span_px"])],
                                          dtype=torch.float32, device="cuda"),
                    }
                    value, _ = net(batch)
                mm = float(value[0])
                history.append(mm)
                last_mm = mm
                n_pred += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] frame {idx}: {exc}", flush=True)
                mm = last_mm
                low_conf = True

        vis = bgr.copy()
        shown = median(history) if history else mm
        if box is not None and shown is not None:
            x1, y1, x2, y2 = (int(round(x0 + box[0])), int(round(y0 + box[1])),
                              int(round(x0 + box[2])), int(round(y0 + box[3])))
            color = (0, 200, 0) if not low_conf else (0, 165, 255)
            draw_label(
                vis, (x1, y1, x2, y2), f"~{shown:.1f} mm", color,
                sub=(f"raw {mm:.1f}" if mm is not None and not low_conf else "hold"),
            )
        cv2.putText(
            vis, f"{video.stem} | beta-only ({seed}) | frame {idx}/{total}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        writer.write(vis)
        last_vis = vis

        if len(previews) < 8 and idx >= (len(previews) + 1) * max(total // 8, 1):
            p = preview_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(p), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
            previews.append(p)

        if (idx // args.stride) % 100 == 0:
            elapsed = time.time() - started
            print(f"frame {idx}/{total}  det={n_det} pred={n_pred}  "
                  f"{elapsed:.0f}s", flush=True)

    cap.release()
    writer.release()
    detector.close()

    # preview contact sheet (resize panels)
    thumbs = []
    for p in previews[:4]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = 640 / img.shape[1]
        thumbs.append(cv2.resize(img, None, fx=scale, fy=scale))
    if thumbs:
        sheet = cv2.vconcat(thumbs)
        sheet_path = video.with_name(f"{video.stem}_betaonly_{seed}_preview.jpg")
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    elapsed = time.time() - started
    print(f"done: {out}  frames={total}  detections={n_det}  preds={n_pred}  "
          f"elapsed={elapsed/60:.1f}min  size_median={median(history) if history else float('nan'):.2f}mm",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
