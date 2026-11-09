#!/usr/bin/env python3
"""B/D 测试集 <3mm 帧的逐帧"真值 vs 预测"图（人工目测用）。

对每帧输出一张图：内容裁剪 + 检测框 + 顶部条带（病例、真值、预测、偏差）。
另出 index.csv（逐帧）与 index_case.csv（病例均值）以及一页式联络图 PDF。

用法：
    python scripts/label_lt3mm_review.py \
        --run output/runs/beta-regression/v11-dev3-g20/<run_id>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig, ProjectConfig  # noqa: E402
from core.data.datasets.v11_frames import (  # noqa: E402
    V11FramesDataset,
    detect_image_area,
)
from core.models.v11 import V11Net  # noqa: E402
from dlef.config import load_config  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def collate(items):
    return {
        **{
            k: torch.stack([x[k] for x in items])
            for k in ("rgb_views", "depth", "scalars", "D", "S", "y")
        },
        "case_id": [x["case_id"] for x in items],
        "sample_id": [x["sample_id"] for x in items],
    }


def predict(net, rows):
    ds = V11FramesDataset(rows, DATA, 2, None)
    preds: dict[str, float] = {}
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
            p, _ = net(b)
            for sid, val in zip(b["sample_id"], p.cpu().tolist()):
                preds[str(sid)] = float(val)
    return preds


def draw_label(im, lines):
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    size = max(20, int(pil.height * 0.030))
    font = ImageFont.truetype(FONT, size)
    widths, heights = [], []
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        widths.append(bb[2] - bb[0])
        heights.append(bb[3] - bb[1])
    pad = int(size * 0.4)
    band_h = sum(heights) + pad * (len(lines) + 1)
    draw.rectangle([0, 0, pil.width, band_h], fill=(0, 0, 0, 175))
    y = pad
    for line, h in zip(lines, heights):
        draw.text((pad, y), line, font=font, fill=(255, 255, 255))
        y += h + pad
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--splits", default="test_b,test_d")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument(
        "--bin", default="", help="真值区间，如 6,7 表示 [6,7) mm；为空时用 --threshold"
    )
    ap.add_argument("--out", default="output/review/lt3mm_bd")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(
        run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False
    )
    net.load_state_dict(ck.model)
    net = net.to("cuda").eval()
    # 与 config 对齐（dev3-g20 均为关闭）
    cfg = load_config(run / "config.resolved.json", ProjectConfig).schema.data
    if cfg.span_calibration or cfg.scale_aug > 0:
        print("!! 该配置启用了 span_calibration/scale_aug，本脚本不适用")
        return 2

    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.bin:
        lo, hi = (float(x) for x in args.bin.split(","))
    else:
        lo, hi = 0.0, args.threshold
    rows = [
        r for r in rows_all
        if r["split"] in splits and lo <= float(r["truth_mm"]) < hi
    ]
    rows.sort(key=lambda r: (r["split"], str(r["case_id"]), str(r["sample_id"])))
    if not rows:
        print("no rows matched")
        return 1
    preds = predict(net, rows)

    out = ROOT / args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    frame_no: dict[tuple[str, str], int] = {}
    for i, r in enumerate(rows, start=1):
        sid = str(r["sample_id"])
        pred = preds[sid]
        truth = float(r["truth_mm"])
        im = cv2.imread(str(DATA / r["image"]), cv2.IMREAD_COLOR)
        if im is None:
            print(f"!! 读图失败 {r['image']}")
            continue
        x0, y0, w, h = detect_image_area(im)
        content = im[y0:y0 + h, x0:x0 + w].copy()
        x1, y1, x2, y2 = (int(round(float(v))) for v in r["bbox_content"])
        cv2.rectangle(content, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = {"test_b": "B", "test_d": "D", "test_r": "R", "test_clean": "C"}[r["split"]]
        case_key = (tag, str(r["case_id"]))
        frame_no[case_key] = frame_no.get(case_key, 0) + 1
        name = f"{i:02d}_{tag}_{r['case_id']}_f{frame_no[case_key]:02d}"
        content = draw_label(
            content,
            [
                f"{tag}  {r['case_id']}   第 {frame_no[case_key]} 帧",
                f"真值 {truth:.2f} mm    预测 {pred:.2f} mm    "
                f"偏差 {pred - truth:+.2f} mm",
            ],
        )
        cv2.imwrite(str(img_dir / f"{name}.png"), content)
        records.append(
            {
                "file": f"images/{name}.png",
                "split": tag,
                "case_id": r["case_id"],
                "sample_id": sid,
                "truth_mm": truth,
                "pred_mm": pred,
                "err_mm": pred - truth,
            }
        )

    with (out / "index.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    by_case: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        by_case.setdefault((rec["split"], rec["case_id"]), []).append(rec)
    with (out / "index_case.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "case_id", "n_frames", "truth_mm", "pred_mean_mm", "err_mm"])
        for (tag, case), recs in sorted(by_case.items()):
            w.writerow([
                tag,
                case,
                len(recs),
                f"{np.median([x['truth_mm'] for x in recs]):.2f}",
                f"{np.mean([x['pred_mm'] for x in recs]):.2f}",
                f"{np.mean([x['pred_mm'] for x in recs]) - np.median([x['truth_mm'] for x in recs]):+.2f}",
            ])

    # 联络图：每页 6 帧
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.backends.backend_pdf import PdfPages

    for f in (
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    with PdfPages(out / "contact_sheet.pdf") as pdf:
        for start in range(0, len(records), 6):
            chunk = records[start:start + 6]
            fig, axes = plt.subplots(2, 3, figsize=(16, 11))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, rec in zip(axes.ravel(), chunk):
                ax.imshow(plt.imread(str(out / rec["file"])))
                ax.set_title(
                    f"{rec['split']} {rec['case_id']}  "
                    f"真值 {rec['truth_mm']:.2f} / 预测 {rec['pred_mm']:.2f}",
                    fontsize=10,
                )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"帧 {len(records)}，病例 {len(by_case)}（{args.splits} 真值 [{lo}, {hi}) mm）")
    print(f"图片: {img_dir}")
    print(f"index: {out / 'index.csv'}  {out / 'index_case.csv'}")
    print(f"联络图: {out / 'contact_sheet.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
