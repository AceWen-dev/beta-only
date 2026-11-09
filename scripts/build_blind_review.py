#!/usr/bin/env python3
"""盲法标签复核包（D-0047 第 3 步）。

目标：让医生在不看到原标签与模型预测的情况下，独立重复测量一批小息肉
（2–3mm）病例，并混入同来源 3–4mm 与 6–8mm 对照，避免其按"这批都是小的"猜。

产物（output/review/<date>/）：
- images/<code>.png       内容裁剪图，无标注（测量用）
- images/<code>_box.png   同一张图 + 模型使用的检测框（便于核对测的是同一个病灶）
- worksheet.xlsx          只含匿名编号的测量表（发给医生）
- README.md               复核说明（发给医生）
- _key/key.csv            编号 → case/split/source/原标签/模型预测（**不要发给医生**）

用法：
    python scripts/build_blind_review.py --audit output/reports/case_audit_20261103-111009.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.data.datasets.v11_frames import assign_dev_folds, detect_image_area  # noqa: E402

SPLIT_MAP = {"dev_fold0": ("train", "val"), "D": ("test_d",), "B": ("test_b",)}

INSTRUCTIONS = """# 小息肉尺寸盲法复核

## 背景
本批 2–3mm 病例的模型误差明显高于其他档位，需要确认是模型问题还是标签/测量问题。
请您在**不知道原标签和模型预测**的前提下独立估计病灶最大径。

## 材料
- `images/` 中每张图对应一个病例（一个匿名编号）。
- `<编号>.png`：内镜画面裁剪图，无任何标注。
- `<编号>_box.png`：同一画面，绿框标出模型使用的病灶位置，便于确认测的是同一处。

## 做法
1. 按编号顺序逐例估计**最大径（mm）**，填入 `worksheet.xlsx` 的「第一次估计」。
2. 全部完成后，**隔至少 1 天**，再按同样方式独立填「第二次估计」（可改变顺序）。
3. 「是否可测」：病灶过小/模糊/被遮挡导致无法可靠判断时填「否」。
4. 不必回看前面的估计，也不要互相讨论。

## 说明
- 这批病例中混有不同尺寸的病灶，属正常设计，请按实际观感估计。
- 编号是随机的，顺序与尺寸、来源无关。
- 结果仅用于方法学核对，不用于评价任何个人。
"""


def load_cases(audit_csv: Path) -> list[dict]:
    with audit_csv.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick_controls(cases: list[dict], targets: list[dict], per_band: int) -> list[dict]:
    """同 (split, source) 下，从 3–4mm 与 6–8mm 各取至多 per_band 个对照。"""
    out: list[dict] = []
    keys = sorted({(c["split"], c["source"]) for c in targets})
    for split, source in keys:
        pool = [c for c in cases if c["split"] == split and c["source"] == source]
        for lo, hi in ((3.0, 4.0), (6.0, 8.0)):
            band = [c for c in pool if lo <= float(c["truth_med"]) < hi]
            band.sort(key=lambda c: c["case_id"])
            if len(band) <= per_band:
                chosen = band
            else:
                idx = np.unique(
                    np.round(np.linspace(0, len(band) - 1, per_band)).astype(int)
                )
                chosen = [band[i] for i in idx]
            out.extend(chosen)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit", default="output/reports/case_audit_20261103-111009.csv"
    )
    ap.add_argument("--target-lo", type=float, default=2.0)
    ap.add_argument("--target-hi", type=float, default=3.0)
    ap.add_argument("--controls-per-band", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out", default="output/review/small_polyp")
    args = ap.parse_args()

    cases = load_cases(Path(args.audit))
    targets = [
        c for c in cases
        if args.target_lo <= float(c["truth_med"]) < args.target_hi
    ]
    target_keys = {(c["split"], c["case_id"]) for c in targets}
    controls = pick_controls(cases, targets, args.controls_per_band)
    selected = targets + controls
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    rows_all = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    dev = [r for r in rows_all if r["split"] in ("train", "val")]
    fold_of = assign_dev_folds(dev, 3)

    out = ROOT / args.out
    img_dir = out / "images"
    key_dir = out / "_key"
    img_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    def rows_for(case: dict) -> list[dict]:
        splits = SPLIT_MAP[case["split"]]
        rows = [r for r in rows_all if str(r["case_id"]) == case["case_id"]]
        if case["split"] == "dev_fold0":
            rows = [r for r in rows if fold_of.get(str(r["case_id"])) == 0]
        else:
            rows = [r for r in rows if r["split"] in splits]
        return sorted(rows, key=lambda r: str(r["sample_id"]))

    key_rows = []
    for i, case in enumerate(selected, start=1):
        code = f"P-{i:03d}"
        rows = rows_for(case)
        if not rows:
            print(f"!! {case['case_id']}: manifest 中无对应行，跳过")
            continue
        r = rows[0]
        im = cv2.imread(str(DATA / r["image"]), cv2.IMREAD_COLOR)
        if im is None:
            print(f"!! {case['case_id']}: 读图失败 {r['image']}")
            continue
        x0, y0, w, h = detect_image_area(im)
        content = im[y0:y0 + h, x0:x0 + w]
        cv2.imwrite(str(img_dir / f"{code}.png"), content, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        boxed = content.copy()
        x1, y1, x2, y2 = (int(round(float(v))) for v in r["bbox_content"])
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(
            str(img_dir / f"{code}_box.png"), boxed, [cv2.IMWRITE_PNG_COMPRESSION, 6]
        )
        key_rows.append(
            {
                "code": code,
                "case_id": case["case_id"],
                "split": case["split"],
                "source": case["source"],
                "n_frames": case["n"],
                "truth_mm": case["truth_med"],
                "model_pred_mm": case["pred_mean"],
                "signed_err_mm": case["signed_err"],
                "role": (
                    "target"
                    if (case["split"], case["case_id"]) in target_keys
                    else "control"
                ),
                "sample_id": r["sample_id"],
                "image": r["image"],
            }
        )

    with (key_dir / "key.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        w.writeheader()
        w.writerows(key_rows)

    # 工作表：只含匿名编号
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "测量记录"
    headers = ["编号", "第一次估计 (mm)", "第二次估计 (mm)", "是否可测 (是/否)", "备注"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
    for row in key_rows:
        ws.append([row["code"], None, None, "是", ""])
    widths = [10, 18, 18, 16, 32]
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[chr(64 + i)].width = width
    ws.freeze_panes = "A2"
    wb.save(out / "worksheet.xlsx")

    (out / "README.md").write_text(INSTRUCTIONS, encoding="utf-8")
    write_labeled(key_rows, out, img_dir)
    write_labeled_images(key_rows, out)
    n_t = sum(1 for r in key_rows if r["role"] == "target")
    print(f"病例 {len(key_rows)}（目标 {n_t} + 对照 {len(key_rows) - n_t}）")
    print(f"图片（盲）: {img_dir}")
    print(f"图片（带真值/预测）: {out / 'internal' / 'images'}")
    print(f"工作表（盲）: {out / 'worksheet.xlsx'}")
    print(f"内部对照表（含真值/预测，勿外发）: {out / 'internal' / 'review_index.xlsx'}")
    print(f"密钥（勿外发）: {key_dir / 'key.csv'}")
    return 0


def write_labeled_images(key_rows: list[dict], out: Path) -> None:
    """把真值/预测直接烧进图片，写到 internal/images/（与盲法图同名）。

    盲法目录 images/ 保持干净：交给医生的文件里不能出现任何答案。
    """
    from PIL import Image, ImageDraw, ImageFont

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    src_dir = out / "images"
    dst_dir = out / "internal" / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for r in key_rows:
        role = "目标 2–3mm" if r["role"] == "target" else "对照"
        text = (
            f"{r['code']}  {r['split']}/{r['source']}  {role}   "
            f"真值 {float(r['truth_mm']):.2f}mm   预测 "
            f"{float(r['model_pred_mm']):.2f}mm   偏差 "
            f"{float(r['signed_err_mm']):+.2f}mm"
        )
        for suffix in ("", "_box"):
            src = src_dir / f"{r['code']}{suffix}.png"
            if not src.is_file():
                continue
            im = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(im, "RGBA")
            size = max(22, int(im.height * 0.032))
            font = ImageFont.truetype(font_path, size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            pad = int(size * 0.45)
            draw.rectangle(
                [0, 0, min(im.width, tw + 2 * pad), th + 2 * pad],
                fill=(0, 0, 0, 170),
            )
            draw.text((pad, pad - bbox[1] * 0.4), text, font=font, fill=(255, 255, 255))
            im.save(dst_dir / f"{r['code']}{suffix}.png")


def write_labeled(key_rows: list[dict], out: Path, img_dir: Path) -> None:
    """内部核对版：把真值/模型预测写进表格，并生成带标注的联络图 PDF。

    这一份**不能发给复核医生**——盲法是这次复核的全部价值所在；它只用于
    课题内部快速核对病例与数值是否对得上。
    """
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

    internal = out / "internal"
    internal.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "编号": r["code"],
                "角色": "目标(2–3mm)" if r["role"] == "target" else "对照",
                "split": r["split"],
                "来源": r["source"],
                "case_id": r["case_id"],
                "真值_mm": float(r["truth_mm"]),
                "模型预测_mm": float(r["model_pred_mm"]),
                "偏差_mm": float(r["signed_err_mm"]),
                "帧数": int(r["n_frames"]),
                "sample_id": r["sample_id"],
                "图片": f"images/{r['code']}.png",
            }
            for r in key_rows
        ]
    )
    df.to_csv(internal / "review_index.csv", index=False, encoding="utf-8-sig")
    df.to_excel(internal / "review_index.xlsx", index=False)

    # 联络图：每页 6 例（3×2），标注 编号/来源/真值/预测/偏差
    per_page, cols = 6, 3
    pdf_path = internal / "contact_sheet.pdf"
    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(key_rows), per_page):
            chunk = key_rows[start:start + per_page]
            fig, axes = plt.subplots(2, cols, figsize=(15, 10))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, r in zip(axes.ravel(), chunk):
                im = plt.imread(str(img_dir / f"{r['code']}.png"))
                ax.imshow(im)
                ax.set_title(
                    f"{r['code']}  {r['split']}/{r['source']}\n"
                    f"真值 {float(r['truth_mm']):.2f}mm   预测 "
                    f"{float(r['model_pred_mm']):.2f}mm   "
                    f"偏差 {float(r['signed_err_mm']):+.2f}",
                    fontsize=11,
                )
            fig.suptitle(
                f"小息肉复核病例（内部核对版，勿外发） {start + 1}–{start + len(chunk)}",
                fontsize=14,
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
