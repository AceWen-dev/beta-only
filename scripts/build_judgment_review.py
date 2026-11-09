#!/usr/bin/env python3
"""判据病例盲审包（D-0079 步骤 1）。

目标：把判据真正的 5 个失败病例（B 7–8 的 polyp-size-04、B 2–3 三例、
company 9–10 四例）与同来源 3–4/6–8mm 对照混在一起，生成匿名联系表，
让医生在不知道原标签的情况下独立测量。

产物：
  output/review/judgment_cases_20261105/
    images/<code>_f<k>.jpg / <code>_f<k>_box.jpg
    contact/<code>.jpg
    worksheet.csv          # 只含匿名编号
    _key/key.csv           # 编号→病例/真值（不要给医生）
    README.md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

TARGETS = {
    ("single-v2", "test_b"): [(7.0, 8.0), (2.0, 3.0)],
    ("company-v1", "test_company"): [(9.0, 10.0)],
}
CONTROLS = [(3.0, 4.0), (6.0, 8.0)]

README = """# 判据病例盲审

本批混有不同尺寸的病灶（含 2–3、3–4、6–8、7–8、9–10mm），请按实际观感
独立估计最大径，不要参考原报告或模型输出。

- `images/<编号>_f<k>.jpg`：内镜内容区，无标注；
- `images/<编号>_f<k>_box.jpg`：同一帧，绿框为模型使用的病灶位置；
- `contact/<编号>.jpg`：该病例的多帧联系表；
- 在 `worksheet.csv` 填「第一次估计」「第二次估计」「是否可测」；
- 编号已随机打乱，顺序与尺寸/来源无关；
- 结果仅用于方法学核对，不评价个人。
"""


def load_rows(version: str, split: str) -> list[dict]:
    manifest = json.loads(
        (DATA / f"curated/v11_frames/{version}/manifest.json").read_text()
    )
    return [r for r in manifest["rows"] if r["split"] == split]


def group_cases(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row["case_id"]), []).append(row)
    for value in out.values():
        value.sort(key=lambda r: str(r["sample_id"]))
    return out


def crop_content(row: dict) -> np.ndarray:
    image = cv2.imread(str(DATA / row["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(DATA / row["image"])
    crop = row.get("crop_box")
    if crop:
        x0, y0, w, h = (int(v) for v in crop)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 12).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x0, y0, w, h = (int(stats[i][j]) for j in range(4))
        else:
            h, w = image.shape[:2]
            x0 = y0 = 0
    return image[y0:y0 + h, x0:x0 + w]


def save_case(code: str, rows: list[dict], out: Path, max_frames: int = 3) -> int:
    img_dir = out / "images"
    contact_dir = out / "contact"
    idx = np.unique(
        np.round(np.linspace(0, len(rows) - 1, min(max_frames, len(rows)))).astype(int)
    )
    frames = []
    for k, i in enumerate(idx, start=1):
        row = rows[int(i)]
        content = crop_content(row)
        cv2.imwrite(
            str(img_dir / f"{code}_f{k}.jpg"), content,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        boxed = content.copy()
        x1, y1, x2, y2 = (int(round(float(v))) for v in row["bbox_content"])
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(
            str(img_dir / f"{code}_f{k}_box.jpg"), boxed,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        scale = 360 / max(content.shape[:2])
        thumb = cv2.resize(content, None, fx=scale, fy=scale)
        thumb = cv2.copyMakeBorder(thumb, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frames.append(thumb)
    if frames:
        height = max(t.shape[0] for t in frames)
        padded = [
            cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for t in frames
        ]
        cv2.imwrite(str(contact_dir / f"{code}.jpg"), cv2.hconcat(padded))
    return len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/review/judgment_cases_20261105")
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()

    out = ROOT / args.out
    for name in ("images", "contact", "_key"):
        (out / name).mkdir(parents=True, exist_ok=True)

    selected: list[dict] = []
    for (version, split), bands in TARGETS.items():
        cases = group_cases(load_rows(version, split))
        for lo, hi in bands:
            for case_id, rows in cases.items():
                truth = float(rows[0]["truth_mm"])
                if lo <= truth < hi:
                    selected.append({"case_id": case_id, "cohort": split,
                                     "version": version, "truth": truth,
                                     "rows": rows, "role": "target"})
        for lo, hi in CONTROLS:
            pool = [
                (cid, rows) for cid, rows in cases.items()
                if lo <= float(rows[0]["truth_mm"]) < hi
            ]
            pool.sort(key=lambda item: item[0])
            take = pool[:2] if len(pool) <= 2 else [
                pool[i] for i in np.unique(
                    np.round(np.linspace(0, len(pool) - 1, 2)).astype(int)
                )
            ]
            for cid, rows in take:
                selected.append({"case_id": cid, "cohort": split,
                                 "version": version, "truth": float(rows[0]["truth_mm"]),
                                 "rows": rows, "role": "control"})

    random.Random(args.seed).shuffle(selected)
    key_rows = []
    for i, case in enumerate(selected, start=1):
        code = f"J-{i:03d}"
        n = save_case(code, case["rows"], out)
        key_rows.append({
            "code": code, "case_id": case["case_id"], "cohort": case["cohort"],
            "version": case["version"], "truth_mm": case["truth"],
            "n_frames": len(case["rows"]), "role": case["role"],
        })

    with (out / "_key" / "key.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    with (out / "worksheet.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["编号", "第一次估计(mm)", "第二次估计(mm)", "是否可测(是/否)", "备注"])
        for row in key_rows:
            writer.writerow([row["code"], "", "", "", ""])

    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {out}: {len(key_rows)} cases, targets="
          f"{sum(1 for r in key_rows if r['role'] == 'target')}, controls="
          f"{sum(1 for r in key_rows if r['role'] == 'control')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
