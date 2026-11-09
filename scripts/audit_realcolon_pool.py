#!/usr/bin/env python3
"""REAL-colon 训练池可见性审计（D-0019 后续数据扩展的前置）。

对 45 个未进入 R 测试集的病灶：
1. 每病灶在标注帧时间轴上均匀采样 ≤120 帧做评分（避免逐帧全量）；
2. 逐帧指标：框内 Laplacian 方差（清晰度）、框内外对比度比、曝光标记、
   框合法性（在图内、≥12px）；
3. 按 score 排序后做最小间隔去重（≥300 帧 ≈10s），每病灶上限 30 帧；
4. 产出 output/reports/realcolon_pool_audit.csv + 每病灶入选帧统计。

用法见 main；不修改任何数据，只产出审计报告与缩略图。
"""

from __future__ import annotations

import collections
import csv
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

PM = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
OUT = Path(__file__).resolve().parents[1] / "output" / "reports"
PER_LESION_SAMPLE = 120
MIN_GAP = 300
CAP = 30
MIN_BOX = 12


def load_data():
    test_ids = {
        r["lesion_id"]
        for r in csv.DictReader(open(PM / "datasets/benchmark-v1/real-colon-clean-consensus/labels.csv"))
    }
    lesions = {
        r["unique_object_id"]: (r["unique_video_name"], float(r["size [mm]"]))
        for r in csv.DictReader(open(PM / "data/raw/public/REAL-colon/lesion_info.csv"))
    }
    return test_ids, lesions


def frame_metrics(img: np.ndarray, bbox) -> tuple[float, float] | None:
    x, y, w, h = [int(v) for v in bbox]
    H, W = img.shape[:2]
    if w < MIN_BOX or h < MIN_BOX or x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    crop = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(crop, cv2.CV_64F).var())
    ring = np.zeros((H, W), np.uint8)
    pad = max(20, w // 2)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    ring[y0:y1, x0:x1] = 255
    ring[y:y + h, x:x + w] = 0
    outside = img[y0:y1, x0:x1][:, :, 0]
    ring_mask = ring[y0:y1, x0:x1] > 0
    out_std = float(outside[ring_mask].std()) if ring_mask.any() else 0.0
    ratio = float(crop.std()) / (out_std + 1e-6)
    return lap, ratio


def main() -> int:
    test_ids, lesions = load_data()
    remain = {k: v for k, v in lesions.items() if k not in test_ids}
    by_video: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for lid, (video, size) in remain.items():
        by_video[video].append((lid, size))

    z = zipfile.ZipFile(PM / "data/raw/public/REAL-colon/anns.zip")
    rows = []
    for video, lesion_list in sorted(by_video.items()):
        wanted = {lid.split("_")[1]: (lid, size) for lid, size in lesion_list}
        d = json.loads(z.read(f"anns/{video}_frames.json"))
        id2img = {im["id"]: im for im in d["images"]}
        per_lid: dict[str, list] = collections.defaultdict(list)
        for a in d["annotations"]:
            lid_info = wanted.get(str(a.get("class_index")))
            if lid_info is None or a.get("ignore"):
                continue
            im = id2img.get(a["image_id"])
            if im is None:
                continue
            per_lid[lid_info[0]].append((im, a))
        for lid, size in lesion_list:
            video_name = video
            cands = per_lid.get(lid, [])
            if not cands:
                continue
            cands.sort(key=lambda t: t[0].get("frame_number", 0) or 0)
            if len(cands) > PER_LESION_SAMPLE:
                step = len(cands) / PER_LESION_SAMPLE
                cands = [cands[int(i * step)] for i in range(PER_LESION_SAMPLE)]
            scored = []
            for im, a in cands:
                if str(a.get("exposure_degree", "")).lower() != "completely":
                    continue
                p = PM / "data/raw/public/REAL-colon/frames" / f"{video_name}_frames" / im["file_name"]
                img = cv2.imread(str(p))
                if img is None:
                    continue
                m = frame_metrics(img, a["bbox"])
                if m is None:
                    continue
                # anns 的 images 没有 frame_number 字段；帧号在文件名里（001-001_22953.jpg）
                try:
                    fn = int(Path(im["file_name"]).stem.split("_")[-1])
                except ValueError:
                    fn = 0
                scored.append((lid, video_name, size, im["file_name"],
                               fn, a["bbox"], m[0], m[1]))
            if not scored:
                continue
            laps = np.array([s[6] for s in scored])
            rats = np.array([s[7] for s in scored])
            def r_rank(v, arr):
                return float((arr < v).sum()) / max(1, len(arr) - 1)
            for s in scored:
                score = 0.5 * r_rank(s[6], laps) + 0.5 * r_rank(s[7], rats)
                rows.append((*s, score))
            # 最小间隔去重后取 top CAP
            picked = []
            for s in sorted(scored, key=lambda t: -(0.5 * r_rank(t[6], laps) + 0.5 * r_rank(t[7], rats))):
                fn = s[4] or 0
                if any(abs(fn - (p[4] or 0)) < MIN_GAP for p in picked):
                    continue
                picked.append(s)
                if len(picked) >= CAP:
                    break
            s0 = picked[0]
            print(f"{lid:>14} size={size:>4.0f}mm cand={len(cands):>3} kept={len(picked):>2} "
                  f"best lap={s0[6]:.0f} contrast={s0[7]:.2f}")

            for s in picked:
                rows.append(s)
    out = OUT / "realcolon_pool_audit.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lesion_id", "video", "size_mm", "file", "frame_number",
                    "bbox_xywh", "laplacian", "contrast_ratio", "score"])
        w.writerows(rows)
    kept = collections.Counter(r[0] for r in rows)
    small = {k: v for k, v in kept.items() if lesions[k][1] <= 2}
    print(f"\n总入选帧: {len(rows)}  病灶: {len(kept)}  1-2mm 病灶: {len(small)}（{sum(small.values())} 帧）")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
