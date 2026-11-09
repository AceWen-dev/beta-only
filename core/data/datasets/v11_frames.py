"""v11 帧数据集：manifest + 原图 + 冻结深度特征。

manifest 行：sample_id / case_id / split / image / feature_path / truth_mm /
bbox_content / content_size / content_box。路径相对 `$DLEF_DATA_ROOT`。
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dlef.contracts import DataSpec

from ...config import DataConfig
from ..paths import curated_dir, data_root
from ..spec import DataBundle
from ..transforms.base import Transform

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def fit_span_calibration(rows: list[dict], n_bins: int = 16) -> dict:
    """按输入 S 分位分箱，拟合 k=truth/S 的尺寸依赖（只用 train 行）。

    机制：小息肉的检测框相对标签偏大（train 上 k=truth/span_px 在 2–4mm 档
    比 4mm 以上低约 35%，见 D-0019），导致 β·D·S 系统性高估小息肉。
    修正：S' = S·k(S)/k_ref，k(S) 由训练集分箱中位数插值——只依赖输入 S，
    部署可用；k_ref 取全体 train 行的 k 中位数，β_ref 的尺度语义不变。
    """

    pairs = []
    for r in rows:
        if r.get("split") != "train":
            continue
        bbox = [float(v) for v in r["bbox_content"]]
        s = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        t = float(r["truth_mm"])
        if s > 0 and t > 0:
            pairs.append((s, t / s))
    if len(pairs) < n_bins * 3:
        return {}
    pairs.sort()
    s_arr = np.asarray([p[0] for p in pairs], dtype=float)
    k_arr = np.asarray([p[1] for p in pairs], dtype=float)
    qs = np.quantile(s_arr, np.linspace(0.0, 1.0, n_bins + 1))
    centers, ks = [], []
    for j in range(n_bins):
        lo, hi = qs[j], qs[j + 1]
        m = (s_arr >= lo) & ((s_arr <= hi) if j == n_bins - 1 else (s_arr < hi))
        if int(m.sum()) < 3:
            continue
        centers.append(float(np.median(s_arr[m])))
        ks.append(float(np.median(k_arr[m])))
    if len(centers) < 2:
        return {}
    return {
        "centers": centers,
        "k": ks,
        "k_ref": float(np.median(k_arr)),
    }


def span_scale_factor(s: float, cal: dict | None) -> float:
    """S 的修正系数 k(S)/k_ref；未启用或拟合失败时恒为 1。"""
    if not cal:
        return 1.0
    k = float(np.interp(s, cal["centers"], cal["k"]))
    return k / cal["k_ref"]


def zoom_content(
    content: np.ndarray, bbox: list[float], alpha: float
) -> tuple[np.ndarray, list[float]]:
    """把内容图与 bbox 同步缩放 α 倍（模拟相机放大率变化，标签不变）。"""
    if alpha == 1.0:
        return content, bbox
    h, w = content.shape[:2]
    interp = cv2.INTER_AREA if alpha < 1 else cv2.INTER_LINEAR
    content = cv2.resize(
        content, (max(1, round(w * alpha)), max(1, round(h * alpha))), interpolation=interp
    )
    return content, [v * alpha for v in bbox]


def detect_image_area(image: np.ndarray, *, min_fraction: float = 0.1) -> tuple[int, int, int, int]:
    """找出真正的内镜画面区域：返回 `(x0, y0, width, height)`。

    与 PolypMeasure `polyp_measurement.datasets.internal_samples.detect_image_area`
    逐字同款（亮度阈值 → 最大连通域 → 外接框）。manifest 冻结的 `content_box`
    与该现算结果在约 38% 的样本上差 1–57px（D-0018），导致同配方训练不可复现；
    因此内容裁剪一律现算，不再使用冻结值。
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = (gray > 12).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0, 0, image.shape[1], image.shape[0]
    # 忽略背景（标签 0），取面积最大的组件
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x, y, width, height = (
        int(stats[largest, cv2.CC_STAT_LEFT]),
        int(stats[largest, cv2.CC_STAT_TOP]),
        int(stats[largest, cv2.CC_STAT_WIDTH]),
        int(stats[largest, cv2.CC_STAT_HEIGHT]),
    )
    if width * height < min_fraction * image.shape[0] * image.shape[1]:
        return 0, 0, image.shape[1], image.shape[0]  # 检测不可信时如实退化为整图
    return x, y, width, height


def _letterbox(array: np.ndarray, size: int = 224, fill: int = 114) -> np.ndarray:
    h, w = array.shape[:2]
    ratio = min(size / h, size / w)
    rh, rw = max(1, round(h * ratio)), max(1, round(w * ratio))
    out = np.full((size, size, 3), fill, np.float32)
    img = cv2.resize(array, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32)
    top, left = (size - rh) // 2, (size - rw) // 2
    out[top:top + rh, left:left + rw] = img
    return out


def _crop_box(bbox, w, h, factor):
    x1, y1, x2, y2 = map(float, bbox)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    return (
        max(0, int(np.floor(cx - bw * factor / 2))),
        max(0, int(np.floor(cy - bh * factor / 2))),
        min(w, int(np.ceil(cx + bw * factor / 2))),
        min(h, int(np.ceil(cy + bh * factor / 2))),
    )


def rgb_views(content_rgb: np.ndarray, bbox, views: int, size: int = 224) -> np.ndarray:
    out = []
    for idx, factor in enumerate((None, 1.25, 2.5)):
        if idx >= views:
            break
        if factor is None:
            crop = content_rgb
            ox, oy = 0, 0
        else:
            x1, y1, x2, y2 = _crop_box(bbox, content_rgb.shape[1], content_rgb.shape[0], factor)
            crop = content_rgb[y1:y2, x1:x2]
            ox, oy = x1, y1
        if idx == 2:
            crop = crop.copy()
            p1 = (int(round(bbox[0] - ox)), int(round(bbox[1] - oy)))
            p2 = (int(round(bbox[2] - ox)), int(round(bbox[3] - oy)))
            cv2.rectangle(crop, p1, p2, (0, 255, 0), 2)
        v = _letterbox(crop, size=size) / 255.0
        v = (v - MEAN) / STD
        out.append(v.transpose(2, 0, 1).astype(np.float32))
    return np.stack(out)


class V11FramesDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        root: Path,
        views: int,
        transform: Transform,
        span_cal: dict | None = None,
        scale_aug: float = 0.0,
        input_size: int = 224,
    ):
        self.rows = rows
        self.root = root
        self.views = views
        self.transform = transform
        self.span_cal = span_cal
        self.scale_aug = float(scale_aug)
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.rows[index]
        im = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(self.root / r["image"])
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # 清单若固化了 crop_box（company-v1 等新版本），必须用它：
        # 否则退化裁剪（灰度法）与清单里的 bbox 坐标系不一致。
        crop_box = r.get("crop_box")
        if crop_box:
            x0, y0, w, h = (int(v) for v in crop_box)
        else:
            x0, y0, w, h = detect_image_area(rgb)
        content = rgb[y0:y0 + h, x0:x0 + w]
        bbox = [float(v) for v in r["bbox_content"]]
        if self.scale_aug > 0:
            import random

            alpha = random.uniform(1.0 / self.scale_aug, self.scale_aug)
            content, bbox = zoom_content(content, bbox, alpha)
        views = rgb_views(content, bbox, self.views, self.input_size)
        with np.load(self.root / r["feature_path"], allow_pickle=False) as z:
            depth = np.asarray(z["depth"], np.float32)
            scalars = np.asarray(z["scalars"], np.float32)
        # S 用（可能被 scale_aug 缩放过的）bbox：表观大小与 S 必须一致变化
        span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        sample = {
            "rgb_views": torch.from_numpy(views),
            "depth": torch.from_numpy(depth),
            "scalars": torch.from_numpy(scalars),
            "D": torch.tensor(float(np.exp(scalars[6])), dtype=torch.float32),
            "S": torch.tensor(span * span_scale_factor(span, self.span_cal), dtype=torch.float32),
            # D-0078：span 旁路用；S_norm=S/W，W=本次实际内容区宽度。
            "S_norm": torch.tensor(span / content.shape[1], dtype=torch.float32),
            "W": torch.tensor(float(content.shape[1]), dtype=torch.float32),
            "y": torch.tensor(float(r["truth_mm"]), dtype=torch.float32),
            "case_id": str(r["case_id"]),
            "sample_id": str(r["sample_id"]),
        }
        return self.transform.fn(sample) if self.transform is not None else sample


def carve_val_cases(rows: list[dict], n_cases: int) -> set[str]:
    """从 train 病例中确定性抽取 n_cases 个 4–6mm 病例作为新 val。

    只挑 4–6mm（训练集最充裕的档），把稀缺档帧全部留给训练；按 case_id 排序
    后等距取样——与 data.seed 无关，跨 run/跨 worker 稳定。case 级切分，无泄漏。
    """

    by_case: dict[str, list[float]] = {}
    for r in rows:
        if r.get("split") != "train":
            continue
        by_case.setdefault(str(r["case_id"]), []).append(float(r["truth_mm"]))
    eligible = sorted(
        c
        for c, ts in by_case.items()
        if 4.0 <= float(np.median(ts)) < 6.0
    )
    if n_cases >= len(eligible):
        return set(eligible)
    idx = np.unique(np.round(np.linspace(0, len(eligible) - 1, n_cases)).astype(int))
    return {eligible[i] for i in idx}


def assign_dev_folds(rows: list[dict], n_folds: int) -> dict[str, int]:
    """dev 池（train+val）病例级折分配：按（来源, 中位尺寸档）分层轮转。

    确定性（与 seed 无关）；分层保证每折都覆盖全部来源与尺寸档——
    选模指标因此能看见 2–3 与 7–10 尾档（D-0022 修复 valmerge 的选模盲区）。
    """

    by_case: dict[str, list[float]] = {}
    for r in rows:
        if r.get("split") not in ("train", "val"):
            continue
        by_case.setdefault(str(r["case_id"]), []).append(float(r["truth_mm"]))
    strata: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for c, ts in by_case.items():
        t = float(np.median(ts))
        bin_t = 10 if t >= 10 else int(t)
        strata[(str(c).split("-")[0], bin_t)].append(c)
    fold_of: dict[str, int] = {}
    # 用**全局**计数器轮转，而不是每个层内从 0 开始：只有一个病例的层
    # （如 9–10mm 各来源各 1 例）否则会全部落到 fold 0（D-0044）。
    g = 0
    for key in sorted(strata):
        for c in sorted(strata[key]):
            fold_of[c] = g % n_folds
            g += 1
    return fold_of


def build_v11_frames(config: DataConfig, role: str, transform: Transform) -> DataBundle:
    manifest = curated_dir(config.name, config.version) / "manifest.json"
    rows_all = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
    split_name = f"test_{config.test_set}" if role == "test" else role
    if config.dev_folds > 0 and role in ("train", "val"):
        # D-0022：dev 池病例级三折，选模覆盖全部尺寸档
        fold_of = assign_dev_folds(rows_all, config.dev_folds)
        if role == "train":
            rows = [
                r for r in rows_all
                if r["split"] in ("train", "val")
                and fold_of[str(r["case_id"])] != config.dev_fold
            ]
        else:
            rows = [
                r for r in rows_all
                if r["split"] in ("train", "val")
                and fold_of[str(r["case_id"])] == config.dev_fold
            ]
    elif config.val_from_train_cases > 0 and role in ("train", "val"):
        # D-0020：旧 val 帧并入训练，从 train 病例确定性切出新 val（case 级）
        val_cases = carve_val_cases(rows_all, config.val_from_train_cases)
        if role == "train":
            rows = [
                r for r in rows_all
                if r["split"] in ("train", "val")
                and str(r["case_id"]) not in val_cases
            ]
        else:
            rows = [
                r for r in rows_all
                if r["split"] == "train" and str(r["case_id"]) in val_cases
            ]
    else:
        rows = [r for r in rows_all if r["split"] == split_name]
    # D-0039：可选把 test split 并入训练（实验用；该测试集随即失效）
    if role == "train" and config.extra_train_splits.strip():
        wanted = {x.strip() for x in config.extra_train_splits.split(",") if x.strip()}
        seen = {r["sample_id"] for r in rows}
        extra = [r for r in rows_all if r["split"] in wanted
                 and r["sample_id"] not in seen]
        rows = rows + extra
    # D-0053：可选剔除大尺寸训练帧（只作用于 train）
    if role == "train" and config.train_max_truth_mm > 0:
        cutoff = float(config.train_max_truth_mm)
        rows = [r for r in rows if float(r["truth_mm"]) < cutoff]
    if config.max_samples:
        rows = rows[: config.max_samples]
    if not rows:
        raise ValueError(f"empty split {split_name!r} in {manifest}")
    dataset = V11FramesDataset(
        rows,
        data_root(),
        config.views,
        transform,
        span_cal=fit_span_calibration(rows_all) if config.span_calibration else None,
        scale_aug=config.scale_aug if role == "train" else 0.0,
        input_size=config.input_size,
    )
    membership = hashlib.sha256(
        json.dumps(sorted(r["sample_id"] for r in rows), separators=(",", ":")).encode()
    ).hexdigest()[:16]
    spec = DataSpec(
        name=config.name,
        version=f"{config.name}@{config.version}",
        schema={
            "rgb_views": f"float32[{config.views},3,{config.input_size},{config.input_size}]",
            "depth": "float32[2,2,128,128]",
            "scalars": "float32[10]",
            "y": "float32[]",
        },
        splits={role: membership},
        transform=transform.name,
        stats={"n": len(rows)},
        provenance={"manifest": str(manifest), "role": role},
    )
    return DataBundle(dataset=dataset, spec=spec)
