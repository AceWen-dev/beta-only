"""v11 帧数据集：manifest + 原图 + 冻结深度特征。

manifest 行：sample_id / case_id / split / image / feature_path / truth_mm /
bbox_content / content_size / content_box。路径相对 `$DLEF_DATA_ROOT`。
"""

from __future__ import annotations

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


def rgb_views(content_rgb: np.ndarray, bbox, views: int) -> np.ndarray:
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
        v = _letterbox(crop) / 255.0
        v = (v - MEAN) / STD
        out.append(v.transpose(2, 0, 1).astype(np.float32))
    return np.stack(out)


class V11FramesDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path, views: int, transform: Transform):
        self.rows = rows
        self.root = root
        self.views = views
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.rows[index]
        im = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(self.root / r["image"])
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x0, y0, w, h = r["content_box"]
        content = rgb[y0:y0 + h, x0:x0 + w]
        views = rgb_views(content, r["bbox_content"], self.views)
        with np.load(self.root / r["feature_path"], allow_pickle=False) as z:
            depth = np.asarray(z["depth"], np.float32)
            scalars = np.asarray(z["scalars"], np.float32)
        bbox = [float(v) for v in r["bbox_content"]]
        sample = {
            "rgb_views": torch.from_numpy(views),
            "depth": torch.from_numpy(depth),
            "scalars": torch.from_numpy(scalars),
            "D": torch.tensor(float(np.exp(scalars[6])), dtype=torch.float32),
            "S": torch.tensor(max(bbox[2] - bbox[0], bbox[3] - bbox[1]), dtype=torch.float32),
            "y": torch.tensor(float(r["truth_mm"]), dtype=torch.float32),
            "case_id": str(r["case_id"]),
            "sample_id": str(r["sample_id"]),
        }
        return self.transform.fn(sample) if self.transform is not None else sample


def build_v11_frames(config: DataConfig, role: str, transform: Transform) -> DataBundle:
    manifest = curated_dir(config.name, config.version) / "manifest.json"
    rows_all = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
    split_name = f"test_{config.test_set}" if role == "test" else role
    rows = [r for r in rows_all if r["split"] == split_name]
    if config.max_samples:
        rows = rows[: config.max_samples]
    if not rows:
        raise ValueError(f"empty split {split_name!r} in {manifest}")
    dataset = V11FramesDataset(rows, data_root(), config.views, transform)
    membership = hashlib.sha256(
        json.dumps(sorted(r["sample_id"] for r in rows), separators=(",", ":")).encode()
    ).hexdigest()[:16]
    spec = DataSpec(
        name=config.name,
        version=f"{config.name}@{config.version}",
        schema={
            "rgb_views": f"float32[{config.views},3,224,224]",
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
