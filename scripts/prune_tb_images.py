#!/usr/bin/env python3
"""删除 TensorBoard 中指定的 image 标签（默认删单面板 chart/B|D|R|clean）。

保留 chart/per_bin_mae（四面板组合图）。TB 事件只能重写，思路同 prune_tb_eval。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_per_bin import ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

RUNS = ROOT / "output/runs/beta-regression"
DROP = {"chart/B", "chart/D", "chart/R", "chart/clean"}


def rewrite(tb_dir: Path) -> int:
    if not tb_dir.is_dir():
        return 0
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0, "images": 0, "tensors": 0})
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    images = ea.Tags().get("images", [])
    keep_img = [t for t in images if t not in DROP]
    if len(keep_img) == len(images):
        return 0
    tmp = tb_dir.parent / "_tb_img_rebuild"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    writer = SummaryWriter(log_dir=str(tmp))
    try:
        for tag in scalars:
            for ev in ea.Scalars(tag):
                writer.add_scalar(tag, ev.value, ev.step, walltime=ev.wall_time)
        for tag in keep_img:
            for ev in ea.Images(tag):
                arr = cv2.imdecode(np.frombuffer(ev.encoded_image_string, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    continue
                writer.add_image(tag, cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), ev.step,
                                 walltime=ev.wall_time, dataformats="HWC")
        run_dir = tb_dir.parent
        for name, tag in (("config.resolved.json", "run/config"), ("manifest.json", "run/manifest")):
            p = run_dir / name
            if p.is_file():
                writer.add_text(tag, p.read_text(encoding="utf-8", errors="replace"), 0)
        writer.flush()
    finally:
        writer.close()
    for old in tb_dir.glob("events.out.tfevents*"):
        old.unlink()
    for new in tmp.glob("events.out.tfevents*"):
        new.rename(tb_dir / new.name)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return len(images) - len(keep_img)


def main() -> int:
    n = 0
    dirs = 0
    for tb in sorted(RUNS.glob("*/2*/tensorboard")):
        d = rewrite(tb)
        if d:
            dirs += 1
            n += d
    print(f"处理 {dirs} 个 TB 目录，删除 {n} 个单面板 image tag；保留 chart/per_bin_mae")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
