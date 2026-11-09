#!/usr/bin/env python3
"""删除 TensorBoard 里的 `eval/` 标签（D-0032）。

TB 事件文件只能追加，无法删单个 tag；因此对每个 run 的 tensorboard 目录：
1. 读出全部 scalars / images；
2. 过滤掉 tag 以 `eval/` 开头的项，其余按原 step / wall_time 重写；
3. 重新写入 config/manifest 文本；用新事件文件替换旧文件。

只动镜像（TensorBoard），不碰 metrics.jsonl / evals/*.json / checkpoints。
"""

from __future__ import annotations

import json
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
SKIP_PREFIXES = ("eval/",)


def rewrite(tb_dir: Path) -> tuple[int, int]:
    """返回 (保留的 tag 数, 删除的 tag 数)。"""
    if not tb_dir.is_dir():
        return 0, 0
    ea = EventAccumulator(
        str(tb_dir),
        size_guidance={"scalars": 0, "images": 0, "tensors": 0},
    )
    ea.Reload()
    scalars = ea.Tags().get("scalars", [])
    images = ea.Tags().get("images", [])
    keep_s = [t for t in scalars if not t.startswith(SKIP_PREFIXES)]
    drop_s = [t for t in scalars if t.startswith(SKIP_PREFIXES)]
    if not keep_s and not images and not drop_s:
        return 0, 0

    # 清理上次失败残留的临时目录
    for stale in tb_dir.parent.glob("_tb_rebuild*"):
        if stale.is_dir():
            for f in stale.iterdir():
                f.unlink()
            stale.rmdir()
    tmp = tb_dir.parent / "_tb_rebuild"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    writer = SummaryWriter(log_dir=str(tmp))
    try:
        for tag in keep_s:
            for ev in ea.Scalars(tag):
                writer.add_scalar(tag, ev.value, ev.step, walltime=ev.wall_time)
        for tag in images:
            for ev in ea.Images(tag):
                arr = cv2.imdecode(
                    np.frombuffer(ev.encoded_image_string, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if arr is None:
                    continue
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                writer.add_image(tag, arr, ev.step, walltime=ev.wall_time, dataformats="HWC")
        run_dir = tb_dir.parent
        for name, tag in (("config.resolved.json", "run/config"),
                          ("manifest.json", "run/manifest")):
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
    return len(keep_s), len(drop_s)


def main() -> int:
    total_keep = total_drop = n_dirs = 0
    for tb in sorted(RUNS.glob("*/2*/tensorboard")):
        keep, drop = rewrite(tb)
        if keep or drop:
            n_dirs += 1
            total_keep += keep
            total_drop += drop
    print(f"处理 {n_dirs} 个 TB 目录 | 保留 {total_keep} 个 tag | 删除 {total_drop} 个 eval/ tag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
