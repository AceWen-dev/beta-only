#!/usr/bin/env python3
"""把每个 run 的 tensorboard 目录同步为 output/tb 下的符号链接（D-0035）。

TensorBoard 以 `--logdir output/tb` 运行；新 run 必须在这里有链接才看得见。
链接名格式：``<experiment>__<HHMMSS>__s<seed>``（来自 run id）。
脚本可重复执行：缺失的补上，指向已不存在目录的旧链接删除。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_per_bin import ROOT  # noqa: E402

RUNS = ROOT / "output/runs/beta-regression"
TB = ROOT / "output/tb"


def main() -> int:
    TB.mkdir(parents=True, exist_ok=True)
    existing = {p.name: p for p in TB.iterdir() if p.is_symlink()}
    wanted: dict[str, Path] = {}
    for tb in RUNS.glob("*/2*/tensorboard"):
        run_dir = tb.parent
        try:
            m = json.loads((run_dir / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_id = m.get("run_id", run_dir.name)          # 20261102-153854_xxx_s20261020_yyy
        parts = run_id.split("_")
        stamp = parts[0].split("-")[-1] if "-" in parts[0] else parts[0]
        seed = next((p for p in parts if p.startswith("s") and p[1:].isdigit()), "")
        name = f"{m.get('experiment', run_dir.parent.name)}__{stamp}__{seed}"
        wanted[name] = tb

    added = removed = 0
    for name, target in wanted.items():
        link = TB / name
        if link.is_symlink() and Path(link.resolve()) == target.resolve():
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
        added += 1
    for name, link in existing.items():
        if name not in wanted:
            try:
                link.unlink()
                removed += 1
            except OSError:
                pass
    print(f"TensorBoard 链接：新增/更新 {added} 个，删除失效 {removed} 个，共 {len(wanted)} 个 run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
