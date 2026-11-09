#!/usr/bin/env python3
"""训练集标签分布 + 各 split 每档帧数对比。

左图：train 每个 1mm 档的帧数，按数据来源（company/newdata/phantom/virtual）
堆叠——展示来源构成与整数标签粒度；
右图：train/val/D/B/R/clean 每个 1mm 档的帧数分组对比——直观呈现
「2–3mm 档测试集有样本、训练集稀缺」的错配。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np

import plot_per_bin as pp  # 复用 CJK 字体设置

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
OUT = ROOT / "output/charts/label_distribution.png"

SRC_COLORS = {
    "company": "#3572b0",
    "newdata": "#3a9d5d",
    "phantom": "#b07a3a",
    "virtual": "#8e6bbf",
}
SPLIT_COLORS = {
    "train": "#3572b0",
    "val": "#8e6bbf",
    "D": "#e07b39",
    "B": "#3a9d5d",
    "R": "#c04a5a",
    "clean": "#b07a3a",
}
SPLITS = (("train", "train"), ("val", "val"), ("test_d", "D"),
          ("test_b", "B"), ("test_r", "R"), ("test_clean", "clean"))


def source_of(case_id: str) -> str:
    for p in ("company", "newdata", "phantom", "virtual"):
        if str(case_id).startswith(p):
            return p
    return str(case_id).split("-")[0] or "other"


def main() -> int:
    rows = json.loads((DATA / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
    bins = list(range(10)) + [10]  # 10 = ≥10mm 合并档
    bin_labels = [f"{k}-{k+1}" for k in range(10)] + ["≥10"]

    def bin_of(t: float) -> int:
        return 10 if t >= 10 else int(t)

    # 左图：train 按来源堆叠
    train = [r for r in rows if r["split"] == "train"]
    stack = {s: np.zeros(len(bins), dtype=int) for s in SRC_COLORS}
    for r in train:
        stack[source_of(r["case_id"])][bin_of(float(r["truth_mm"]))] += 1
    int_frac = sum(1 for r in train if float(r["truth_mm"]) * 100 % 100 == 0) / len(train)

    # 右图：各 split 每档帧数
    per_split = {}
    for split, _ in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        c = Counter(bin_of(float(r["truth_mm"])) for r in rs)
        per_split[split] = np.array([c.get(b, 0) for b in bins])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图
    bottom = np.zeros(len(bins))
    for s in SRC_COLORS:
        ax1.bar(range(len(bins)), stack[s], bottom=bottom, color=SRC_COLORS[s],
                label=s, width=0.8)
        bottom += stack[s]
    for i, v in enumerate(bottom):
        if v > 0:
            ax1.text(i, v + 6, str(int(v)), ha="center", fontsize=8)
    ax1.set_xticks(range(len(bins)))
    ax1.set_xticklabels(bin_labels, fontsize=8)
    ax1.set_title(f"训练集标签分布（按来源堆叠，共 {len(train)} 帧；"
                  f"整数 mm 标签占 {int_frac*100:.0f}%）", fontsize=12, fontweight="bold")
    ax1.set_ylabel("帧数")
    ax1.set_xlabel("标签所在 1mm 档（≥10 合并）")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.5)

    # 右图
    width = 0.13
    xs = np.arange(len(bins))
    for i, (split, label) in enumerate(SPLITS):
        ax2.bar(xs + (i - 2.5) * width, per_split[split], width=width,
                color=SPLIT_COLORS[label], label=label)
    for i, v in enumerate(per_split["train"]):
        if v > 0:
            ax2.text(xs[i] - 2.5 * width, v + 5, str(int(v)), ha="center",
                     fontsize=7, color="#3572b0")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(bin_labels, fontsize=8)
    ax2.set_yscale("symlog", linthresh=50)
    ax2.set_yticks([0, 10, 50, 100, 200, 400])
    ax2.set_title("各 split 每档帧数对比（对数刻度）——2–3/9–10 档"
                  "「测试有、训练缺」", fontsize=12, fontweight="bold")
    ax2.set_ylabel("帧数（symlog）")
    ax2.set_xlabel("1mm 档（≥10 合并）")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.5)

    fig.suptitle("标签分布：训练集构成 与 各 split 档位错配", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
