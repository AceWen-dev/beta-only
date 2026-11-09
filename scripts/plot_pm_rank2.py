#!/usr/bin/env python3
"""PM 侧 rank2-official（原始 PolypMeasure 管线）的逐区间 MAE 图。

数据源：PolypMeasure runs/size/v11-rank2-official-20261030/results_test.json
（其自带逐帧评测的 intervals），版式与 DLEF 的 plot_per_bin 完全一致：
cohort 标注来源、只画本实验曲线、<10mm 总 MAE 标注右上角、红虚线 2mm。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_per_bin as pp  # 复用字体设置与版式常量

PM = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure")
SRC = PM / "runs/size/v11-rank2-official-20261030/results_test.json"
OUT = Path(__file__).resolve().parents[1] / "output" / "charts" / "pm-rank2-official.png"
COLOR = "#6a3d9a"
RUN_LABEL = "runs/size/v11-rank2-official-20261030 (seed 20261020)"


def main() -> int:
    payload = json.loads(SRC.read_text())
    test = payload["test"]

    curves, maes = {}, {}
    for cohort in ("D", "B", "R"):
        if cohort not in test:
            continue
        block = test[cohort]
        arr = np.array([
            float(block["intervals"].get(f"{k}-{k+1}", {}).get("mae", np.nan))
            for k in pp.BINS
        ])
        curves[cohort] = arr
        maes[cohort] = float(block["mae_lt10"])

    finite = [v for a in curves.values() for v in a if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    panels = [("d", "D（公司数据集）", "D"), ("b", "B（PolypSize）", "B"),
              ("r", "R（REAL-colon）", "R"), ("clean", "clean（未评测）", None)]
    for ax, (key, label, cohort) in zip(axes, panels):
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        if cohort is not None and cohort in curves:
            arr = curves[cohort]
            ax.plot(pp.X, arr, marker="o", markersize=4, linewidth=1.8,
                    color=COLOR, label=f"PM rank2-official（{maes[cohort]:.2f} 为 <10mm MAE）")
            ax.text(0.97, 0.95, f"<10mm 总 MAE = {maes[cohort]:.3f} mm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=COLOR, alpha=0.9))
        else:
            ax.text(0.5, 0.5, "该 run 未评测此测试集", transform=ax.transAxes,
                    ha="center", va="center", color="#777", fontsize=11)
        for x, k in zip(pp.X, pp.BINS):
            ok = cohort is not None and cohort in curves and np.isfinite(curves[cohort][k])
            if not ok:
                ax.text(x, 0.10, "N/A", ha="center", va="bottom", fontsize=7,
                        color="#777")
        ax.axhline(pp.THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(pp.X)
        ax.set_xticklabels(pp.BIN_LABELS, fontsize=7.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        if cohort is not None and cohort in curves:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("MAE (mm)")
    axes[2].set_ylabel("MAE (mm)")
    fig.suptitle("PM rank2-official（原 PolypMeasure 管线）· v2 冻结测试集逐区间 MAE"
                 "（红虚线 = 2mm 达标线）", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, f"best_ema.pt（PM 选模口径）   |   run: {RUN_LABEL}   |   "
             "N/A = 该区间无样本", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
