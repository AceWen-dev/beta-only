#!/usr/bin/env python3
"""最优配方（valmerge，3 seed）与 baseline（shuffle）的同图逐区间对比。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np

import plot_per_bin as pp  # 复用字体设置与数据装载

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/charts/valmerge_vs_baseline.png"


def curves_for(experiment: str, ids: dict[str, str], seeds: bool):
    """返回 {cohort: (mean, lo, hi)} 与 {cohort: <10mm MAE}。"""
    runs = pp.completed_runs(experiment)
    if not seeds:
        runs = runs[-1:]
    per, lt10s = [], []
    for run_dir in runs:
        best = run_dir / "checkpoints" / "best.pt"
        if not best.is_file():
            continue
        evals = pp.evals_for_checkpoint(experiment, best, ids)
        if not evals:
            continue
        per.append({
            c: np.array([float(evals[c]["metrics"].get(pp.bin_key(k), np.nan))
                         for k in pp.BINS])
            for c in evals
        })
        lt10s.append({c: float(evals[c]["metrics"]["v11/mae_lt10"]) for c in evals})
    if not per:
        return None, {}
    curves = {}
    for cohort, _ in pp.COHORTS:
        stack = np.stack([r[cohort] for r in per if cohort in r])
        if stack.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            curves[cohort] = (np.nanmean(stack, axis=0), np.nanmin(stack, axis=0),
                              np.nanmax(stack, axis=0))
    lt10_mean = {
        c: float(np.nanmean([l[c] for l in lt10s if c in l]))
        for c in lt10s[0] if any(c in l for l in lt10s)
    }
    return curves, lt10_mean


def main() -> int:
    ids = pp.load_benchmarks()
    best, best_lt = curves_for("v11-baseline-valmerge", ids, seeds=True)
    base, base_lt = curves_for("v11-baseline-shuffle", ids, seeds=False)

    finite = [v for c in list(best.values()) + list(base.values()) for v in c[0]
              if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (cohort, label) in zip(axes, pp.COHORTS):
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        if cohort in base:
            ax.plot(pp.X, base[cohort][0], marker="s", markersize=4,
                    linewidth=1.6, color="#3a9d5d", label="baseline shuffle")
        if cohort in best:
            mean, lo, hi = best[cohort][:3]
            ax.fill_between(pp.X, lo, hi, color="#e07b39", alpha=0.18, linewidth=0)
            ax.plot(pp.X, mean, marker="o", markersize=4, linewidth=2.0,
                    color="#e07b39", label="valmerge（3 seed 均值）")
        ax.axhline(pp.THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        lines = []
        if cohort in best_lt:
            lines.append(f"valmerge  <10mm: {best_lt[cohort]:.3f}")
        if cohort in base_lt:
            lines.append(f"baseline   <10mm: {base_lt[cohort]:.3f}")
        if lines:
            ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#666", alpha=0.9))
        t = dict(pp.COHORTS)[cohort] if pp.FONT else pp.EN_FALLBACK[cohort]
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xticks(pp.X)
        ax.set_xticklabels(pp.BIN_LABELS, fontsize=7.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("MAE (mm)")
    axes[2].set_ylabel("MAE (mm)")
    fig.suptitle("relloss+valmerge（3 seed，当前最优） vs baseline shuffle · "
                 "v2 冻结测试集逐区间 MAE（红虚线 = 2mm）",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "valmerge = 3 seed 均值 ± 最小/最大   |   "
             "<10mm MAE 为各自 eval 记录口径   |   N/A = 该区间无样本",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
