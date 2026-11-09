#!/usr/bin/env python3
"""合并对比图：valmerge（3 seed）+ relloss + baseline + ResNet 消融（partial/frozen）。

四个 cohort 面板共用一个图例；valmerge 画均值 ± seed 最小/最大带。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np

import plot_per_bin as pp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/charts/ablation_valmerge_vs_baselines.png"

SERIES = (
    ("v11-baseline-valmerge", "valmerge（3 seed）", "#e07b39", 2.2, True),
    ("v11-baseline-relloss", "relloss", "#3572b0", 1.6, False),
    ("v11-baseline-shuffle", "baseline（shuffle）", "#3a9d5d", 1.6, False),
    ("v11-backbone-partial-best", "ResNet 消融: partial", "#8e6bbf", 1.4, False),
    ("v11-backbone-frozen-best", "ResNet 消融: frozen", "#b07a3a", 1.4, False),
)


def curves_for(experiment: str, ids: dict[str, str], seeds: bool):
    runs = pp.completed_runs(experiment)
    if not seeds:
        runs = runs[-1:]
    per = []
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
    if not per:
        return None
    curves = {}
    for cohort, _ in pp.COHORTS:
        stack = np.stack([r[cohort] for r in per if cohort in r])
        if stack.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            curves[cohort] = (np.nanmean(stack, axis=0), np.nanmin(stack, axis=0),
                              np.nanmax(stack, axis=0))
    return curves


def main() -> int:
    ids = pp.load_benchmarks()
    series = []
    for exp, label, color, lw, band in SERIES:
        c = curves_for(exp, ids, seeds=band)
        if c:
            series.append((label, color, lw, band, c))

    finite = [v for _, _, _, _, c in series for cc in c.values() for v in cc[0]
              if np.isfinite(v)]
    for _, _, _, _, c in series:
        finite += [v for cc in c.values() for v in cc[2] if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (cohort, label) in zip(axes, pp.COHORTS):
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        for lbl, color, lw, band, c in series:
            if cohort not in c:
                continue
            mean, lo, hi = c[cohort][:3]
            if band:
                ax.fill_between(pp.X, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(pp.X, mean, marker="o", markersize=3.5, linewidth=lw,
                    color=color, label=lbl)
        ax.axhline(pp.THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        t = dict(pp.COHORTS)[cohort] if pp.FONT else pp.EN_FALLBACK[cohort]
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xticks(pp.X)
        ax.set_xticklabels(pp.BIN_LABELS, fontsize=7.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("MAE (mm)")
    axes[2].set_ylabel("MAE (mm)")
    fig.suptitle("valmerge vs relloss vs baseline vs ResNet 消融（partial/frozen）· "
                 "v2 冻结测试集逐区间 MAE（红虚线 = 2mm）",
                 fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, 0.065, "valmerge = 3 seed 均值 ± 最小/最大   |   "
             "全部为各自 eval 记录口径（D-0023 修复前）   |   N/A = 该区间无样本",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
