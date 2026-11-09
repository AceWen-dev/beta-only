#!/usr/bin/env python3
"""画取景漂移审计图：每个病例 pred vs 表观跨度 p/W，融合/β/直接三支对比。

输入：scripts/framing_drift_audit.py 生成的 frames.csv / per_case.csv。
输出：output/reports/framing_drift/drift_cases.png（漂移最大的 12 例）
      output/reports/framing_drift/drift_slopes.png（91 例斜率分布）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
for f in (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/reports/framing_drift")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    out = ROOT / args.dir

    with (out / "per_case.csv").open(encoding="utf-8-sig") as fh:
        cases = list(csv.DictReader(fh))
    with (out / "frames.csv").open(encoding="utf-8-sig") as fh:
        frames = list(csv.DictReader(fh))
    for c in cases:
        for k in ("slope_pred", "slope_beta", "slope_direct", "truth", "a_ratio"):
            c[k] = float(c[k])
    for r in frames:
        for k in ("a", "pred", "pred_beta", "pred_direct", "truth", "gate", "S_px"):
            r[k] = float(r[k])

    by_case: dict[str, list[dict]] = {}
    for r in frames:
        by_case.setdefault(r["case_id"], []).append(r)
    for rs in by_case.values():
        rs.sort(key=lambda r: r["a"])

    chosen = sorted(cases, key=lambda c: -abs(c["slope_pred"]))[: args.top]

    ncol = 3
    nrow = int(np.ceil(len(chosen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), chosen):
        rs = by_case[c["case_id"]]
        x = [r["a"] for r in rs]
        ax.axis("on")
        ax.plot(x, [r["pred"] for r in rs], "o-", color="#1f77b4", lw=2,
                label="融合输出")
        ax.plot(x, [r["pred_beta"] for r in rs], "s--", color="#d62728", lw=1.4,
                label="β 分支 (βqS)")
        ax.plot(x, [r["pred_direct"] for r in rs], "^-.", color="#2ca02c", lw=1.4,
                label="直接分支")
        ax.axhline(c["truth"], color="k", ls=":", lw=1.2, label="真值")
        ax.set_title(
            f"{c['case_id']}  真值 {c['truth']:.1f}mm  a比 {c['a_ratio']:.2f}×\n"
            f"斜率 融合 {c['slope_pred']:.2f} / β {c['slope_beta']:.2f} / "
            f"直接 {c['slope_direct']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("表观跨度 p/W", fontsize=8)
        ax.set_ylabel("mm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=False)
    fig.suptitle(
        "取景漂移：同一病例预测 vs 表观跨度（斜率 0 = 不随取景变化）",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out / "drift_cases.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title, color in (
        (axes[0], "slope_pred", "融合输出", "#1f77b4"),
        (axes[1], "slope_beta", "β 分支", "#d62728"),
        (axes[2], "slope_direct", "直接分支", "#2ca02c"),
    ):
        vals = np.array([c[key] for c in cases])
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=np.arange(-0.5, 1.65, 0.1), color=color, alpha=0.75)
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(float(np.median(vals)), color="k", lw=1.2, ls=":")
        ax.set_title(f"{title}  中位斜率 {np.median(vals):.2f}  (n={len(vals)})",
                     fontsize=11)
        ax.set_xlabel("d log(pred) / d log(S)")
        ax.set_ylabel("病例数")
    fig.suptitle("取景漂移斜率分布（0 = 理想；1 = 完全跟随表观大小）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "drift_slopes.png", dpi=140)
    plt.close(fig)

    print(f"图: {out / 'drift_cases.png'}")
    print(f"图: {out / 'drift_slopes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
