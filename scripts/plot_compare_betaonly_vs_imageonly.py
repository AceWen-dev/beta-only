#!/usr/bin/env python3
"""β-only vs 图像直接回归：冻结测试集逐 1mm 区间 MAE 对比（只画 final EMA）。

传统臂 = `v11-company-imageonly`：单视图 RGB → softplus 直接回归 mm，
不使用深度/几何，也不经过 β·D·S 参数化。两个实验使用同一 seed
（20261020）、同一 benchmark suite 口径；B 的 7–8mm 单例按 D-0083
剔除后**直接连线**，不画断点。数据取 eval 模式 JSON。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_per_bin as pp

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DLEF_DATA_ROOT", ""))
REPORTS = ROOT / "output" / "reports"
OUT = ROOT / "output" / "charts" / "compare-betaonly-vs-imageonly-s20261020.png"

IMAGE = REPORTS / "v11-company-imageonly-s20261020-no78_evalmode.json"
BETA = REPORTS / "v11-company-diranchor40-betaonly-s20261020-no78_evalmode.json"

PANELS = (
    ("company", "公司（company-v1）"),
    ("b", "B（PolypSize）"),
    ("r", "R（REAL-colon，观察）"),
)
COLORS = {"image": "#6b7a8f", "beta": "#e07b39"}
LABELS = {
    "image": "图像直接回归 image-only（单视图，15ep）",
    "beta": "β-only diranchor40（双视图 + 深度，40ep）",
}


def b_bin_counts() -> Counter:
    rows = json.loads(
        (DATA / "curated/v11_frames/single-v2/manifest.json").read_text()
    )["rows"]
    return Counter(
        int(float(r["truth_mm"])) for r in rows if r["split"] == "test_b"
    )


def lt10_no7(d: dict, counts: Counter) -> float:
    per = d["per_bin"]
    num = sum(
        per[str(k)] * counts[k]
        for k in range(1, 10)
        if str(k) in per and k != 7
    )
    return num / sum(counts[k] for k in range(1, 10) if k != 7)


def main() -> int:
    if not DATA.is_dir():
        raise SystemExit("set DLEF_DATA_ROOT")
    image = json.loads(IMAGE.read_text())["final_ema.pt"]
    beta = json.loads(BETA.read_text())["final_ema.pt"]
    counts = b_bin_counts()
    lt10 = {
        "image": {"company": image["company"]["lt10"],
                  "b": lt10_no7(image["b"], counts),
                  "r": image["r"]["lt10"]},
        "beta": {"company": beta["company"]["lt10"],
                 "b": lt10_no7(beta["b"], counts),
                 "r": beta["r"]["lt10"]},
    }

    finite = [
        v
        for data in (image, beta)
        for cohort, _ in PANELS
        for v in data[cohort]["per_bin"].values()
        if np.isfinite(v)
    ]
    y_max = max(2.8, max(finite) + 0.3)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(17.6, 5.0))
    for ax, (cohort, title) in zip(axes, PANELS):
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(2, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        for key, data in (("image", image), ("beta", beta)):
            per = data[cohort]["per_bin"]
            ks = [
                k for k in range(2, 10)
                if str(k) in per and not (cohort == "b" and k == 7)
            ]
            ax.plot(
                [k + 0.5 for k in ks], [per[str(k)] for k in ks],
                marker="o", markersize=5, linewidth=2.2,
                color=COLORS[key], label=LABELS[key], zorder=3,
            )
        ax.axhline(pp.THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        note = "\n".join(
            f"{'图像直接' if k == 'image' else 'β-only'}  <10mm: "
            f"{lt10[k][cohort]:.3f}" + ("（剔除 7–8）" if cohort == "b" else "")
            for k in ("image", "beta")
        )
        ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                      ec="#666", alpha=0.92))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([k + 0.5 for k in range(2, 10)])
        ax.set_xticklabels([f"{k}–{k + 1}" for k in range(2, 10)], fontsize=8)
        ax.set_xlim(2, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("MAE (mm)")
    fig.suptitle("β-only vs 图像直接回归：冻结测试集逐区间 MAE（final EMA）",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.012,
             "同一 seed 20261020 · final EMA · eval 模式 · 红虚线 = 2mm 达标线 · "
             "B 的 7–8mm 单例（5 帧）按 D-0083 剔除后直接连线 · "
             "R 按 benchmark2 口径排除 1–2mm（D-0076），仅作观察、不作达标要求 · "
             "image-only = 单视图 RGB 直接回归 mm，无深度/几何",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)

    payload = {
        "seed": 20261020,
        "checkpoint": "final_ema.pt",
        "sources": {"image": str(IMAGE), "beta": str(BETA)},
        "per_bin": {
            cohort: {key: data[cohort]["per_bin"] for key, data in
                     (("image", image), ("beta", beta))}
            for cohort, _ in PANELS
        },
        "lt10": lt10,
        "note": "B lt10 recomputed excluding bin 7 via manifest frame counts",
    }
    out_json = REPORTS / "compare_betaonly_vs_imageonly_20261108.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print("wrote", OUT)
    print("wrote", out_json)
    print(json.dumps(lt10, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
