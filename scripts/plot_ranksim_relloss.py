#!/usr/bin/env python3
"""ranksim+relloss+valmerge 的 eval 模式评测与出图。

产出：
1. output/reports/ranksim_relloss_evalmode.json（两 checkpoint × 四 cohort 逐档）
2. output/charts/v11-ranksim-relloss.png（该实验自身曲线：best / best_ema）
3. output/charts/ranksim_relloss_vs_valmerge.png（与上一最优 valmerge 对照）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import plot_per_bin as pp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402

RUN = ROOT / "output/runs/beta-regression/v11-ranksim-relloss/20261102-093428_5bce26a3_s20261020_f6da1d"
VALMERGE = ROOT / "output/runs/beta-regression/v11-baseline-valmerge/20261031-225639_e36b2588_s20261020_10aa8f"
OUT_STANDALONE = ROOT / "output/charts/v11-ranksim-relloss.png"
OUT_COMPARE = ROOT / "output/charts/ranksim_relloss_vs_valmerge.png"
OUT_JSON = ROOT / "output/reports/ranksim_relloss_evalmode.json"
SPLITS = {"d": "test_d", "b": "test_b", "r": "test_r", "clean": "test_clean"}


def measure(run: Path, ckpt: str):
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / f"checkpoints/{ckpt}", map_location="cpu", weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to("cuda").eval()
    rows_all = json.loads((DATA / "curated/v11_frames/single-v2/manifest.json").read_text())["rows"]
    out = {}
    for cohort, split in SPLITS.items():
        rows = [r for r in rows_all if r["split"] == split]
        ds = V11FramesDataset(rows, DATA, 2, None)

        def collate(items):
            return {
                **{k: torch.stack([x[k] for x in items]) for k in
                   ("rgb_views", "depth", "scalars", "D", "S", "y")},
                "case_id": [x["case_id"] for x in items],
            }

        ps, ts = [], []
        with torch.inference_mode():
            for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
                b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in b.items()}
                p, _ = net(b)
                ps.extend(p.cpu().tolist())
                ts.extend(b["y"].tolist())
        ps, ts = np.array(ps), np.array(ts)
        bins = {
            k: float(np.mean(np.abs(ps[(ts >= k) & (ts < k + 1)] - ts[(ts >= k) & (ts < k + 1)])))
            for k in pp.BINS if ((ts >= k) & (ts < k + 1)).any()
        }
        out[cohort] = {
            "epoch": int(ck.epoch),
            "per_bin": bins,
            "lt10": float(np.mean(np.abs(ps[ts < 10] - ts[ts < 10]))),
            "mae": float(np.mean(np.abs(ps - ts))),
        }
    del net
    torch.cuda.empty_cache()
    return out


def draw(curves: dict, lt10: dict, title: str, out: Path, compare: dict | None = None):
    """curves: {label: (color, {cohort: array})}；compare: {cohort: (array, lt10)}"""
    finite = [v for _, c in curves.values() for a in c.values() for v in a if np.isfinite(v)]
    if compare:
        finite += [v for a, _ in compare.values() for v in a if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (cohort, label) in zip(axes, pp.COHORTS):
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        box = []
        if compare and cohort in compare:
            arr, lt = compare[cohort]
            ax.plot(pp.X, arr, linestyle="--", marker="s", markersize=3.5,
                    linewidth=1.5, color="#7f7f7f", label="valmerge（上一最优）")
            box.append(f"valmerge  <10mm: {lt:.3f}")
        for lbl, (color, c) in curves.items():
            if cohort not in c:
                continue
            ax.plot(pp.X, c[cohort], marker="o", markersize=4, linewidth=2.0,
                    color=color, label=lbl)
            box.append(f"{lbl}  <10mm: {lt10[lbl][cohort]:.3f}")
        ax.axhline(pp.THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        if box:
            ax.text(0.97, 0.95, "\n".join(box), transform=ax.transAxes, ha="right",
                    va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#666", alpha=0.9))
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
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "eval 模式评测（D-0023 修复后）· 相对误差主损失 + RankSim + valmerge 数据 · "
             "红虚线 = 2mm · N/A = 该区间无样本", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    best = measure(RUN, "best.pt")
    ema = measure(RUN, "best_ema.pt")
    OUT_JSON.write_text(json.dumps({"best": best, "best_ema": ema}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    for tag, d in (("best", best), ("best_ema", ema)):
        print(tag, {c: round(v["lt10"], 3) for c, v in d.items()})

    def as_arrays(d):
        return {c: np.array([d[c]["per_bin"].get(k, np.nan) for k in pp.BINS])
                for c in d}

    def lt_map(d):
        return {c: d[c]["lt10"] for c in d}

    # 1) 该实验自身
    b_lbl = f"best.pt（ep{best['d']['epoch']}）"
    e_lbl = f"best_ema.pt（ep{ema['d']['epoch']}）"
    draw(
        curves={
            b_lbl: ("#3572b0", as_arrays(best)),
            e_lbl: ("#e07b39", as_arrays(ema)),
        },
        lt10={b_lbl: lt_map(best), e_lbl: lt_map(ema)},
        title="ranksim + 相对误差损失 + valmerge · v2 冻结测试集逐区间 MAE",
        out=OUT_STANDALONE,
    )

    # 2) 与 valmerge 对照
    ids = pp.load_benchmarks()
    vm_curves, vm_lt = pp_curves = None, None
    from plot_best_vs_baseline import curves_for  # 复用带 3-seed 均值的逻辑

    vm_curves, vm_lt = curves_for("v11-baseline-valmerge", ids, seeds=True)
    r_lbl = f"rank+relloss（best_ema, ep{ema['d']['epoch']}）"
    draw(
        curves={r_lbl: ("#e07b39", as_arrays(ema))},
        lt10={r_lbl: lt_map(ema)},
        title="rank + 相对误差损失 vs 上一最优 valmerge · v2 冻结测试集逐区间 MAE",
        out=OUT_COMPARE,
        compare={c: (vm_curves[c][0], vm_lt[c]) for c in vm_curves if c in vm_lt},
    )
    print("saved:", OUT_STANDALONE)
    print("saved:", OUT_COMPARE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
