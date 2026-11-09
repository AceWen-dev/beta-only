#!/usr/bin/env python3
"""Per-experiment per-bin MAE line charts（v2 冻结测试集）。

每个实验一张 2×2 图（B/D/R/clean），x 轴 1mm 区间，y 轴该区间 MAE；
红色虚线 = 2mm 成功标准；每档**右上角标注该 cohort 的总 MAE**；
只画该实验自己的曲线（不叠参照基线）。

- 普通实验取最新一次 completed run 的 best.pt（D-0018 修复管线）；
- `SEED_STUDY` 中的实验聚合全部 seed：均值线 + min-max 带；
- 输出两份：`output/charts/` 与各实验 `<exp>-eval/per_bin_mae.png`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ---- 中文字体：找 CJK 字体，找不到退化为英文标注 ----
_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _setup_cjk_font() -> str | None:
    registered = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Serif CJK SC", "AR PL UMing CN"):
        if name in registered:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for p in _CANDIDATES:
        if Path(p).is_file():
            font_manager.fontManager.addfont(p)
            fam = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = fam
            plt.rcParams["axes.unicode_minus"] = False
            return fam
    return None


FONT = _setup_cjk_font()

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output" / "runs" / "beta-regression"
CHART_ROOT = ROOT / "output" / "charts"
RECAL_JSON = ROOT / "output/reports/bn_recal_per_bin.json"

COHORTS = (
    ("d", "D（公司数据集）"),
    ("b", "B（PolypSize）"),
    ("r", "R（REAL-colon）"),
    ("clean", "clean（D 子集）"),
    ("company", "公司（新划分 company-v1）"),
)
EN_FALLBACK = {"d": "D (Company)", "b": "B (PolypSize)", "r": "R (REAL-colon)",
               "clean": "clean (D subset)", "company": "Company (company-v1)"}
EXPERIMENTS = (
    "v11-baseline-valmerge",
    "v11-baseline-relloss",
    "v11-baseline-shuffle",
    "v11-baseline-best",
    "v11-ranksim-align",
    "v11-ranksim-best",
    "v11-ranksim-repro",
    "v11-backbone-partial-best",
    "v11-backbone-frozen-best",
    "v11-proj-best",
)
DISPLAY = {
    "v11-baseline-valmerge": "relloss + valmerge（3 seed）",
    "v11-baseline-relloss": "baseline + 相对误差损失",
    "v11-baseline-shuffle": "baseline shuffle（对照起点）",
    "v11-baseline-best": "baseline case_bin",
    "v11-ranksim-align": "ranksim align（3 seed）",
    "v11-ranksim-best": "ranksim 80 轮（旧管线）",
    "v11-ranksim-repro": "ranksim 40 轮（旧管线）",
    "v11-backbone-partial-best": "backbone partial",
    "v11-backbone-frozen-best": "backbone frozen",
    "v11-proj-best": "proj + rank",
}
COLORS = {
    "v11-baseline-valmerge": "#e07b39",
    "v11-baseline-relloss": "#3572b0",
    "v11-baseline-shuffle": "#3a9d5d",
    "v11-baseline-best": "#7f7f7f",
    "v11-ranksim-align": "#c04a5a",
    "v11-ranksim-best": "#8e6bbf",
    "v11-ranksim-repro": "#b07a3a",
    "v11-backbone-partial-best": "#8e6bbf",
    "v11-backbone-frozen-best": "#b07a3a",
    "v11-proj-best": "#c04a5a",
}
# 这些实验聚合全部 completed run（seed 研究）；其余只取最新 run。
SEED_STUDY = {"v11-ranksim-align", "v11-baseline-valmerge"}
OLD_PIPELINE = {"v11-ranksim-best", "v11-ranksim-repro"}
BINS = tuple(range(10))
BIN_LABELS = tuple(f"{k}-{k + 1}" for k in BINS)
X = np.arange(10, dtype=float) + 0.5
THRESHOLD = 2.0


def bin_key(k: int) -> str:
    return f"v11/mae_{k}_{k + 1}"


def load_benchmarks() -> dict[str, str]:
    result = {}
    for cohort, _ in COHORTS:
        payload = json.loads(
            (ROOT / "input/benchmarks" / f"v11-{cohort}-test-v2.json").read_text()
        )
        result[payload["benchmark_id"]] = cohort
    return result


def evals_for_checkpoint(experiment: str, checkpoint: Path, ids: dict[str, str]):
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out: dict[str, dict] = {}
    for path in (RUN_ROOT / f"{experiment}-eval").glob("*/evals/*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("checkpoint_digest") != digest:
            continue
        cohort = ids.get(record.get("benchmark_id"))
        if cohort is not None:
            out[cohort] = record
    return out


def completed_runs(experiment: str) -> list[Path]:
    runs = []
    for manifest in (RUN_ROOT / experiment).glob("*/manifest.json"):
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "completed":
            runs.append(manifest.parent)
    return sorted(runs, key=lambda p: p.name)


def load_recal() -> dict:
    if not RECAL_JSON.is_file():
        return {}
    return json.loads(RECAL_JSON.read_text())


def recal_curves(experiment: str, recal: dict):
    """BN 重估口径（D-0023 修正）：从 bn_recal_per_bin.json 构建曲线。"""
    entries = recal.get(experiment)
    if not entries:
        return None, []
    per_run, labels = [], []
    keymap = {"test_b": "b", "test_d": "d", "test_r": "r", "test_clean": "clean"}
    for e in entries:
        ck = e["口径"]
        entry_curves, entry_lt10, entry_mae = {}, {}, {}
        for c_full, c in keymap.items():
            if c_full not in e["cohorts"]:
                continue
            src = e["cohorts"][c_full]
            if ck == "recorded":
                entry_curves[c] = np.array([
                    float(src["per_bin"].get(str(k), np.nan)) for k in BINS
                ])
                entry_lt10[c] = float(src["lt10"])
                entry_mae[c] = float(src["mae"])
            else:
                entry_curves[c] = np.array([
                    float(src[ck]["per_bin"].get(str(k), np.nan)) for k in BINS
                ])
                entry_lt10[c] = float(src[ck]["lt10"])
                entry_mae[c] = float(src[ck]["mae"])
        per_run.append(entry_curves)
        per_run[-1]["_lt10"] = entry_lt10
        per_run[-1]["_mae"] = entry_mae
        labels.append(f"{e['run'].split('_')[0]} s{e['seed']}")
    if not per_run:
        return None, labels
    curves: dict[str, tuple] = {}
    for cohort, _ in COHORTS:
        arrs = [r[cohort] for r in per_run if cohort in r]
        if not arrs:
            continue
        stack = np.stack(arrs)
        if stack.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            curves[cohort] = (
                np.nanmean(stack, axis=0),
                np.nanmin(stack, axis=0),
                np.nanmax(stack, axis=0),
                float(np.nanmean([r["_mae"][cohort] for r in per_run if cohort in r])),
                float(np.nanmean([r["_lt10"][cohort] for r in per_run if cohort in r])),
            )
    return curves, labels


def experiment_curves(experiment: str, ids: dict[str, str]):
    """{cohort: (mean, lo, hi, mae_mean)} 与 run 标签。"""
    runs = completed_runs(experiment)
    if experiment not in SEED_STUDY:
        runs = runs[-1:]
    per_run, labels = [], []
    for run_dir in runs:
        best = run_dir / "checkpoints" / "best.pt"
        if not best.is_file():
            continue
        evals = evals_for_checkpoint(experiment, best, ids)
        if not evals:
            continue
        per_run.append({
            cohort: np.array([
                float(evals[cohort]["metrics"].get(bin_key(k), np.nan)) for k in BINS
            ])
            for cohort in evals
        })
        per_run[-1]["_mae"] = {
            cohort: (
                float(evals[cohort]["metrics"].get("v11/mae", np.nan)),
                float(evals[cohort]["metrics"].get("v11/mae_lt10", np.nan)),
            )
            for cohort in evals
        }
        seed = json.loads((run_dir / "manifest.json").read_text()).get("seed")
        labels.append(f"{run_dir.name.split('_')[0]} s{seed}")
    if not per_run:
        return None, labels
    curves: dict[str, tuple] = {}
    for cohort, _ in COHORTS:
        arrs = [r[cohort] for r in per_run if cohort in r]
        if not arrs:
            continue
        stack = np.stack(arrs)
        if stack.size == 0:
            continue
        with np.errstate(invalid="ignore"):
            curves[cohort] = (
                np.nanmean(stack, axis=0),
                np.nanmin(stack, axis=0),
                np.nanmax(stack, axis=0),
                float(np.nanmean([r["_mae"][cohort][0] for r in per_run if cohort in r])),
                float(np.nanmean([r["_mae"][cohort][1] for r in per_run if cohort in r])),
            )
    return curves, labels


def plot_experiment(experiment: str, ids: dict[str, str], recal: dict) -> Path | None:
    if experiment in recal:
        curves, labels = recal_curves(experiment, recal)
        mode = "BN 重估口径（D-0023 修正）"
    else:
        curves, labels = experiment_curves(experiment, ids)
        mode = "recorded（旧口径）" if experiment in OLD_PIPELINE else "recorded（无污染）"
    if not curves:
        return None
    multi = len(labels) > 1

    finite = [v for c in curves.values() for v in c[0] if np.isfinite(v)]
    for c in curves.values():
        finite += [v for v in c[2] if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    color = COLORS[experiment]
    for ax, (cohort, label) in zip(axes, COHORTS):
        ax.axhspan(THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        if cohort in curves:
            mean, lo, hi, mae_mean, lt10_mean = curves[cohort]
            if multi:
                ax.fill_between(X, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(X, mean, marker="o", markersize=4, linewidth=1.8, color=color,
                    label=f"本实验（{len(labels)} seed 均值）" if multi else labels[0])
        ax.axhline(THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        # 总 MAE 标注在右上角
        if cohort in curves:
            mae_mean = curves[cohort][3]
            ax.text(0.97, 0.95, f"<10mm 总 MAE = {lt10_mean:.3f} mm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec=color, alpha=0.9))
        title = label if FONT else EN_FALLBACK[cohort]
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(X)
        ax.set_xticklabels(BIN_LABELS, fontsize=7.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        if multi and cohort == list(curves)[0]:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("MAE (mm)")
    axes[2].set_ylabel("MAE (mm)")
    suffix = "均值 ± seed 最小/最大" if multi else "best.pt 单 run"
    title_txt = DISPLAY[experiment]
    fig.suptitle(f"{title_txt} · v2 冻结测试集逐区间 MAE（红虚线 = 2mm 达标线）",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, f"口径: {mode}   |   {suffix}   |   runs: {', '.join(labels)}   |   "
             "N/A = 该区间无样本", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))

    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    path = CHART_ROOT / f"{experiment}.png"
    fig.savefig(path, dpi=180)
    eval_copy = RUN_ROOT / f"{experiment}-eval" / "per_bin_mae.png"
    eval_copy.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(eval_copy, dpi=180)
    plt.close(fig)
    return path



def main() -> int:
    ids = load_benchmarks()
    recal = load_recal()
    for experiment in EXPERIMENTS:
        path = plot_experiment(experiment, ids, recal)
        print(f"{DISPLAY[experiment]}: {path if path else 'no eval yet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
