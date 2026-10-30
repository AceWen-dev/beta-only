#!/usr/bin/env python3
"""One per-bin MAE line chart per experiment, saved beside its eval folder."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "output" / "runs" / "beta-regression"
BENCH_ROOT = ROOT / "input" / "benchmarks"

COHORTS = (("d", "D"), ("b", "B"), ("r", "R"), ("clean", "clean"))
EXPERIMENTS = ("v11-baseline", "v11-ranksim", "v11-proj")
DISPLAY = {
    "v11-baseline": "baseline",
    "v11-ranksim": "ranksim",
    "v11-proj": "proj",
}
COLORS = {
    "v11-baseline": "#3572b0",
    "v11-ranksim": "#e07b39",
    "v11-proj": "#3a9d5d",
}
BINS = tuple(range(10))
BIN_LABELS = tuple(f"{k}–{k + 1}" for k in BINS)
X = np.arange(10, dtype=float) + 0.5  # bin centres, not bin edges
THRESHOLD = 2.0


def bin_key(k: int) -> str:
    return f"v11/mae_{k}_{k + 1}"


def load_benchmarks() -> dict[str, str]:
    result = {}
    for cohort, _ in COHORTS:
        payload = json.loads((BENCH_ROOT / f"v11-{cohort}-test-v2.json").read_text())
        result[payload["benchmark_id"]] = cohort
    return result


def latest_evals(experiment: str, benchmark_ids: dict[str, str]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    eval_root = RUN_ROOT / f"{experiment}-eval"
    for path in sorted(eval_root.glob("*/evals/*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("data_version") != "v11_frames@single-v2":
            continue
        cohort = benchmark_ids.get(record.get("benchmark_id"))
        if cohort is None:
            continue
        old = best.get(cohort)
        if old is None or record.get("created_at", "") > old.get("created_at", ""):
            best[cohort] = record
    return best


def latest_train_run(experiment: str) -> str:
    candidates = []
    for path in (RUN_ROOT / experiment).glob("*/manifest.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        versions = payload.get("data_versions", {})
        if any(str(v).endswith("@single-v2") for v in versions.values()):
            candidates.append((payload.get("started_at", ""), payload.get("run_id", "")))
    return max(candidates)[1] if candidates else "-"


def values_for(record: dict) -> np.ndarray:
    metrics = record["metrics"]
    return np.array(
        [float(metrics.get(bin_key(k), np.nan)) for k in BINS], dtype=float
    )


def plot_experiment(experiment: str, evals: dict[str, dict]) -> Path | None:
    if not evals:
        return None

    figures = []
    all_values = []
    for cohort, _ in COHORTS:
        record = evals.get(cohort)
        if record is not None:
            all_values.append(values_for(record))
    finite = np.concatenate(all_values) if all_values else np.array([])
    finite = finite[np.isfinite(finite)]
    y_max = max(5.0, float(finite.max()) + 0.5) if finite.size else 5.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (cohort, label) in zip(axes, COHORTS):
        record = evals.get(cohort)
        if record is None:
            ax.text(0.5, 0.5, "not evaluated yet", transform=ax.transAxes,
                    ha="center", va="center", color="#777")
            ax.set_xticks(X)
            ax.set_xticklabels(BIN_LABELS, fontsize=7.5)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, y_max)
            ax.set_title(f"{label} test", fontsize=12, fontweight="bold")
            continue

        values = values_for(record)
        ax.axhspan(THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.plot(
            X,
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=COLORS[experiment],
        )
        for x, value in zip(X, values):
            if not np.isfinite(value):
                ax.text(x, 0.10, "N/A", ha="center", va="bottom",
                        fontsize=7, color="#777")
        ax.axhline(THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2)
        mean_mae = float(record["metrics"].get("v11/mae", np.nan))
        ax.text(
            0.98,
            0.92,
            f"average MAE={mean_mae:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d7de", alpha=0.9),
        )
        ax.set_title(f"{label} test", fontsize=12, fontweight="bold")
        ax.set_xticks(X)
        ax.set_xticklabels(BIN_LABELS, fontsize=7.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("MAE (mm)")
    axes[2].set_ylabel("MAE (mm)")
    fig.suptitle(
        f"{DISPLAY[experiment]} · per-bin MAE on v2 test sets "
        f"(dashed red = 2 mm; points at bin centres; grey lines = bin boundaries)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"run: {latest_train_run(experiment)}   ·   "
        "N/A = no samples in that bin",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out_dir = RUN_ROOT / f"{experiment}-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "per_bin_mae.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return path


def main() -> int:
    benchmark_ids = load_benchmarks()
    for experiment in EXPERIMENTS:
        evals = latest_evals(experiment, benchmark_ids)
        path = plot_experiment(experiment, evals)
        if path is None:
            print(f"{DISPLAY[experiment]}: no v2 eval yet")
        else:
            print(f"{DISPLAY[experiment]}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
