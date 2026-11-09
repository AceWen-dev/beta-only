#!/usr/bin/env python3
"""EndoStreamDepth 米制刻度的跨队列一致性审计（只读，D-0086 诊断）。

物理关系（针孔近似）：truth_mm ≈ q_mm · span_px / f_px
→ f_px = q_mm · span_px / truth_mm，同一内镜/标定下应跨病灶、跨距离近似常数。
这里按队列比较 q_mm 与隐含焦距 f_px，回答「B 是否整体偏一个常数因子」。

用法：
    python scripts/endo_scale_audit.py --out output/reports/endo_scale_20261106
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")
sys.path.insert(0, str(ROOT / "scripts"))
import plot_per_bin as pp  # noqa: E402  (中文字体设置)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (显示名, 数据版本, split, 深度源)
COHORTS = (
    ("company_train", "company-v1-endo", "train", "endo"),
    ("company_val", "company-v1-endo", "val", "endo"),
    ("company_test", "company-v1-endo", "test_company", "endo"),
    ("B_test", "b-r-endo", "test_b", "endo"),
    ("R_test", "b-r-endo", "test_r", "endo"),
    ("company_test_DA", "company-v1", "test_company", "da"),
    ("B_test_DA", "single-v2", "test_b", "da"),
    ("R_test_DA", "single-v2", "test_r", "da"),
)
MAIN = ("company_train", "company_val", "company_test", "B_test", "R_test")
DA = ("company_test_DA", "B_test_DA", "R_test_DA")
COLORS = {
    "company_train": "#9e9e9e",
    "company_val": "#8e6bbf",
    "company_test": "#3572b0",
    "B_test": "#e07b39",
    "R_test": "#3a9d5d",
    "company_test_DA": "#3572b0",
    "B_test_DA": "#e07b39",
    "R_test_DA": "#3a9d5d",
}


def load_rows(name: str, version: str, split: str, depth_kind: str,
              max_frames: int = 0) -> list[dict]:
    manifest = json.loads(
        (DATA / f"curated/v11_frames/{version}/manifest.json").read_text()
    )
    rows = [r for r in manifest["rows"] if r["split"] == split]
    if max_frames:
        rows = rows[:max_frames]
    out = []
    for i, r in enumerate(rows):
        z = np.load(DATA / r["feature_path"], allow_pickle=False)
        s = z["scalars"]
        bbox = [float(v) for v in r["bbox_content"]]
        span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        truth = float(r["truth_mm"])
        q = float(np.exp(float(s[6])))
        out.append({
            "cohort": name,
            "version": version,
            "split": split,
            "depth_kind": depth_kind,
            "case_id": str(r["case_id"]),
            "sample_id": str(r["sample_id"]),
            "truth_mm": truth,
            "q": q,
            "span_px": span,
            "f_px": (q * span / truth) if truth > 0 else float("nan"),
        })
        if (i + 1) % 3000 == 0:
            print(f"  {name}: {i + 1}/{len(rows)}", flush=True)
    return out


def case_medians(rows: list[dict], key: str) -> np.ndarray:
    by_case: dict[str, list[float]] = {}
    for r in rows:
        v = r[key]
        if np.isfinite(v):
            by_case.setdefault(r["case_id"], []).append(v)
    return np.array([np.median(v) for v in by_case.values()])


def stats(rows: list[dict]) -> dict:
    n_case = len({r["case_id"] for r in rows})
    q_case = case_medians(rows, "q")
    f_case = case_medians(rows, "f_px")
    q_frame = np.array([r["q"] for r in rows])
    f_frame = np.array([r["f_px"] for r in rows])
    f_frame = f_frame[np.isfinite(f_frame)]

    def qs(a):
        if a.size == 0:
            return None
        return {
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
        }

    frame_case_med = np.array([
        np.median([r["f_px"] for r in rows if r["case_id"] == c])
        for c in sorted({r["case_id"] for r in rows})
    ])
    frame_case_med = frame_case_med[np.isfinite(frame_case_med)]
    return {
        "n_frames": len(rows),
        "n_cases": n_case,
        "q_frame": qs(q_frame),
        "q_case": qs(q_case),
        "f_frame": qs(f_frame),
        "f_case": qs(f_case),
        "f_case_cv": float(np.std(frame_case_med) / np.mean(frame_case_med))
        if frame_case_med.size > 1 else None,
    }


def bootstrap_ratio(rows_a: list[dict], rows_b: list[dict], n: int = 5000,
                    seed: int = 0) -> dict:
    """病例级中位 f 的比值 b/a 及 95% bootstrap 区间。"""
    rng = np.random.default_rng(seed)
    a = case_medians(rows_a, "f_px")
    b = case_medians(rows_b, "f_px")
    vals = np.array([
        np.median(rng.choice(b, b.size)) / np.median(rng.choice(a, a.size))
        for _ in range(n)
    ])
    lo, med, hi = np.percentile(vals, [2.5, 50, 97.5])
    return {"ratio": float(med), "ci95": [float(lo), float(hi)]}


def size_trend(rows: list[dict]) -> dict:
    """病例级 log f ~ log truth 斜率；负斜率说明 f 随尺寸变小。"""
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        if np.isfinite(r["f_px"]):
            by_case.setdefault(r["case_id"], []).append(r)
    t = np.array([np.median([r["truth_mm"] for r in v]) for v in by_case.values()])
    f = np.array([np.median([r["f_px"] for r in v]) for v in by_case.values()])
    if t.size < 3:
        return {}
    order = np.argsort(t)
    rank = np.empty(t.size)
    rank[order] = np.arange(t.size)
    rho = float(np.corrcoef(rank, f)[0, 1])
    slope, intercept = np.polyfit(np.log(t), np.log(f), 1)
    pred = slope * np.log(t) + intercept
    r2 = float(1 - np.sum((np.log(f) - pred) ** 2)
               / np.sum((np.log(f) - np.log(f).mean()) ** 2))
    return {"n_cases": int(t.size), "spearman": rho,
            "loglog_slope": float(slope), "r2": r2}


def per_bin_medians(rows: list[dict], key: str = "f_px") -> dict[int, dict]:
    out = {}
    for b in range(1, 10):
        sel = [r for r in rows if b <= r["truth_mm"] < b + 1 and np.isfinite(r[key])]
        if not sel:
            continue
        case_vals = {}
        for r in sel:
            case_vals.setdefault(r["case_id"], []).append(r[key])
        vals = [np.median(v) for v in case_vals.values()]
        out[b] = {"median": float(np.median(vals)), "n_cases": len(vals),
                  "n_frames": len(sel)}
    return out


def box(ax, values, positions, colors, width=0.55):
    if isinstance(colors, str):
        colors = [colors] * len(values)
    bp = ax.boxplot(values, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, whis=(5, 95))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color("#444")


def draw(endo: dict, da: dict, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    names = list(MAIN)
    x = np.arange(1, len(names) + 1)

    q_e = [case_medians(endo[n], "q") for n in names]
    box(axes[0, 0], q_e, x, [COLORS[n] for n in names])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"{n}\n(n={len(v)})" for n, v in zip(names, q_e)],
                               fontsize=8)
    axes[0, 0].set_ylabel("q (mm, 病灶中位深度)")
    axes[0, 0].set_title("Endo：每病例 q_mm 分布")

    f_e = [case_medians(endo[n], "f_px") for n in names]
    box(axes[0, 1], f_e, x, [COLORS[n] for n in names])
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"{n}\n(n={len(v)})" for n, v in zip(names, f_e)],
                               fontsize=8)
    axes[0, 1].set_ylabel("f = q·span/truth (px)")
    axes[0, 1].set_title("Endo：隐含焦距（病例中位）")
    for xi, n, v in zip(x, names, f_e):
        axes[0, 1].annotate(f"{np.median(v):.0f}", (xi, np.median(v)),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8)

    for n in ("company_test", "B_test", "R_test"):
        pb = per_bin_medians(endo[n])
        bins = sorted(pb)
        axes[0, 2].plot(bins, [pb[b]["median"] for b in bins], marker="o",
                        color=COLORS[n], label=n)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xticks(range(1, 10))
    axes[0, 2].set_xlabel("truth 档 (mm)")
    axes[0, 2].set_ylabel("f (px, 病例中位)")
    axes[0, 2].set_title("Endo：逐真值档的隐含焦距")
    axes[0, 2].legend(fontsize=8)

    names_da = list(DA)
    xd = np.arange(1, len(names_da) + 1)
    q_d = [case_medians(da[n], "q") for n in names_da]
    box(axes[1, 0], q_d, xd, [COLORS[n] for n in names_da])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(xd)
    axes[1, 0].set_xticklabels([f"{n}\n(n={len(v)})" for n, v in zip(names_da, q_d)],
                               fontsize=8)
    axes[1, 0].set_ylabel("q (relative, 病灶中位)")
    axes[1, 0].set_title("DA：每病例 q（相对深度）分布")

    f_d = [case_medians(da[n], "f_px") for n in names_da]
    box(axes[1, 1], f_d, xd, [COLORS[n] for n in names_da])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(xd)
    axes[1, 1].set_xticklabels([f"{n}\n(n={len(v)})" for n, v in zip(names_da, f_d)],
                               fontsize=8)
    axes[1, 1].set_ylabel("f = q·span/truth (任意单位)")
    axes[1, 1].set_title("DA：隐含 f（相对深度，仅作跨队列对照）")
    for xi, n, v in zip(xd, names_da, f_d):
        axes[1, 1].annotate(f"{np.median(v):.3f}", (xi, np.median(v)),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8)

    for n in names_da:
        pb = per_bin_medians(da[n])
        bins = sorted(pb)
        axes[1, 2].plot(bins, [pb[b]["median"] for b in bins], marker="o",
                        color=COLORS[n], label=n)
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_xticks(range(1, 10))
    axes[1, 2].set_xlabel("truth 档 (mm)")
    axes[1, 2].set_ylabel("f (任意单位, 病例中位)")
    axes[1, 2].set_title("DA：逐真值档的隐含 f")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("EndoStreamDepth 米制刻度跨队列一致性（truth <10mm，病例中位）",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "f = q·span/truth；若同一内镜/标定，f 应跨病灶与距离近似常数。"
             "B/R 只有 test，无法用训练集标定。",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output/reports/endo_scale_20261106"))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="每队列最多取多少帧（0=全部）")
    ap.add_argument("--max-truth", type=float, default=10.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, summary = {}, {}
    for name, version, split, kind in COHORTS:
        print(f"[{name}] {version}/{split} ...", flush=True)
        rows = load_rows(name, version, split, kind, args.max_frames)
        rows = [r for r in rows if r["truth_mm"] < args.max_truth]
        all_rows[name] = rows
        summary[name] = {
            "data": f"{version}/{split}",
            "depth": kind,
            **stats(rows),
            "trend": size_trend(rows),
            "per_bin_f": per_bin_medians(rows),
        }
        print(f"  {name}: frames={len(rows)} cases={summary[name]['n_cases']} "
              f"q_med={summary[name]['q_case']['median']:.3f} "
              f"f_med={summary[name]['f_case']['median']:.3f}", flush=True)

    with (out / "frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[MAIN[0]][0]))
        w.writeheader()
        for name in MAIN + DA:
            w.writerows(all_rows[name])

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 跨队列常数因子：病例级中位 f 的比值 + bootstrap 区间
    ratios = {}
    for a, names in (("company_test", ("company_train", "company_val", "B_test", "R_test")),
                     ("company_test_DA", ("B_test_DA", "R_test_DA"))):
        for n in names:
            ratios[f"{n}/{a}"] = bootstrap_ratio(all_rows[a], all_rows[n])
    (out / "ratios.json").write_text(
        json.dumps(ratios, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Endo 米制刻度跨队列一致性（truth <10mm）", "",
        "f = q_mm · span_px / truth_mm；同一内镜/标定下应跨病灶、跨距离近似常数。", "",
        "| 队列 | 帧 | 病例 | q_mm 中位 [p25,p75] | f 中位 [p25,p75] (px) | f 病例间 CV |",
        "|---|---:|---:|---|---|---:|",
    ]
    for name, *_ in COHORTS:
        s = summary[name]
        q, f = s["q_case"], s["f_case"]
        lines.append(
            f"| {name} | {s['n_frames']} | {s['n_cases']} | "
            f"{q['median']:.2f} [{q['p25']:.2f}, {q['p75']:.2f}] | "
            f"{f['median']:.1f} [{f['p25']:.1f}, {f['p75']:.1f}] | "
            f"{s['f_case_cv']:.2%} |"
        )
    lines += ["", "## 相对 company_test 的隐含 f 比值（病例级 bootstrap 95% CI）", "",
              "| 比值 | 中位 | 95% CI |", "|---|---:|---|"]
    for k, v in ratios.items():
        lines.append(f"| {k} | {v['ratio']:.3f} | "
                     f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    lines += ["", "## 病例级 f ~ truth 趋势", "",
              "| 队列 | 病例 | Spearman | log-log 斜率 | R² |",
              "|---|---:|---:|---:|---:|"]
    for name, *_ in COHORTS:
        tr = summary[name]["trend"]
        lines.append(f"| {name} | {tr.get('n_cases', 0)} | "
                     f"{tr.get('spearman', float('nan')):+.2f} | "
                     f"{tr.get('loglog_slope', float('nan')):+.2f} | "
                     f"{tr.get('r2', float('nan')):.2f} |")
    lines += ["", "## 逐真值档 f（病例中位）", "",
              "| 档 | company_test | B_test | R_test |", "|---:|---:|---:|---:|"]
    for b in range(1, 10):
        vals = []
        for n in ("company_test", "B_test", "R_test"):
            pb = summary[n]["per_bin_f"].get(str(b), summary[n]["per_bin_f"].get(b))
            vals.append(f"{pb['median']:.2f} (n={pb['n_cases']})" if pb else "—")
        lines.append(f"| {b}–{b + 1} | " + " | ".join(vals) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    endo = {n: all_rows[n] for n in MAIN}
    da = {n: all_rows[n] for n in DA}
    draw(endo, da, out / "q_mm_cross_cohort.png")
    print(f"done: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
