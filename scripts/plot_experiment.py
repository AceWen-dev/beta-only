#!/usr/bin/env python3
"""单实验逐区间图（只画该实验自身曲线，eval 模式现场推理）。

约定（2026-11-02 起）：**每个跑完的实验都要出这张图**。

用法：
    python scripts/plot_experiment.py --run output/runs/beta-regression/<exp>/<run_id>
    python scripts/plot_experiment.py --run ... --checkpoints best.pt,last.pt --name my_exp

产物：
    output/charts/<name>.png        （name 默认取 run 的 experiment 名）
    <run_dir>/per_bin_mae.png       （随 run 目录留档）
    output/reports/<name>_evalmode.json（逐档数值）

图上：四面板按数据集标注（D 公司 / B PolypSize / R REAL-colon / clean=D 子集），
红虚线 = 2mm 达标线，右上角标每个 checkpoint 的 <10mm 总 MAE。
"""

from __future__ import annotations

import argparse
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
from core.tensorboard import write_benchmark_scalars, write_chart_image  # noqa: E402

SPLITS = {"d": "test_d", "b": "test_b", "r": "test_r", "clean": "test_clean",
          "company": "test_company"}
# R 的 1–2mm 档在 company_train 中没有任何样本（D-0047），2026-11-05 起
# benchmark2 的 R 队列默认排除该档（只纳入 truth >= 2mm）；suite 的
# min_truth_mm 可显式覆盖。冻结的 v11-r-test-v2.json 本身不变。
MIN_TRUTH_DEFAULT = {"r": 2.0}
# B 的 7–8mm 只有一个病例/5 帧（polyp-size-04），2026-11-06 起按离群档从
# benchmark2 出图/汇总中排除；冻结的 v11-b-test-v2 本身不变。
EXCLUDE_BINS_DEFAULT = {"b": (7,)}
COLORS = ("#3572b0", "#e07b39", "#3a9d5d", "#8e6bbf")


def filter_rows(rows, split: str, min_truth_mm: float = 0.0,
                exclude_bins=()):
    """按 split、最小真值和排除档位筛选成员（只影响离线出图/汇总）。"""
    excluded = {int(b) for b in exclude_bins}
    return [
        r for r in rows
        if r["split"] == split
        and float(r["truth_mm"]) >= min_truth_mm
        and int(float(r["truth_mm"])) not in excluded
    ]


def measure(run: Path, ckpt: str, cohorts: list[str], member_map=None):
    """ckpt 支持 "文件:ema" 语法（如 last.pt:ema → 取该 checkpoint 里存的 EMA 权重）。"""
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    use_ema = ckpt.endswith(":ema")
    fname = ckpt[:-4] if use_ema else ckpt
    ck = torch.load(run / "checkpoints" / fname, map_location="cpu", weights_only=False)
    if use_ema:
        if ck.ema is None:
            raise RuntimeError(f"{fname} has no ema state")
        net.load_state_dict(dict(ck.ema))
    else:
        net.load_state_dict(ck.model)
    net = net.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    device = next(net.parameters()).device
    version = str(resolved.get("data", {}).get("version", "single-v2"))
    manifest_cache = {}

    def manifest_rows(name: str):
        if name not in manifest_cache:
            manifest_cache[name] = json.loads(
                (DATA / f"curated/v11_frames/{name}/manifest.json").read_text()
            )["rows"]
        return manifest_cache[name]

    out = {}
    for cohort in cohorts:
        if member_map and cohort in member_map:
            spec = member_map[cohort]
            m_version, m_split = spec[0], spec[1]
            m_min_truth = float(spec[2]) if len(spec) > 2 else 0.0
            m_exclude = tuple(spec[3]) if len(spec) > 3 else ()
        else:
            # 无 suite 时的旧口径：company 用 run 的版本，其余 single-v2
            m_version = version if cohort == "company" else "single-v2"
            m_split = SPLITS[cohort]
            m_min_truth = MIN_TRUTH_DEFAULT.get(cohort, 0.0)
            m_exclude = EXCLUDE_BINS_DEFAULT.get(cohort, ())
        rows = filter_rows(manifest_rows(m_version), m_split, m_min_truth,
                           m_exclude)
        ds = V11FramesDataset(
            rows, DATA, int(resolved.get("data", {}).get("views", 2)), None,
            input_size=int(resolved.get("data", {}).get("input_size", 224)),
        )

        def collate(items):
            return {
                **{k: torch.stack([x[k] for x in items]) for k in
                   ("rgb_views", "depth", "scalars", "D", "S", "S_norm", "W", "y")},
                "case_id": [x["case_id"] for x in items],
            }

        ps, ts = [], []
        with torch.inference_mode():
            for b in DataLoader(ds, batch_size=8, num_workers=2, collate_fn=collate):
                b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
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
    return int(ck.epoch), out


def draw(curves: dict, lt10: dict, title: str, out: Path, cohorts=None):
    finite = []
    for _, c in curves.values():
        for key, arr in c.items():
            if key.endswith("__band"):
                continue
            finite += [v for v in np.ravel(arr) if np.isfinite(v)]
    y_max = max(5.0, (max(finite) + 0.5) if finite else 5.0)
    panels = list(cohorts or pp.COHORTS)
    if len(panels) == 1:
        fig, ax0 = plt.subplots(figsize=(9, 5))
        axes = np.array([ax0])
    elif len(panels) <= 3:
        fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5))
        axes = np.atleast_1d(axes).ravel()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharex=True, sharey=True)
        axes = axes.ravel()
    for ax, (cohort, label) in zip(axes, panels):
        if not any(cohort in c for _, c in curves.values()):
            ax.set_visible(False)
            continue
        ax.axhspan(pp.THRESHOLD, y_max, color="#d62728", alpha=0.05)
        for boundary in range(1, 10):
            ax.axvline(boundary, color="#e3e3e3", linewidth=0.8, zorder=0)
        box = []
        for lbl, (color, c) in curves.items():
            if cohort not in c:
                continue
            ax.plot(pp.X, c[cohort], marker="o", markersize=4, linewidth=2.0,
                    color=color, label=lbl)
            band = c.get(cohort + "__band")
            if band is not None:
                ax.fill_between(pp.X, band[0], band[1], color=color, alpha=0.15,
                                linewidth=0)
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
    if len(panels) == 1:
        axes[0].set_ylabel("MAE (mm)")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    else:
        axes[0].set_ylabel("MAE (mm)")
        if len(axes) > 2:
            axes[2].set_ylabel("MAE (mm)")
        fig.suptitle(title + " · 冻结测试集逐区间 MAE（红虚线 = 2mm）",
                     fontsize=13, fontweight="bold")
        fig.text(0.5, 0.01, "eval 模式现场推理（D-0023 修复后）· 只画本实验自身 checkpoint · "
                 "N/A = 该区间无样本", ha="center", fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run 目录")
    ap.add_argument("--runs", default=None, help="多 run（逗号分隔，同实验多 seed → 均值带）")
    ap.add_argument("--note", default="", help="标题补充说明（如旧管线）")
    ap.add_argument("--checkpoints", default="final_ema.pt,best.pt")
    ap.add_argument("--cohorts", default="d,b,r,clean")
    ap.add_argument("--suite", default="",
                    help="benchmark suite 文件（如 input/benchmarks/benchmark2.json）")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    runs_arg = args.runs or args.run
    if not runs_arg:
        print("need --run or --runs"); return 2
    runs = [Path(r.strip()).resolve() for r in runs_arg.split(",") if r.strip()]
    run = runs[0]
    name = args.name or json.loads((run / "manifest.json").read_text())["experiment"]
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    # D-0059：数据版本不是 single-v2 时，默认只评该版本自己的 test 划分。
    # 旧 d/b/r/clean 与新训练集高度重叠（D 209/356、clean 137/206 帧），
    # 拿它们给公司模型打分会产生严重偏乐观的数字。
    resolved = json.loads((run / "config.resolved.json").read_text())
    version = str(resolved.get("data", {}).get("version", ""))
    suite_path = Path(args.suite) if args.suite else None
    member_map = None
    SUITES = {
        "company-v1": "benchmark2.json",
        "company-v1-endo": "benchmark2-endo.json",
        "company-v1-big": "benchmark2-big.json",
    }
    if suite_path is None and version and version != "single-v2":
        suite_name = SUITES.get(version)
        candidate = ROOT / "input/benchmarks" / suite_name if suite_name else None
        if candidate and candidate.is_file():
            suite_path = candidate
    if suite_path is not None:
        suite = json.loads(suite_path.read_text())
        cohorts = list(suite["members"])
        member_map = {}
        for c, spec in suite["members"].items():
            if isinstance(spec, dict):
                member_map[c] = (
                    spec["version"],
                    spec["split"],
                    float(spec.get("min_truth_mm", 0.0)),
                    tuple(int(x) for x in spec.get("exclude_bins", [])),
                )
            else:  # 旧格式：只给文件名，用默认映射
                member_map[c] = None
        member_map = {c: v for c, v in member_map.items() if v}
    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    multi = len(runs) > 1

    curves, lt10, raw = {}, {}, {}
    for i, ckpt in enumerate(ckpts):
        per_run = []
        fname = ckpt[:-4] if ckpt.endswith(":ema") else ckpt
        for rd in runs:
            if not (rd / "checkpoints" / fname).is_file():
                continue
            _, out = measure(rd, ckpt, cohorts, member_map)
            per_run.append(out)
        if not per_run:
            continue
        if per_run[0] and not any(c in per_run[0] for c in cohorts):
            continue
        if multi:
            lbl = f"{ckpt}（{len(per_run)} seed 均值）"
            arrs = {}
            lts = {}
            for c in cohorts:
                stack = np.stack([
                    np.array([o[c]["per_bin"].get(k, np.nan) for k in pp.BINS])
                    for o in per_run if c in o
                ])
                with np.errstate(invalid="ignore"):
                    arrs[c] = np.nanmean(stack, axis=0)
                    lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                arrs[c + "__band"] = (lo, hi)
                lts[c] = float(np.nanmean([o[c]["lt10"] for o in per_run if c in o]))
            curves[lbl] = (COLORS[i % len(COLORS)], arrs)
            lt10[lbl] = lts
            raw[ckpt] = [o for o in per_run]
        else:
            out = per_run[0]
            ep = out[cohorts[0]]["epoch"]
            lbl = f"{ckpt} (ep{ep})"
            curves[lbl] = (COLORS[i % len(COLORS)], {
                c: np.array([out[c]["per_bin"].get(k, np.nan) for k in pp.BINS]) for c in out
            })
            lt10[lbl] = {c: out[c]["lt10"] for c in out}
            raw[ckpt] = out
        # 离线评测也写 TensorBoard（benchmark/<cohort>/<checkpoint>/mae_by_bin），
        # 使 eval.py 之外测出的 benchmark 指标同样能在 TB 查看（D-0030）。
        for cohort_name, res in out.items():
            write_benchmark_scalars(
                run,
                {f"v11/mae_{k}_{k + 1}": v for k, v in res["per_bin"].items()}
                | {"v11/mae_lt10": res["lt10"], "v11/mae": res["mae"]},
                cohort=cohort_name.upper() if cohort_name in ("b", "d", "r") else cohort_name,
                checkpoint=ckpt.replace(".pt", "").replace(":", "-"),
            )
        print(lbl, {c: round(v, 3) for c, v in lt10[lbl].items()})

    if not curves:
        print("no checkpoint found"); return 1
    chart = ROOT / "output/charts" / f"{name}.png"
    panels = [(c, dict(pp.COHORTS).get(c, c)) for c in cohorts]
    draw(curves, lt10, name + (f" · {args.note}" if args.note else ""), chart,
         cohorts=panels)
    (run / "per_bin_mae.png").write_bytes(chart.read_bytes())
    # 组合图进 TB
    for rd in runs:
        write_chart_image(rd, chart, tag="chart/per_bin_mae")

    rep = ROOT / "output/reports" / f"{name}_evalmode.json"
    rep.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"chart: {chart}")
    print(f"copy:  {run / 'per_bin_mae.png'}")
    print(f"json:  {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
