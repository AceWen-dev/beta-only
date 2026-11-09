#!/usr/bin/env python3
"""小档（<3mm）标签/框约定审计：R 的失败来自标注口径还是模型？

D-0047 在 single-v2 上测出测试小档 `k=truth/(D·S)` 只有全局的 0.43–0.46×，
并留下第 3 步「盲法标签复核 + 框宽审计」。本脚本做其中的**量化**部分，
针对 company-v1 时代唯一统计上站得住的失败：REAL-colon R 的 1–2 / 2–3mm。

对同一 checkpoint 在四个队列上推理一次：
    company_train（company-v1/train，训练分布参考）
    company_test （company-v1/test_company，冻结 benchmark）
    B            （single-v2/test_b）
    R            （single-v2/test_r）

逐帧记录 k=truth/(D·S)、框几何、β/直接/门控三个分支，按 队列×档 汇总。

判据（预注册，与 D-0047 一致）：
- R 小档 k 中位数 ≤0.6× company_train 同档 → 标签/框约定不同，模型侧修不了；
- k 相近但 signed bias 单向且 |bias| ≥0.5mm → 域偏移/模型问题，可继续优化；
- 门控 wg 或 β 分支在 R 小档显著偏离 company → 融合机制在该域失效。

产物（默认 output/reports/smallbin_audit/）：
    frames.csv    逐帧明细
    cases.csv     逐病例（case_error_audit 同构，供 build_blind_review 使用）
    cases_r.csv   只含 R 的病例行（盲法复核包输入）
    summary.json  队列×档 汇总
    summary.md    同样内容的 markdown（早上直接看这个）

用法：
    python scripts/smallbin_audit.py --run output/runs/beta-regression/v11-company-base/<run_id>
    python scripts/smallbin_audit.py --run ... --device cpu --max-frames 8   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data")

from core.config import ModelConfig  # noqa: E402
from core.data.datasets.v11_frames import V11FramesDataset  # noqa: E402
from core.models.v11 import V11Net  # noqa: E402

BINS = list(range(1, 10))
SMALL = (1, 2)
REF_COHORT = "company_train"
COHORTS = {
    "company_train": ("company-v1", "train"),
    "company_test": ("company-v1", "test_company"),
    "B": ("single-v2", "test_b"),
    "R": ("single-v2", "test_r"),
}
# EndoStreamDepth 模型必须用同源特征评估（B/R 是 b-r-endo）
ENDO_COHORTS = {
    "company_train": ("company-v1-endo", "train"),
    "company_test": ("company-v1-endo", "test_company"),
    "B": ("b-r-endo", "test_b"),
    "R": ("b-r-endo", "test_r"),
}


def cohort_specs(data_version: str) -> dict:
    return ENDO_COHORTS if "endo" in str(data_version).lower() else COHORTS


def source_of(case_id: str) -> str:
    c = str(case_id)
    for p in ("company", "newdata", "phantom", "virtual", "realcolon", "polyp"):
        if c.startswith(p):
            return p
    return c.split("-")[0]


def rows_of(version: str, split: str) -> list[dict]:
    manifest = json.loads(
        (DATA / f"curated/v11_frames/{version}/manifest.json").read_text()
    )["rows"]
    return [r for r in manifest if r["split"] == split]


def collate(items):
    return {
        **{
            k: torch.stack([x[k] for x in items])
            for k in ("rgb_views", "depth", "scalars", "D", "S", "S_norm", "W", "y")
        },
        "case_id": [x["case_id"] for x in items],
        "sample_id": [x["sample_id"] for x in items],
    }


def infer(net, rows, device, batch_size: int = 8, input_size: int = 224,
          views: int = 2) -> dict[str, dict]:
    """按 sample_id 返回逐帧预测与分支量。DataLoader 不打乱，与 rows 顺序对齐。"""
    ds = V11FramesDataset(rows, DATA, views, None, input_size=input_size)
    out: dict[str, dict] = {}
    with torch.inference_mode():
        for b in DataLoader(ds, batch_size=batch_size, num_workers=2, collate_fn=collate):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            pred, _pooled, br = net(b, return_branches=True)
            sc = b["scalars"]
            for i, sid in enumerate(b["sample_id"]):
                out[str(sid)] = {
                    "pred": float(pred[i]),
                    # 分档头（D-0069）没有 β/direct/门控分支 → 缺失记 NaN，不伪造
                    "pred_beta": float(br["pred_beta"][i]) if "pred_beta" in br else float("nan"),
                    "pred_direct": float(br["pred_direct"][i]) if "pred_direct" in br else float("nan"),
                    "gate": float(br["gate"][i]) if "gate" in br else float("nan"),
                    "D": float(b["D"][i]),
                    "S": float(b["S"][i]),
                    "truth": float(b["y"][i]),
                    "bw_over_w": float(sc[i, 1]),
                    "bh_over_h": float(sc[i, 2]),
                    "h_over_w": float(sc[i, 5]),
                    "log_q": float(sc[i, 6]),
                    "mad_over_q": float(sc[i, 8]),
                    "fg_frac": float(sc[i, 9]),
                }
    return out


def bbox_px(row: dict) -> tuple[float, float]:
    bb = row.get("bbox_content") or [0, 0, 0, 0]
    return float(bb[2] - bb[0]), float(bb[3] - bb[1])


def med(xs) -> float:
    vals = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 目录（默认用 company-base 最优 run）")
    ap.add_argument("--checkpoint", default="final_ema.pt")
    ap.add_argument("--cohorts", default=",".join(COHORTS), help="逗号分隔")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时每队列只取前 N 帧（冒烟用）")
    ap.add_argument("--out", default="output/reports/smallbin_audit")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    resolved = json.loads((run / "config.resolved.json").read_text())
    net = V11Net(ModelConfig.from_dict(resolved["model"]))
    ck = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck.model)
    net = net.to(device).eval()
    print(f"run={run.name} ckpt={args.checkpoint} epoch={ck.epoch} device={device}")

    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    cohort_map = cohort_specs(resolved.get("data", {}).get("version", ""))
    frames: list[dict] = []
    for cohort in cohorts:
        version, split = cohort_map[cohort]
        rows = rows_of(version, split)
        rows.sort(key=lambda r: (str(r["case_id"]), str(r["sample_id"])))
        if args.max_frames:
            rows = rows[: args.max_frames]
        preds = infer(
            net, rows, device,
            input_size=int(resolved.get("data", {}).get("input_size", 224)),
            views=int(resolved.get("data", {}).get("views", 2)),
        )
        for r in rows:
            sid = str(r["sample_id"])
            if sid not in preds:
                continue
            p = preds[sid]
            bw, bh = bbox_px(r)
            ds = p["D"] * p["S"]
            truth = float(r["truth_mm"])
            frames.append({
                "cohort": cohort,
                "source": source_of(r["case_id"]),
                "case_id": str(r["case_id"]),
                "sample_id": sid,
                "bin": int(np.floor(truth)) if truth < 10 else 10,
                "truth": truth,
                "pred": round(p["pred"], 4),
                "signed_err": round(p["pred"] - truth, 4),
                "abs_err": round(abs(p["pred"] - truth), 4),
                "pred_beta": round(p["pred_beta"], 4),
                "pred_direct": round(p["pred_direct"], 4),
                "gate": round(p["gate"], 4),
                "D": round(p["D"], 6),
                "S": round(p["S"], 4),
                "k_true": round(truth / ds, 6) if ds > 0 else float("nan"),
                "k_pred": round(p["pred"] / ds, 6) if ds > 0 else float("nan"),
                "bbox_w_px": round(bw, 2),
                "bbox_h_px": round(bh, 2),
                "bbox_aspect": round(bw / bh, 4) if bh > 0 else float("nan"),
                "bw_over_w": round(p["bw_over_w"], 5),
                "bh_over_h": round(p["bh_over_h"], 5),
                "h_over_w": round(p["h_over_w"], 5),
                "mad_over_q": round(p["mad_over_q"], 5),
                "fg_frac": round(p["fg_frac"], 5),
            })
        print(f"  {cohort:14s} {len(rows):6d} 帧 → {len(preds)} 预测")

    if not frames:
        print("no frames"); return 1

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- 逐帧 CSV ----
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0]))
        w.writeheader()
        w.writerows(frames)

    # ---- 逐病例 CSV（case_error_audit 同构） ----
    by_case = defaultdict(list)
    for f in frames:
        by_case[(f["cohort"], f["case_id"])].append(f)
    SPLIT_NAME = {"company_train": "dev_train", "company_test": "company",
                  "B": "B", "R": "R"}
    cases = []
    for (cohort, cid), fs in by_case.items():
        ts = np.array([f["truth"] for f in fs])
        ps = np.array([f["pred"] for f in fs])
        cases.append({
            "split": SPLIT_NAME[cohort],
            "cohort": cohort,
            "source": fs[0]["source"],
            "case_id": cid,
            "n": len(ts),
            "truth_med": round(float(np.median(ts)), 3),
            "truth_min": round(float(ts.min()), 3),
            "truth_max": round(float(ts.max()), 3),
            "label_span": round(float(ts.max() - ts.min()), 3),
            "pred_mean": round(float(ps.mean()), 3),
            "signed_err": round(float(ps.mean() - np.median(ts)), 3),
            "within_std": round(float(ps.std()), 3),
            "abs_err": round(float(np.mean(np.abs(ps - ts))), 3),
            "k_true_med": round(med([f["k_true"] for f in fs]), 6),
        })
    fieldnames = list(cases[0])
    for name, subset in (("cases.csv", cases),
                         ("cases_r.csv", [c for c in cases if c["cohort"] == "R"])):
        if not subset:
            continue
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for c in sorted(subset, key=lambda x: (x["cohort"], x["source"],
                                                   -abs(x["signed_err"]))):
                w.writerow(c)

    # ---- 队列 × 档 汇总 ----
    agg = defaultdict(list)
    for f in frames:
        agg[(f["cohort"], f["bin"])].append(f)
    ref_k = {b: med([f["k_true"] for f in agg[(REF_COHORT, b)]])
             for b in BINS if (REF_COHORT, b) in agg}

    def case_weighted(fs) -> float:
        per = defaultdict(list)
        for f in fs:
            per[f["case_id"]].append(f["abs_err"])
        return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")

    summary = {}
    for (cohort, b), fs in sorted(agg.items()):
        ks = [f["k_true"] for f in fs]
        finite_k = [x for x in ks if np.isfinite(x)]
        summary[f"{cohort}|{b}"] = {
            "cohort": cohort,
            "bin": b,
            "n_frames": len(fs),
            "n_cases": len({f["case_id"] for f in fs}),
            "frame_mae": round(float(np.mean([f["abs_err"] for f in fs])), 3),
            "case_mae": round(case_weighted(fs), 3),
            "bias_mean": round(float(np.mean([f["signed_err"] for f in fs])), 3),
            "bias_median": round(med([f["signed_err"] for f in fs]), 3),
            "k_true_med": round(med(ks), 6),
            "k_true_q25": round(float(np.quantile(finite_k, 0.25)), 6) if finite_k else None,
            "k_true_q75": round(float(np.quantile(finite_k, 0.75)), 6) if finite_k else None,
            "k_pred_med": round(med([f["k_pred"] for f in fs]), 6),
            "k_ratio_vs_train": round(med(ks) / ref_k[b], 3)
            if b in ref_k and ref_k[b] else None,
            "gate_med": round(med([f["gate"] for f in fs]), 4),
            "beta_share_med": round(
                med([f["pred_beta"] / max(f["pred"], 1e-6)
                     for f in fs if f["pred_beta"] is not None]), 4),
            "bbox_w_px_med": round(med([f["bbox_w_px"] for f in fs]), 1),
            "bbox_aspect_med": round(med([f["bbox_aspect"] for f in fs]), 3),
            "fg_frac_med": round(med([f["fg_frac"] for f in fs]), 4),
            "S_med": round(med([f["S"] for f in fs]), 2),
            "D_med": round(med([f["D"] for f in fs]), 5),
        }
    # ---- 队列级 <10mm 总量（用来和官方 eval / evalmode.json 对齐验证） ----
    totals = {}
    for cohort in cohorts:
        fs = [f for f in frames if f["cohort"] == cohort and f["truth"] < 10]
        if not fs:
            continue
        totals[cohort] = {
            "n_frames": len(fs),
            "n_cases": len({f["case_id"] for f in fs}),
            "lt10_frame_mae": round(float(np.mean([f["abs_err"] for f in fs])), 4),
            "lt10_case_mae": round(case_weighted(fs), 4),
            "worst_bin": max(
                (round(float(np.mean([f["abs_err"] for f in fs if f["bin"] == b])), 3)
                 for b in sorted({f["bin"] for f in fs})), default=None),
        }
    (out / "summary.json").write_text(
        json.dumps({"run": run.name, "checkpoint": args.checkpoint,
                    "epoch": int(ck.epoch), "totals_lt10": totals, "cells": summary},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- markdown ----
    lines = [
        "# 小档标签/框约定审计（R vs company）",
        "",
        f"- run: `{run.name}`，checkpoint `{args.checkpoint}`（epoch {ck.epoch}）",
        f"- 队列: {', '.join(cohorts)}；`k = truth/(D·S)`，`k_ratio_vs_train` = "
        f"该队列该档 k 中位数 ÷ {REF_COHORT} 同档 k 中位数",
        "- 判据：小档 `k_ratio_vs_train ≤0.6` → 标注/框约定不同（模型侧修不了）；"
        "k 相近但 `bias_mean` 单向且 ≥0.5mm → 域偏移/模型问题",
        "",
        "## 队列级 <10mm（校验用：应与官方 eval / evalmode.json 一致）",
        "",
        "| 队列 | 帧 | 例 | <10mm 帧MAE | <10mm 病例MAE | 最差档 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in cohorts:
        t = totals.get(c)
        if not t:
            continue
        lines.append(f"| {c} | {t['n_frames']} | {t['n_cases']} | "
                     f"{t['lt10_frame_mae']:.3f} | {t['lt10_case_mae']:.3f} | "
                     f"{t['worst_bin']} |")
    lines.append("")
    for b in BINS:
        present = [c for c in cohorts if f"{c}|{b}" in summary]
        if not present:
            continue
        mark = " ← 小档" if b in SMALL else ""
        lines.append(f"## {b}–{b+1} mm{mark}")
        lines.append("")
        lines.append("| 队列 | 帧 | 例 | 帧MAE | 病例MAE | 偏差均值 | k中位 | k/train | 门控 | β占比 | 框宽px | fg |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in present:
            s = summary[f"{c}|{b}"]
            kr = s["k_ratio_vs_train"]
            lines.append(
                f"| {c} | {s['n_frames']} | {s['n_cases']} | {s['frame_mae']:.2f} | "
                f"{s['case_mae']:.2f} | {s['bias_mean']:+.2f} | {s['k_true_med']:.4f} | "
                f"{'—' if kr is None else f'{kr:.2f}×'} | {s['gate_med']:.2f} | "
                f"{s['beta_share_med']:.2f} | {s['bbox_w_px_med']:.0f} | "
                f"{s['fg_frac_med']:.3f} |")
        lines.append("")

    # R 小档逐病例（单向偏差检查）
    lines += ["## R 小档逐病例（1–3mm）", "",
              "| case | n | truth中位 | pred均值 | 偏差 | 帧内std | k中位 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in sorted([c for c in cases if c["cohort"] == "R" and c["truth_med"] < 3.5],
                    key=lambda x: x["signed_err"]):
        lines.append(f"| {c['case_id']} | {c['n']} | {c['truth_med']:.1f} | "
                     f"{c['pred_mean']:.2f} | {c['signed_err']:+.2f} | "
                     f"{c['within_std']:.2f} | {c['k_true_med']:.4f} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"frames: {out/'frames.csv'} ({len(frames)})")
    print(f"cases:  {out/'cases.csv'} ({len(cases)})")
    print(f"summary: {out/'summary.md'}")
    for b in SMALL:
        for c in cohorts:
            s = summary.get(f"{c}|{b}")
            if s:
                print(f"  {c:14s} {b}-{b+1}mm n={s['n_frames']:4d}/{s['n_cases']:3d}例 "
                      f"帧MAE={s['frame_mae']:.2f} 偏差={s['bias_mean']:+.2f} "
                      f"k={s['k_true_med']:.4f} k/train={s['k_ratio_vs_train']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
