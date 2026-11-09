#!/usr/bin/env python3
"""过夜队列汇总（D-0066）：把三 seed 噪声底 + 小档审计拼成一份 markdown。

所有数字都从已落盘的 JSON 读，不手抄：
- `output/reports/v11-company-base-3seed_evalmode.json`  三 seed 离线逐档（company/B/R）
- `output/runs/beta-regression/v11-company-base-eval/*/evals/*.json`  公司 test 官方 eval
- 各变体的 `*_evalmode.json`（ctx / endo / big）用于「Δ 是否落在噪声内」判定
- `output/reports/smallbin_audit/summary.{json,md}`  小档审计

用法：
    python scripts/summarize_overnight.py \
        --runs-order <s21_dir>,<s22_dir>,<base_dir> \
        --out output/reports/overnight_20261104.md
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("company", "b", "r")
COHORT_LABEL = {"company": "公司 test", "b": "PolypSize B", "r": "REAL-colon R"}
BINS = tuple(range(1, 10))
THRESHOLD = 2.0
VARIANTS = {
    "ctx（D-0060 否决）": "output/reports/v11-company-ctx_evalmode.json",
    "big（D-0064 否决）": "output/reports/v11-company-base-big_evalmode.json",
    "endo（D-0065 等价）": "output/reports/v11-company-base-endo_evalmode.json",
}


def jload(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def seed_of(run_dir: str) -> str:
    m = re.search(r"_s(\d+)_", str(run_dir))
    return m.group(1) if m else "?"


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{nd}f}"


def sfmt(v, nd=3):
    """带符号版本；缺失/非有限时同样返回破折号。"""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:+.{nd}f}"


def per_seed_offline(three: dict, order: list[str]) -> dict[str, dict]:
    """{seed: {cohort: {'lt10':..,'per_bin':{bin:mae}}}}，取 final_ema.pt。"""
    out: dict[str, dict] = {}
    block = (three or {}).get("final_ema.pt")
    if not isinstance(block, list):
        return out
    for entry, run_dir in zip(block, order):
        seed = seed_of(run_dir)
        out[seed] = {
            c: {"lt10": entry.get(c, {}).get("lt10"),
                "per_bin": {int(k): v for k, v in entry.get(c, {}).get("per_bin", {}).items()}}
            for c in COHORTS if c in entry
        }
    return out


def variant_lt10(path: Path) -> dict[str, float]:
    raw = jload(path) or {}
    block = raw.get("final_ema.pt")
    if isinstance(block, list):
        block = block[0] if block else None
    if not isinstance(block, dict):
        return {}
    return {c: block.get(c, {}).get("lt10") for c in COHORTS if c in block}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def final_ema_digests(order: list[str]) -> dict[str, str]:
    """{checkpoint_digest: seed}，只收 --runs-order 里各 run 的 final_ema.pt。

    eval 记录不带 checkpoint 路径，只带 `sha256_file(checkpoint)`（DLEF runner.py），
    所以用摘要把官方记录绑回具体 seed 的 final_ema。
    """
    out: dict[str, str] = {}
    for rd in order:
        p = ROOT / rd / "checkpoints" / "final_ema.pt"
        if p.is_file():
            try:
                out.setdefault(sha256_file(p), seed_of(rd))
            except OSError:
                pass
    return out


def official_evals(digests: dict[str, str]) -> dict[str, dict]:
    """{seed: {'lt10':..,'worst_bin':..,...}}（公司 test 官方口径，仅 final_ema）。"""
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(str(ROOT / "output/runs/beta-regression/"
                                  "v11-company-base-eval/*/evals/*.json"))):
        rec = jload(Path(p))
        if not rec or rec.get("split") != "test":
            continue
        seed = digests.get(str(rec.get("checkpoint_digest", "")))
        if seed is None:
            continue
        m = rec.get("metrics") or {}
        out[seed] = {
            "lt10": m.get("v11/mae_lt10"),
            "worst_bin": m.get("v11/worst_bin"),
            "acc10": m.get("v11/acc10"),
            "eval_id": rec.get("eval_id"),
            "benchmark_id": rec.get("benchmark_id"),
        }
    return out


def spread(vals: list[float]) -> tuple[float, float, float, float]:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    if not v:
        return (float("nan"),) * 4
    return float(sum(v) / len(v)), float(min(v)), float(max(v)), float(max(v) - min(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-order", required=True,
                    help="与 plot_experiment --runs 相同顺序的 run 目录，逗号分隔")
    ap.add_argument("--three-seed-json",
                    default="output/reports/v11-company-base-3seed_evalmode.json")
    ap.add_argument("--audit", default="output/reports/smallbin_audit/summary.json")
    ap.add_argument("--out", default="output/reports/overnight_20261104.md")
    ap.add_argument("--steps-log", default="output/logs/overnight_20261104.steps.log")
    args = ap.parse_args()

    order = [r.strip() for r in args.runs_order.split(",") if r.strip()]
    three = jload(ROOT / args.three_seed_json)
    per_seed = per_seed_offline(three, order)
    official = official_evals(final_ema_digests(order))
    audit = jload(ROOT / args.audit) or {}
    cells = audit.get("cells", {})
    totals = audit.get("totals_lt10", {})

    L: list[str] = ["# 过夜结果汇总（D-0066：seed 噪声底 + 小档标签/框审计）", ""]

    # ---------- 0. 速览 ----------
    L += ["## 0. 结论速览", ""]
    stats = {}
    for c in COHORTS:
        stats[c] = spread([s.get(c, {}).get("lt10") for s in per_seed.values()])
    base_seed = "20261020"
    for c in COHORTS:
        mean, lo, hi, sp = stats[c]
        L.append(f"- **{COHORT_LABEL[c]}** `<10mm` 三 seed：均值 **{fmt(mean)}**，"
                 f"范围 {fmt(lo)}–{fmt(hi)}，**散布 {fmt(sp)}**")
    L.append("")
    if per_seed:
        L.append("| 变体（单 seed，均为 s20261020） | 公司 Δ | B Δ | R Δ | 是否落在 seed 散布内 |")
        L.append("|---|---:|---:|---:|---|")
        base_lt = {c: (per_seed.get(base_seed, {}).get(c, {}) or {}).get("lt10")
                   for c in COHORTS}
        for label, rel in VARIANTS.items():
            v = variant_lt10(ROOT / rel)
            if not v:
                L.append(f"| {label} | 数据缺失 | | | |")
                continue
            deltas, inside = {}, []
            for c in COHORTS:
                d = (v.get(c) - base_lt[c]) if (v.get(c) is not None and base_lt.get(c) is not None) else None
                deltas[c] = d
                sp = stats[c][3]
                inside.append("—" if d is None or not math.isfinite(sp)
                              else ("是" if abs(d) <= sp else "**否**"))
            L.append(f"| {label} | {sfmt(deltas['company'])} | {sfmt(deltas['b'])} | "
                     f"{sfmt(deltas['r'])} | {' / '.join(inside)} |")
        L.append("")
        L.append("> 判定口径：|Δ| ≤ 同队列三 seed 全距 → 该变体差异**不可区分**，"
                 "原「否决/等价」结论需要重述为「单 seed 下未观察到收益」。")
        L.append("")

    # ---------- 1. 三 seed 明细 ----------
    L += ["## 1. 三 seed 明细（final_ema）", "",
          "| seed | 公司 test 官方 eval `<10mm` | worst_bin | acc10 | 离线 公司 | B | R | run |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for run_dir in order:
        s = seed_of(run_dir)
        off = per_seed.get(s, {})
        ev = official.get(s, {})
        L.append(f"| {s} | {fmt(ev.get('lt10'))} | {fmt(ev.get('worst_bin'))} | "
                 f"{fmt(ev.get('acc10'))} | {fmt((off.get('company') or {}).get('lt10'))} | "
                 f"{fmt((off.get('b') or {}).get('lt10'))} | "
                 f"{fmt((off.get('r') or {}).get('lt10'))} | `{Path(run_dir).name}` |")
    L.append("")

    # ---------- 2. 逐档 ----------
    L += ["## 2. 逐档 MAE：三 seed 均值（范围）", "",
          "「全 seed 超标」= 三个 seed 都 >2mm，是真失败；「部分 seed 超标」= 噪声档，"
          "不该作为优化目标。", ""]
    AUDIT_COHORT = {"company": "company_test", "b": "B", "r": "R"}
    for c in COHORTS:
        L.append(f"### {COHORT_LABEL[c]}")
        L.append("")
        L.append("| 档 | 均值 | 范围 | 超标 seed 数 | n（帧/例） | 偏差均值（s20261020） |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for b in BINS:
            vals = [(s.get(c, {}).get("per_bin") or {}).get(b) for s in per_seed.values()]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if not vals:
                continue
            mean, lo, hi, _ = spread(vals)
            n_over = sum(1 for v in vals if v > THRESHOLD)
            flag = "**全 seed 超标**" if n_over == len(vals) and len(vals) >= 3 else (
                f"{n_over}/{len(vals)}" if n_over else "0")
            cell = cells.get(f"{AUDIT_COHORT[c]}|{b}") or {}
            nf = (f"{cell['n_frames']}/{cell['n_cases']}" if cell else "—")
            bias = cell.get("bias_mean")
            L.append(f"| {b}–{b+1} | {fmt(mean)} | {fmt(lo)}–{fmt(hi)} | {flag} | {nf} | "
                     f"{sfmt(bias, 2)} |")
        L.append("")

    # ---------- 3. 小档审计 ----------
    L += ["## 3. 小档标签/框审计（k = truth/(D·S)）", ""]
    if not cells:
        L += ["审计结果缺失（`output/reports/smallbin_audit/summary.json` 未生成）。", ""]
    else:
        L += [f"- checkpoint：`{audit.get('checkpoint')}`（epoch {audit.get('epoch')}），"
              f"run `{audit.get('run')}`", ""]
        if totals:
            L += ["**校验**（应与官方 eval / evalmode.json 一致）：", "",
                  "| 队列 | 帧 | 例 | `<10mm` 帧MAE | 病例MAE |", "|---|---:|---:|---:|---:|"]
            for c, t in totals.items():
                L.append(f"| {c} | {t['n_frames']} | {t['n_cases']} | "
                         f"{fmt(t['lt10_frame_mae'])} | {fmt(t['lt10_case_mae'])} |")
            L.append("")
        L += ["**判据**：小档 `k_ratio_vs_train ≤0.6` → 标注/框约定不同（模型侧修不了）；"
              "k 相近但 `bias_mean` 单向且 ≥0.5mm → 域偏移/模型问题。", "",
              "| 队列 | 档 | 帧 | 例 | 帧MAE | 病例MAE | 偏差均值 | k中位 | k/train | 门控 | β占比 | 框宽px |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b in (1, 2, 3):
            for c in ("company_train", "company_test", "B", "R"):
                s = cells.get(f"{c}|{b}")
                if not s:
                    continue
                kr = s.get("k_ratio_vs_train")
                L.append(f"| {c} | {b}–{b+1} | {s['n_frames']} | {s['n_cases']} | "
                         f"{fmt(s['frame_mae'], 2)} | {fmt(s['case_mae'], 2)} | "
                         f"{sfmt(s['bias_mean'], 2)} | {s['k_true_med']:.4f} | "
                         f"{'—' if kr is None else f'{kr:.2f}×'} | {s['gate_med']:.2f} | "
                         f"{s['beta_share_med']:.2f} | {s['bbox_w_px_med']:.0f} |")
        L.append("")
        # 机械判定
        verdicts = []
        for b in (1, 2):
            s = cells.get(f"R|{b}")
            if not s:
                continue
            kr = s.get("k_ratio_vs_train")
            bias = s.get("bias_mean")
            if kr is not None and kr <= 0.6:
                v = "标注/框约定不同 → 模型侧不可修"
            elif bias is not None and abs(bias) >= 0.5:
                v = "域偏移/模型问题 → 可继续优化"
            else:
                v = "未触发判据（k 与偏差都在阈内）"
            verdicts.append(f"- R {b}–{b+1}mm（{s['n_frames']} 帧 / {s['n_cases']} 例，"
                            f"帧MAE {fmt(s['frame_mae'], 2)}）：k/train="
                            f"{'—' if kr is None else f'{kr:.2f}×'}、偏差 {sfmt(bias, 2)} → **{v}**")
        L += verdicts or ["- R 小档无样本，无法判定"]
        L.append("")

    # ---------- 4. 步骤日志 ----------
    L += ["## 4. 队列步骤日志", ""]
    steps = Path(args.steps_log)
    if steps.is_file():
        L += ["```", *steps.read_text().splitlines()[-60:], "```"]
    else:
        L += [f"（未找到 `{steps}`）"]
    L.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
