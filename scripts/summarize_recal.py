#!/usr/bin/env python3
"""把 bias_recal_diagnosis.py 的 JSON 汇总成一节，追加进过夜报告（D-0066 第 4 步）。

只读 output/reports/recal_diagnosis/*.json，不重算任何指标。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("val_crossfit", "company", "B", "R")
LABEL = {
    "val_crossfit": "val（交叉拟合）",
    "company": "company",
    "B": "B",
    "R": "R",
}


def seed_of(run: str) -> str:
    m = re.search(r"_s(\d+)_", run)
    return m.group(1) if m else "?"


def f(v, nd=3) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(x) else f"{x:.{nd}f}"


def arrow(before, after, nd=3) -> str:
    return f"{f(before, nd)} → {f(after, nd)}"


def build(records: list[dict]) -> list[str]:
    records = sorted(records, key=lambda r: seed_of(r["run"]))
    L = [
        "",
        "## 5. 单调标定诊断（只在 val 拟合，test 仅套用）",
        "",
        f"拟合集：`{records[0]['fit_set']['version']}/{records[0]['fit_set']['split']}`，"
        f"{records[0]['fit_set']['n_frames']} 帧 / {records[0]['fit_set']['n_cases']} 例，"
        f"真值 [{records[0]['fit_set']['truth_min']:.1f}, "
        f"{records[0]['fit_set']['truth_max']:.1f}] mm。",
        "",
        "log-linear 斜率 b（>1 = 预测被压缩，标定要把它拉开）：",
        "",
        "| seed | a | b |",
        "|---|---:|---:|",
    ]
    for r in records:
        ll = r["log_linear"]
        L.append(f"| {seed_of(r['run'])} | {f(ll['a'])} | {f(ll['b'])} |")
    L += ["", "`<10mm` 帧级 MAE：标定前 → isotonic / log-linear", "",
          "| seed | " + " | ".join(LABEL[c] for c in COHORTS) + " |",
          "|---|" + "---:|" * len(COHORTS)]
    for r in records:
        cells = []
        for c in COHORTS:
            d = r.get(c) or {}
            cells.append(f"{arrow(d.get('lt10_before'), d.get('lt10_iso'))} / "
                         f"{f(d.get('lt10_lin'))}")
        L.append(f"| {seed_of(r['run'])} | " + " | ".join(cells) + " |")
    L += ["", "worst_bin（最差 1mm 档 MAE）：标定前 → isotonic / log-linear", "",
          "| seed | " + " | ".join(LABEL[c] for c in COHORTS) + " |",
          "|---|" + "---:|" * len(COHORTS)]
    for r in records:
        cells = []
        for c in COHORTS:
            d = r.get(c) or {}
            cells.append(f"{arrow(d.get('worst_bin_before'), d.get('worst_bin_iso'), 2)} / "
                         f"{f(d.get('worst_bin_lin'), 2)}")
        L.append(f"| {seed_of(r['run'])} | " + " | ".join(cells) + " |")
    L += ["", "超标档（>2mm）数量：标定前 → isotonic / log-linear", "",
          "| seed | " + " | ".join(LABEL[c] for c in COHORTS) + " |",
          "|---|" + "---:|" * len(COHORTS)]
    for r in records:
        cells = []
        for c in COHORTS:
            d = r.get(c) or {}
            cells.append(f"{len(d.get('bins_over2_before') or [])} → "
                         f"{len(d.get('bins_over2_iso') or [])} / "
                         f"{len(d.get('bins_over2_lin') or [])}")
        L.append(f"| {seed_of(r['run'])} | " + " | ".join(cells) + " |")
    base = records[0]
    L += [
        "",
        "逐档 MAE（seed "
        f"{seed_of(base['run'])}，company / B / R）：",
        "",
        "| 档 | " + " | ".join(f"{LABEL[c]} 前→iso" for c in COHORTS[1:]) + " |",
        "|---|" + "---:|" * (len(COHORTS) - 1),
    ]
    for b in range(1, 10):
        row = [f"{b}–{b+1}"]
        any_hit = False
        for c in COHORTS[1:]:
            d = base.get(c) or {}
            k = str(b)
            before = (d.get("per_bin_before") or {}).get(k)
            iso = (d.get("per_bin_iso") or {}).get(k)
            if before is not None:
                any_hit = True
            row.append(arrow(before, iso, 2))
        if any_hit:
            L.append("| " + " | ".join(row) + " |")
    sp = [f"{LABEL[c]}: {arrow((base.get(c) or {}).get('spearman'), (base.get(c) or {}).get('spearman_iso'), 3)}"
          for c in COHORTS]
    L += [
        "",
        "**分辨力没变**（单调映射不改变排序）：Spearman " + "；".join(sp) + "。",
        "",
        "结论口径：",
        "",
        "- test 逐档下降且超标档减少 → 压缩偏差**可以用 val 修正**，标定进配方（新决策）。",
        "- val 交叉拟合改善、test 不改善 → 偏差曲线跨队列不稳定（域偏移），标定只对同域有效。",
        "- 三种情况都不代表测量能力提升：档内 pred std 与 Spearman 不变，提升分辨力需要换监督信号。",
        "",
    ]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/reports/recal_diagnosis")
    ap.add_argument("--append-to", default="output/reports/overnight_20261104.md")
    ap.add_argument("--out", default="output/reports/overnight_20261104_recal.md")
    args = ap.parse_args()

    files = sorted((ROOT / args.dir).glob("*.json"))
    if not files:
        print(f"no recal JSON under {args.dir}")
        return 1
    records = []
    for p in files:
        r = json.loads(p.read_text())
        r["val_crossfit"] = r.get("val_crossfit") or {}
        for name, d in (r.get("test") or {}).items():
            r[name] = d
        records.append(r)

    lines = ["# 单调标定诊断汇总（D-0066）", ""] + build(records)
    text = "\n".join(lines) + "\n"
    (ROOT / args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")

    target = ROOT / args.append_to
    if target.is_file():
        cur = target.read_text(encoding="utf-8")
        if "## 5. 单调标定诊断" in cur:
            print(f"{args.append_to} 已含第 5 节，跳过追加")
        else:
            section = "\n".join(build(records)) + "\n"
            target.write_text(cur.rstrip("\n") + "\n" + section, encoding="utf-8")
            print(f"appended section 5 to {args.append_to}")
    else:
        print(f"{args.append_to} 不存在，只写独立文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
