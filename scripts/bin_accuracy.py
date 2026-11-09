#!/usr/bin/env python3
"""分档准确率诊断（D-0069）：把逐档 MAE 拆成「分错档」与「档内误差」两部分。

读 `scripts/smallbin_audit.py` 产出的 `frames.csv`（含 cohort/bin/truth/pred），
按队列报告：
- `bin_acc`：`bin_of(pred) == bin_of(truth)` 的比例（1mm 档命中）
- `bin_acc_pm1`：预测档与真值档相差 ≤1 的比例
- `mae_hit` / `mae_miss`：命中档与分错档各自的 MAE
- 逐档命中数，用来定位「哪一档在漏」

判据背景：档宽 1mm 且标签是整数 mm，**只要分档正确，档内误差必然 <1mm**，
所以「每个 <10mm 档 MAE <2mm」在分档准确率足够高时自动成立。

用法：
    python scripts/bin_accuracy.py --frames output/reports/smallbin_audit_ordinal/frames.csv \
        [--frames 另一份.csv --label base] --out output/reports/ordinal_bin_accuracy.md
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

JUDGED = ("company_test", "B")
LABEL = {"company_test": "公司 test", "B": "PolypSize B", "R": "REAL-colon R",
         "company_train": "company train"}


def bin_of(value: float) -> int:
    """与 core/losses/mm_case_bin.bin_of 同口径。"""
    return 10 if value >= 10.0 else int(value)


def rows_of(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def analyse(rows: list[dict], cohort: str) -> dict | None:
    sub = [r for r in rows if r["cohort"] == cohort]
    sub = [r for r in sub if float(r["truth"]) < 10.0]   # 只判 <10mm
    if not sub:
        return None
    hit = miss = pm1 = 0
    err_hit: list[float] = []
    err_miss: list[float] = []
    per_bin: dict[int, list[int]] = defaultdict(lambda: [0, 0])   # bin -> [命中, 总数]
    for r in sub:
        t, p = float(r["truth"]), float(r["pred"])
        tb, pb = bin_of(t), bin_of(p)
        per_bin[tb][1] += 1
        ok = tb == pb
        per_bin[tb][0] += int(ok)
        hit += int(ok)
        miss += int(not ok)
        pm1 += int(abs(tb - pb) <= 1)
        (err_hit if ok else err_miss).append(abs(p - t))
    n = len(sub)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")   # noqa: E731
    return {
        "n": n, "cases": len({r["case_id"] for r in sub}),
        "bin_acc": hit / n, "bin_acc_pm1": pm1 / n,
        "mae_hit": mean(err_hit), "mae_miss": mean(err_miss),
        "n_miss": miss,
        "mae_lt10": mean([abs(float(r["pred"]) - float(r["truth"])) for r in sub]),
        "per_bin": dict(sorted(per_bin.items())),
    }


def render(blocks: list[tuple[str, str, dict]]) -> str:
    L = ["# 分档准确率诊断（D-0069）", ""]
    L.append("档宽 1mm、标签整数 mm → **分档正确时档内误差必然 <1mm**，"
             "「每个 <10mm 档 MAE <2mm」在分档准确率足够高时自动成立。")
    L.append("")
    L.append("| 来源 | 队列 | 帧/例 | `<10mm` MAE | 分档准确率 | ±1 档 | 命中档 MAE | 分错档 MAE | 分错帧 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, cohort, st in blocks:
        judged = "" if cohort in JUDGED else "（仅观察）"
        miss = "—" if math.isnan(st["mae_miss"]) else f"{st['mae_miss']:.3f}"
        L.append(f"| {label} | {LABEL.get(cohort, cohort)}{judged} | {st['n']}/{st['cases']} "
                 f"| {st['mae_lt10']:.3f} | **{st['bin_acc'] * 100:.1f}%** "
                 f"| {st['bin_acc_pm1'] * 100:.1f}% | {st['mae_hit']:.3f} "
                 f"| {miss} | {st['n_miss']} |")
    L.append("")
    for label, cohort, st in blocks:
        L.append(f"## {label} · {LABEL.get(cohort, cohort)} 逐档命中")
        L.append("")
        L.append("| 档(mm) | 命中/总数 | 命中率 |")
        L.append("|---|---:|---:|")
        for b, (ok, tot) in st["per_bin"].items():
            name = f"{b}–{b + 1}" if b < 10 else "≥10"
            L.append(f"| {name} | {ok}/{tot} | {ok / tot * 100:.1f}% |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", action="append", required=True,
                    help="smallbin_audit 的 frames.csv；可重复，按 --label 顺序配对")
    ap.add_argument("--label", action="append", default=[],
                    help="每份 frames.csv 的显示名（缺省用文件名推断）")
    ap.add_argument("--cohorts", default=",".join(JUDGED))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    labels = list(a.label) + [Path(p).parent.name for p in a.frames[len(a.label):]]
    blocks: list[tuple[str, str, dict]] = []
    for label, path in zip(labels, a.frames):
        rows = rows_of(Path(path))
        for cohort in [c.strip() for c in a.cohorts.split(",") if c.strip()]:
            st = analyse(rows, cohort)
            if st:
                blocks.append((label, cohort, st))
    if not blocks:
        raise SystemExit("没有可用数据（cohort 或 <10mm 帧为空）")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(blocks))
    print(f"wrote {out}")
    for label, cohort, st in blocks:
        print(f"  {label:24s} {cohort:14s} n={st['n']:>5} bin_acc={st['bin_acc'] * 100:5.1f}% "
              f"mae_lt10={st['mae_lt10']:.3f} mae_hit={st['mae_hit']:.3f} "
              f"mae_miss={st['mae_miss']:.3f}")


if __name__ == "__main__":
    main()
