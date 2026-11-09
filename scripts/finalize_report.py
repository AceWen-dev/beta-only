#!/usr/bin/env python3
"""收尾：为最优配方（3 seed）补官方 eval 记录 → 建 claim → 生成报告。

做法严格遵循项目规则：
- 测试成绩只能来自 eval.py 的 eval 记录（不手抄离线数字）；
- 每个 claim 引用 3 个 seed 的 eval id，并 pin benchmark_id；
- 报告由 `python -m dlef.cli report` 生成。
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dlef.claim import Claim, ClaimStore  # noqa: E402
from dlef.run import RunStore  # noqa: E402

RUNS = ROOT / "output/runs"
EXP = "v11-ranksim-log-g20"
BENCHES = (("d", None), ("b", "b"), ("r", "r"), ("clean", "clean"))


def seed_runs():
    out = []
    for run in RunStore(RUNS).iter_runs():
        if run.manifest.experiment != EXP or run.manifest.status != "completed":
            continue
        ck = run.directory / "checkpoints" / "final_ema.pt"
        if ck.is_file():
            out.append((run.manifest.seed, run, ck))
    return sorted(out)


def run_eval(ck: Path, suffix: str, test_set: str | None) -> dict:
    benchmark = ROOT / "input/benchmarks" / f"v11-{suffix}-test-v2.json"
    cmd = [
        sys.executable, "core/commands/eval.py",
        "--checkpoint", str(ck), "--split", "test",
        "--benchmark", str(benchmark),
    ]
    if test_set:
        cmd += ["--set", f"data.test_set={test_set}"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed for {suffix}: {proc.stderr[-800:]}")
    # 取该 eval 实验下最新的记录
    newest = None
    for run in RunStore(RUNS).iter_runs():
        if run.manifest.experiment != f"{EXP}-eval":
            continue
        for path in run.eval_paths():
            rec = json.loads(path.read_text())
            if rec.get("benchmark_id") and rec["benchmark_id"] in benchmark.read_text():
                if newest is None or rec["created_at"] > newest["created_at"]:
                    newest = rec
    if newest is None:
        raise RuntimeError(f"no eval record found for {suffix}")
    return newest


def main() -> int:
    seeds = seed_runs()
    if not seeds:
        print("no completed seed run with final_ema.pt")
        return 1
    by_cohort: dict[str, list[dict]] = {}
    for seed, run, ck in seeds:
        for suffix, test_set in BENCHES:
            rec = run_eval(ck, suffix, test_set)
            by_cohort.setdefault(suffix, []).append(rec)
            m = rec["metrics"]
            print(
                f"seed {seed} {suffix}: lt10={m['v11/mae_lt10']:.3f} "
                f"worst={m['v11/worst_bin']:.3f} eval={rec['eval_id']}",
                flush=True,
            )

    store = ClaimStore(ROOT / "output/claims")
    for suffix, recs in by_cohort.items():
        ids = tuple(r["eval_id"] for r in recs)
        bench = recs[0]["benchmark_id"]
        lt10 = statistics.mean(float(r["metrics"]["v11/mae_lt10"]) for r in recs)
        worst = statistics.mean(float(r["metrics"]["v11/worst_bin"]) for r in recs)
        store.add(Claim(
            claim_id=f"best-{suffix}-lt10",
            text=f"v11 {suffix.upper()} 测试集 <10mm MAE（{len(recs)} seed 均值）",
            eval_ids=ids, metric="v11/mae_lt10", value=lt10, unit="mm",
            benchmark_id=bench,
            notes=f"配方 log+RankSim(γ=2.0)+valmerge+EMA；seeds={[r['eval_id'].split('_s')[1][:8] for r in recs]}",
            tags={"experiment": EXP},
        ))
        store.add(Claim(
            claim_id=f"best-{suffix}-worst",
            text=f"v11 {suffix.upper()} 测试集 worst_bin MAE（{len(recs)} seed 均值）",
            eval_ids=ids, metric="v11/worst_bin", value=worst, unit="mm",
            benchmark_id=bench, tags={"experiment": EXP},
        ))
        print(f"claim {suffix}: lt10={lt10:.3f} worst={worst:.3f} ({len(ids)} evals)")

    out = ROOT / "output/reports/report.md"
    subprocess.run(
        [sys.executable, "-m", "dlef.cli", "report", "--runs", str(RUNS),
         "--claims", str(ROOT / "output/claims"), "--out", str(out)],
        cwd=ROOT, check=True,
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
