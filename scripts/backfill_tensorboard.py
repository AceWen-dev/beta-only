#!/usr/bin/env python3
"""把已有 eval 记录回填进各 run 的 TensorBoard（D-0029）。

训练期逐 epoch 指标原本就写 ``val/<metric>``；本脚本补的是 eval.py 产生的
测试/验证评估记录（``evals/*.json``），写成 ``eval/<split>/<benchmark>/<metric>``。
可重复运行（TensorBoard 同一 tag 重复写会生成新 step，这里固定 step=0）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_per_bin import ROOT  # noqa: E402  复用项目根定位

sys.path.insert(0, str(ROOT))
from core.tensorboard import write_benchmark_scalars, write_eval_scalars  # noqa: E402

RUNS = ROOT / "output/runs/beta-regression"
BENCH_NAMES = {
    "sha256:2c4d4beeb9461597c2cae52cfaa82f0a8947d433882b1cd3b169994205018147": "v11-d-test-v2",
    "sha256:911defbff0b5d18d3b9091320bf556f3cf9a3af9ce103fb097d4df4aa60c6ecb": "v11-b-test-v2",
    "sha256:983c48b649ce41c69a4ad382b1b18e6a557a8586dd967b5799a22421249924be": "v11-r-test-v2",
    "sha256:6a76a6986d680646d0a963cdca8105bd50f1f0930c3aa74eee01e191fead97ad": "v11-clean-test-v2",
}


def training_run_index():
    """run_id -> (run_dir, {digest: checkpoint_name})，用于把 eval 记录挂回训练 run。"""
    idx = {}
    for man in RUNS.glob("*/2*/manifest.json"):
        try:
            m = json.loads(man.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        digests = {}
        for name, digest in (m.get("artifacts") or {}).items():
            if name.startswith("checkpoint:"):
                digests[digest] = name.split(":", 1)[1]
        idx[m.get("run_id")] = (man.parent, digests)
    return idx


def main() -> int:
    n_runs = n_evals = 0
    train_idx = training_run_index()
    for eval_file in sorted(RUNS.glob("*/2*/evals/*.json")):
        run_dir = eval_file.parent.parent
        try:
            rec = json.loads(eval_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = rec.get("metrics") or {}
        if not metrics:
            continue
        split = rec.get("split", "test")
        if split == "val":
            write_eval_scalars(run_dir, metrics, prefix="eval/val/final")
            n_evals += 1
            continue
        bench = BENCH_NAMES.get(rec.get("benchmark_id", ""), "unknown")
        cohort = {"v11-b-test-v2": "B", "v11-d-test-v2": "D",
                  "v11-r-test-v2": "R", "v11-clean-test-v2": "clean"}.get(bench, bench)
        # 找到挂靠的训练 run（eval run manifest 的 config_source 记录着来源）
        parent_run = None
        ck_name = None
        try:
            em = json.loads((run_dir / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            em = {}
        src = str(em.get("config_source", ""))
        if src.startswith("training run "):
            parent_run = train_idx.get(src.split("training run ", 1)[1].strip())
        if parent_run is not None:
            ck_name = parent_run[1].get(rec.get("checkpoint_digest", ""))

        ok = write_benchmark_scalars(run_dir, metrics, cohort=cohort, checkpoint=ck_name)
        if parent_run is not None:
            ok = write_benchmark_scalars(
                parent_run[0], metrics, cohort=cohort, checkpoint=ck_name
            ) or ok
        if ok:
            n_evals += 1
    print(f"回填 {n_evals} 条 eval 记录到 TensorBoard")
    print("查看：tensorboard --logdir output/runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
