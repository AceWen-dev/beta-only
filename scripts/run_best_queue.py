#!/usr/bin/env python3
"""Evaluate the finished runs, then train the remaining arms in order.

Order (D-0017): in-flight baseline-best (case_bin, 40ep) -> reference
ranksim-best (80ep, evals reused) -> ranksim-repro (40ep, the PolypMeasure
rank2-official reproduction) -> baseline-shuffle (sampler control) ->
backbone-partial / backbone-frozen / proj.

The reproduction arm evaluates both best.pt (raw selection) and best_ema.pt
(EMA selection, the PolypMeasure protocol); other arms evaluate best.pt only.

Evaluation is idempotent: an existing completed eval record for the same
checkpoint digest + benchmark id is reused instead of touching test again.
Every train/eval launch first waits until no other run is active (single GPU).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "output" / "runs"
REPORT_ROOT = ROOT / "output" / "reports"
BENCH_ROOT = ROOT / "input" / "benchmarks"

CURRENT_EXPERIMENT = "v11-ranksim-best"
CURRENT_RUN = "20261030-212317_09bdfb94_s20261020_17e52f"
# 已完成、只欠评估的 run：baseline-best（40ep case_bin）与 ranksim-repro
# （40ep，旧 content_box 管线，D-0018 修复前）。
IN_FLIGHT_EXPERIMENT = "v11-baseline-best"
IN_FLIGHT_RUN = "20261030-231251_24bb601b_s20261020_b91f9a"
REPRO_EXPERIMENT = "v11-ranksim-repro"
REPRO_RUN = "20261030-235746_f6bbc540_s20261020_f0e4a2"
BENCHMARKS = (("d", None), ("b", "b"), ("r", "r"), ("clean", "clean"))
BENCHMARK_SUFFIX = "-test-v2"
STALE_SECONDS = 600  # train.log 超过 10 分钟没更新的 running 视为僵死，不阻塞

TARGETS = (
    # valmerge 的 seed 复核（D 2-3=1.85 突破是否稳定；D-0020）。
    ("v11-baseline-valmerge", "input/configs/experiments/v11/baseline-valmerge.yaml", ("baseline", "valmerge"), {"seed": 20261021}),
    ("v11-baseline-valmerge", "input/configs/experiments/v11/baseline-valmerge.yaml", ("baseline", "valmerge"), {"seed": 20261022}),
)
# 每个实验要评估的 checkpoint；默认只评 best.pt（raw 选模）。
EVAL_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "v11-ranksim-align": ("best", "best_ema"),
}

STAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
LOG_PATH = REPORT_ROOT / f"best_queue_{STAMP}.log"
SUMMARY_PATH = REPORT_ROOT / f"best_queue_{STAMP}.json"


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def all_runs():
    from dlef.run import RunStore

    return RunStore(RUNS_ROOT).iter_runs()


def run_by_id(run_id: str):
    return next((run for run in all_runs() if run.run_id == run_id), None)


def ids_for(experiment: str) -> set[str]:
    return {run.run_id for run in all_runs() if run.manifest.experiment == experiment}


def new_run(experiment: str, before: set[str]):
    created = [
        run for run in all_runs()
        if run.manifest.experiment == experiment and run.run_id not in before
    ]
    if len(created) != 1:
        raise RuntimeError(
            f"expected one new run for {experiment!r}, got {[run.run_id for run in created]}"
        )
    run = created[0]
    if run.manifest.status != "completed":
        raise RuntimeError(
            f"run {run.run_id} ended with status={run.manifest.status!r}: "
            f"{run.manifest.error}"
        )
    return run


def read_eval(run) -> dict:
    paths = run.eval_paths()
    if not paths:
        raise RuntimeError(f"run {run.run_id} has no eval record")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_eval(
    eval_experiment: str, benchmark_id: str, checkpoint_digest: str
) -> tuple[dict, str] | None:
    """Return (record, run_id) for a completed eval of this exact checkpoint."""
    for run in all_runs():
        if run.manifest.experiment != eval_experiment:
            continue
        if run.manifest.status != "completed":
            continue
        for path in run.eval_paths():
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("benchmark_id") == benchmark_id
                and record.get("checkpoint_digest") == checkpoint_digest
            ):
                return record, run.run_id
    return None


def run_command(command: list[str], log_path: Path) -> None:
    log(f"$ {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=None,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        )
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(command)}\n{tail}"
        )


def wait_for_current() -> object:
    log(f"waiting for current run {CURRENT_RUN} ...")
    while True:
        run = run_by_id(CURRENT_RUN)
        if run is None:
            raise RuntimeError(f"current run not found: {CURRENT_RUN}")
        status = run.manifest.status
        if status == "completed":
            log(f"current run completed: {run.run_id}")
            return run
        if status in {"failed", "killed"}:
            raise RuntimeError(
                f"current run ended with status={status!r}: {run.manifest.error}"
            )
        time.sleep(30)


def wait_for_run(run_id: str) -> object:
    log(f"waiting for run {run_id} ...")
    while True:
        run = run_by_id(run_id)
        if run is None:
            raise RuntimeError(f"run not found: {run_id}")
        status = run.manifest.status
        if status == "completed":
            log(f"run completed: {run.run_id}")
            return run
        if status in {"failed", "killed"}:
            raise RuntimeError(
                f"run {run_id} ended with status={status!r}: {run.manifest.error}"
            )
        time.sleep(30)


def _run_is_live(run) -> bool:
    """manifest 还是 running，且 train.log 近期有更新（排除僵死 manifest）。"""
    log_path = run.directory / "logs" / "train.log"
    if not log_path.exists():
        return True
    return (time.time() - log_path.stat().st_mtime) < STALE_SECONDS


def wait_for_idle() -> None:
    """单卡：任何活跃 run（训练/评估）没结束前不启动下一段 GPU 工作。"""
    while True:
        running = [
            run.run_id
            for run in all_runs()
            if run.manifest.status == "running" and _run_is_live(run)
        ]
        if not running:
            return
        log(f"waiting for active runs: {running}")
        time.sleep(30)


def checkpoint_path(run, name: str) -> Path | None:
    candidates = sorted(
        (
            path
            for path in (run.directory / "checkpoints").glob(f"{name}.*")
            if path.stem == name
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def evaluate(experiment: str, best: Path, tag: str = "") -> dict:
    results = {}
    checkpoint_digest = sha256_file(best)
    for suffix, test_set in BENCHMARKS:
        eval_experiment = f"{experiment}-eval"
        benchmark_path = BENCH_ROOT / f"v11-{suffix}{BENCHMARK_SUFFIX}.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        reused = existing_eval(
            eval_experiment, benchmark["benchmark_id"], checkpoint_digest
        )
        if reused is not None:
            record, eval_run_id = reused
            log(
                f"{experiment}/{suffix}: reuse eval {record['eval_id']} "
                "(same checkpoint + benchmark)"
            )
            results[suffix] = {
                "eval_run_id": eval_run_id,
                "eval_id": record["eval_id"],
                "benchmark_id": record["benchmark_id"],
                "metrics": record["metrics"],
                "reused": True,
            }
            continue
        before = ids_for(eval_experiment)
        command = [
            sys.executable,
            "core/commands/eval.py",
            "--checkpoint",
            str(best),
            "--split",
            "test",
            "--benchmark",
            str(BENCH_ROOT / f"v11-{suffix}{BENCHMARK_SUFFIX}.json"),
        ]
        if test_set is not None:
            command.extend(["--set", f"data.test_set={test_set}"])
        run_command(
            command,
            REPORT_ROOT / f"best_{STAMP}_{experiment}_{suffix}{tag}_eval.log",
        )
        eval_run = new_run(eval_experiment, before)
        record = read_eval(eval_run)
        results[suffix] = {
            "eval_run_id": eval_run.run_id,
            "eval_id": record["eval_id"],
            "benchmark_id": record["benchmark_id"],
            "metrics": record["metrics"],
        }
        log(
            f"{experiment}/{suffix}: worst_bin="
            f"{record['metrics'].get('v11/worst_bin'):.4f}"
        )
    return results


def main() -> int:
    results = []

    # 1) 已完成但未评估/待复用评估的 run
    for experiment, run_id, checkpoints in (
        (IN_FLIGHT_EXPERIMENT, IN_FLIGHT_RUN, ("best",)),
        (CURRENT_EXPERIMENT, CURRENT_RUN, ("best",)),
        (REPRO_EXPERIMENT, REPRO_RUN, ("best", "best_ema")),
    ):
        run = wait_for_run(run_id)
        results.extend(eval_run_checkpoints(experiment, run, checkpoints))

    # 3) 其余臂：训练 + 评估
    for experiment, config, tags, overrides in TARGETS:
        wait_for_idle()
        before = ids_for(experiment)
        command = [sys.executable, "core/commands/train.py", "--config", config]
        for tag in tags:
            command.extend(["--tag", tag])
        for key, value in overrides.items():
            command.extend(["--set", f"{key}={value}"])
        run_command(
            command,
            REPORT_ROOT / f"best_{STAMP}_{experiment}_train.log",
        )
        run = new_run(experiment, before)
        log(f"{experiment}: run={run.run_id}")
        results.extend(
            eval_run_checkpoints(experiment, run, EVAL_CHECKPOINTS.get(experiment, ("best",)))
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "log": str(LOG_PATH),
        "runs": results,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"summary: {SUMMARY_PATH}")
    return 0


def eval_run_checkpoints(experiment: str, run, names: tuple[str, ...]) -> list[dict]:
    """评估一个 run 的指定 checkpoint，返回 results 列表条目。"""
    wait_for_idle()
    entries = []
    for name in names:
        path = checkpoint_path(run, name)
        if path is None or not path.is_file():
            if name == "best":
                raise RuntimeError(f"best checkpoint missing for {run.run_id}")
            log(f"{experiment}/{run.run_id}: no {name}.pt, skip")
            continue
        log(f"evaluating {experiment} {run.run_id} with {path.name}")
        tests = evaluate(experiment, path, tag=f"_{name}" if name != "best" else "")
        entries.append(
            {
                "experiment": experiment,
                "run_id": run.run_id,
                "checkpoint": path.name,
                "tests": tests,
            }
        )
    return entries


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            log(f"FAILED: {exc!r}")
        raise
    except BaseException as exc:
        log(f"FAILED: {exc!r}")
        raise
    raise SystemExit(code)
