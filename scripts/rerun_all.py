#!/usr/bin/env python3
"""Rerun the v11 baseline and backbone variants, then remove old runs.

This is a project-specific orchestration script. It assumes the project virtual
environment is active, ``DLEF_DATA_ROOT`` is set, and the git worktree is clean
so every new run records a real ``code_version``.

Sequence:
1. snapshot existing runs for the target experiments;
2. train each config in order (baseline first);
3. evaluate each new run on all four frozen benchmarks using ``best.pt``;
4. only if every train/eval succeeds, delete the snapshotted old runs and GC
   their now-unreferenced artifacts;
5. write a JSON summary under ``output/reports/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dlef.retention import GcPlan, RetentionPolicy, RetentionService, RunAction
from dlef.run import RunStore

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "output" / "runs"
REPORT_ROOT = ROOT / "output" / "reports"
BENCHMARK_ROOT = ROOT / "input" / "benchmarks"

TARGETS = (
    ("v11-baseline", "input/configs/experiments/v11/baseline.yaml", "baseline"),
    ("v11-backbone-full", "input/configs/experiments/v11/backbone-full.yaml", "backbone-full"),
    ("v11-backbone-partial", "input/configs/experiments/v11/backbone-partial.yaml", "backbone-partial"),
    ("v11-backbone-frozen", "input/configs/experiments/v11/backbone-frozen.yaml", "backbone-frozen"),
)

# (benchmark suffix, data.test_set override or None for the default D cohort)
BENCHMARKS = (
    ("d", None),
    ("b", "b"),
    ("r", "r"),
    ("clean", "clean"),
)

STAMP = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
LOG_PATH = REPORT_ROOT / f"rerun_{STAMP}.log"
SUMMARY_PATH = REPORT_ROOT / f"rerun_{STAMP}.json"


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_command(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"$ {' '.join(args)}")
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        )
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{tail}")


def all_runs() -> list:
    return RunStore(RUNS_ROOT).iter_runs()


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
            f"run {run.run_id} for {experiment!r} ended with "
            f"status={run.manifest.status!r}: {run.manifest.error}"
        )
    return run


def read_eval(run) -> dict:
    paths = run.eval_paths()
    if not paths:
        raise RuntimeError(f"run {run.run_id} has no eval record")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def main() -> int:
    if not os.environ.get("DLEF_DATA_ROOT"):
        raise SystemExit("DLEF_DATA_ROOT is not set")

    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if git_status:
        raise SystemExit(
            "git worktree is dirty; commit the project before official reruns:\n"
            + git_status
        )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    store = RunStore(RUNS_ROOT)
    running = [run.run_id for run in store.iter_runs() if run.manifest.status == "running"]
    if running:
        raise SystemExit(f"refusing to rerun while runs are still running: {running}")

    target_names = {name for name, _, _ in TARGETS}
    target_names |= {f"{name}-eval" for name in target_names}
    old_ids = {
        run.run_id
        for run in store.iter_runs()
        if run.manifest.experiment in target_names
    }

    log(f"project commit: {commit}")
    log(f"old runs to delete after successful rerun: {len(old_ids)}")
    log(f"log: {LOG_PATH}")

    results = []
    for experiment, config, tag in TARGETS:
        before = ids_for(experiment)
        train_log = REPORT_ROOT / f"rerun_{STAMP}_{experiment}.log"
        run_command(
            [
                sys.executable,
                "core/commands/train.py",
                "--config",
                config,
                "--tag",
                tag,
            ],
            train_log,
        )
        run = new_run(experiment, before)
        best = run.best_checkpoint_path
        if not best.is_file():
            raise RuntimeError(f"best checkpoint missing for {run.run_id}: {best}")

        val_record = read_eval(run)
        log(f"{experiment}: run={run.run_id} best={best.name}")

        tests = {}
        for suffix, test_set in BENCHMARKS:
            eval_experiment = f"{experiment}-eval"
            before_eval = ids_for(eval_experiment)
            args = [
                sys.executable,
                "core/commands/eval.py",
                "--checkpoint",
                str(best),
                "--split",
                "test",
                "--benchmark",
                str(BENCHMARK_ROOT / f"v11-{suffix}-test-v1.json"),
            ]
            if test_set is not None:
                args.extend(["--set", f"data.test_set={test_set}"])
            run_command(
                args,
                REPORT_ROOT / f"rerun_{STAMP}_{experiment}_{suffix}_eval.log",
            )
            eval_run = new_run(eval_experiment, before_eval)
            record = read_eval(eval_run)
            tests[suffix] = {
                "eval_run_id": eval_run.run_id,
                "eval_id": record["eval_id"],
                "benchmark_id": record["benchmark_id"],
                "test_access": record["test_access"],
                "metrics": record["metrics"],
            }
            log(
                f"{experiment}/{suffix}: worst_bin="
                f"{record['metrics'].get('v11/worst_bin'):.4f}"
            )

        results.append(
            {
                "experiment": experiment,
                "run_id": run.run_id,
                "best_checkpoint": str(best),
                "val_eval": {
                    "eval_id": val_record["eval_id"],
                    "metrics": val_record["metrics"],
                },
                "tests": tests,
            }
        )

    # All trainings/evals succeeded: remove only the snapshotted old runs.
    actions = []
    for run in store.iter_runs():
        selected = run.run_id in old_ids
        actions.append(
            RunAction(
                run_id=run.run_id,
                project=run.manifest.project,
                experiment=run.manifest.experiment,
                action="delete" if selected else "keep",
                reason="old run replaced by rerun" if selected else "new rerun",
                size_bytes=run.size_bytes(),
                path=str(run.dir),
                started_at=run.manifest.started_at,
            )
        )
    deletion = RetentionService(store).apply(GcPlan(runs=tuple(actions)), dry_run=False)
    log(f"deleted old runs: {len(deletion.deleted_runs)}")

    artifact_plan = RetentionService(store).plan(
        RetentionPolicy(
            keep_last=1000,
            keep_best=0,
            keep_every=0,
            metric_key="",
            metric_mode="min",
            keep_tags=frozenset({"reported", "baseline", "debug"}),
            protect_status=frozenset({"running"}),
            min_age_days=0,
            artifact_ttl_days=0,
            protect_sweep_runs=False,
        )
    )
    if artifact_plan.delete_runs:
        raise RuntimeError(
            "artifact GC would delete runs: "
            + ", ".join(action.run_id for action in artifact_plan.delete_runs)
        )
    artifact_result = RetentionService(store).apply(artifact_plan, dry_run=False)
    log(f"deleted unreferenced artifacts: {len(artifact_result.deleted_artifacts)}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_commit": commit,
        "log": str(LOG_PATH),
        "deleted_old_run_ids": list(deletion.deleted_runs),
        "runs": results,
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        log(f"FAILED: {exc!r}")
        raise
    raise SystemExit(code)
