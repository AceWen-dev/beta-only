"""Print a run card: manifest + resolved config + benchmark + source files.

This is the fast path from a run id to "what data/model/loss/metrics did this
run use, and where is the source?". It reads only the run directory and the
project conventions; it does not guess from logs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.eval import EvalRecord  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402


def _find_run(root: Path, selector: str) -> Run:
    candidate = Path(selector)
    if candidate.exists():
        return RunStore(root).open(candidate)

    store = RunStore(root)
    for run in store.iter_runs():
        if run.run_id == selector or run.dir.name == selector:
            return run
    raise SystemExit(f"run not found: {selector} (under {root})")


def _load_config(run: Run) -> dict[str, Any]:
    if not run.config_path.is_file():
        return {}
    return json.loads(run.config_path.read_text(encoding="utf-8"))


def _eval_records(run: Run) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for path in run.eval_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(EvalRecord.from_dict(payload))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return records


def _source(relative: str) -> str:
    path = ROOT / relative
    suffix = "" if path.exists() else "  (not found)"
    return f"{relative}{suffix}"


def _print_source(label: str, relative: str) -> None:
    print(f"  {label}: {_source(relative)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default=str(Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output"))) / "runs"),
    )
    parser.add_argument("--run", required=True, help="run id or run directory")
    args = parser.parse_args()

    run = _find_run(Path(args.runs), args.run)
    config = _load_config(run)
    data = config.get("data", {})
    model = config.get("model", {})
    loss = config.get("loss", {})
    optim = config.get("optim", {})
    scheduler = config.get("scheduler", {})
    benchmark = config.get("benchmark", {})
    metrics = config.get("metrics", [])

    print(f"run: {run.run_id}")
    print(f"status: {run.manifest.status}")
    print(f"project/experiment: {run.manifest.project} / {run.manifest.experiment}")
    print(f"seed: {run.manifest.seed}")
    print(f"code_version: {run.manifest.code_version}")
    print(f"config_digest: {run.manifest.config_digest}")
    print(f"config_source: {run.manifest.config_source or run.config_path}")
    print(f"config_resolved: {run.config_path}")
    print()

    print("data:")
    for key in ("name", "version", "seed", "split", "train_transform", "eval_transform"):
        print(f"  {key}: {data.get(key, '')}")
    print(f"  manifest data_versions: {dict(run.manifest.data_versions)}")
    _print_source(
        "dataset_source",
        f"core/data/datasets/{data.get('name', '<name>')}.py",
    )
    _print_source("split_source", "core/data/splits.py")
    for key in ("train_transform", "eval_transform"):
        _print_source(
            f"{key}_source",
            f"core/data/transforms/{data.get(key, '<name>')}.py",
        )
    _print_source("benchmark_source", "core/data/benchmark.py")
    print()

    print("training:")
    for key in ("seed", "epochs", "max_steps", "log_every", "checkpoint_every"):
        print(f"  {key}: {config.get(key, '')}")
    print(f"  batch_size: {data.get('batch_size', '')}")
    for key in ("amp", "grad_accum_steps", "grad_clip"):
        print(f"  {key}: {config.get(key, '')}")
    print(f"  early_stopping: {config.get('early_stopping', {})}")
    print()

    print("components:")
    print(f"  model: {model.get('name', '')} {model}")
    _print_source(
        "model_source", f"core/models/{model.get('name', '<name>')}.py"
    )
    print(f"  loss: {loss.get('name', '')} {loss}")
    _print_source("loss_source", f"core/losses/{loss.get('name', '<name>')}.py")
    print(f"  optimizer: {optim.get('name', '')} {optim}")
    _print_source(
        "optimizer_source",
        f"core/optimizers/{optim.get('name', '<name>')}.py",
    )
    print(f"  scheduler: {scheduler.get('name', '')} {scheduler}")
    _print_source(
        "scheduler_source",
        f"core/schedulers/{scheduler.get('name', '<name>')}.py",
    )
    print(f"  metrics: {list(metrics)}")
    for name in metrics:
        _print_source(f"metric_source[{name}]", f"core/metrics/{name}.py")
    print(f"  benchmark: {benchmark}")
    print()

    records = _eval_records(run)
    print("evals:")
    if not records:
        print("  (none)")
    for record in records:
        print(
            f"  {record.eval_id}: split={record.split} "
            f"benchmark={record.benchmark_id or '-'} "
            f"test_set={record.test_set_id or '-'} "
            f"metrics={dict(record.metrics)} test_access={record.test_access}"
        )
    print()

    print("artifacts:")
    checkpoint = run.checkpoint_path
    if checkpoint.is_file():
        print(f"  checkpoint: {checkpoint}")
    else:
        print("  checkpoint: (none)")
    best_checkpoint = run.best_checkpoint_path
    if best_checkpoint.is_file():
        print(f"  best_checkpoint: {best_checkpoint}")
    if run.log_path.is_file():
        print(f"  training_log: {run.log_path}")
    else:
        print("  training_log: (none)")
    print(f"  evals_dir: {run.directory / 'evals'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
