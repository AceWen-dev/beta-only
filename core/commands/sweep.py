"""Thin sweep entry point: grid over variants, one subprocess per trial.

The params use dotted keys; ``set_dotted`` writes them into a trial config.
This is how you change model/loss across experiments without editing old runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import load_config  # noqa: E402
from dlef.sweep import (  # noqa: E402
    Choice,
    SubprocessSweep,
    SweepSpec,
    SweepStore,
    TrialCommand,
)
from core.config import ProjectConfig  # noqa: E402
from core.runtime import resolve_output_root  # noqa: E402


def set_dotted(payload: dict, key: str, value) -> None:
    parts = key.split(".")
    cursor = payload
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids")
    parser.add_argument("--max-parallel", type=int, default=1)
    args = parser.parse_args()

    base = load_config(args.config, ProjectConfig)
    base_raw = json.loads(json.dumps(base.raw))
    gpu_ids = tuple(int(item) for item in args.gpus.split(",") if item)

    spec = SweepSpec(
        name="lr-accum",
        params={
            "optim.lr": Choice((0.02, 0.005)),
            "grad_accum_steps": Choice((1, 2)),
            # To sweep architecture/loss instead, replace the params above with:
            # "model.name": Choice(("mlp", "linear")),
            # "loss.name": Choice(("mse", "huber")),
            # "seed": Choice((1, 2, 3)),
        },
        method="grid",
        max_trials=4,
        seed=base.schema.seed,
    )
    outputs = resolve_output_root(base.schema, ROOT)
    store = SweepStore(outputs / "sweeps")

    def command_fn(trial, context):
        trial_raw = json.loads(json.dumps(base_raw))
        for key, value in trial.params.items():
            set_dotted(trial_raw, key, value)
        trial_raw["experiment"] = f"{base.schema.experiment}-t{trial.index:02d}"
        config_dir = outputs / "sweeps" / context.sweep_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{trial.trial_id}.json"
        config_path.write_text(json.dumps(trial_raw, indent=2), encoding="utf-8")
        return TrialCommand(
            argv=[
                sys.executable,
                str(ROOT / "core" / "commands" / "train.py"),
                "--config",
                str(config_path),
            ],
            cwd=str(ROOT),
        )

    record = SubprocessSweep(
        store,
        gpu_ids=gpu_ids,
        max_parallel=args.max_parallel,
        gpus_per_trial=1,
    ).run(spec, command_fn)

    trials = {item["trial_id"]: item for item in record.trials}
    print(f"sweep: {record.sweep_id} status={record.status}")
    print()
    print("| trial | optim.lr | accum | status | mae | run | log |")
    print("|---|---|---|---|---|---|---|")
    for result in record.results:
        trial = trials[result["trial_id"]]
        params = trial["params"]
        mae = result.get("metrics", {}).get("mae")
        print(
            f"| {result['trial_id']} | {params.get('optim.lr')} | {params.get('grad_accum_steps')} "
            f"| {result['status']} | {mae} | {result.get('run_id')} | {result.get('log_path')} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
