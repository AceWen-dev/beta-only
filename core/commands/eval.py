"""Thin evaluation entry point: checkpoint -> separate eval run -> EvalRecord.

By default the eval reuses the training run's ``config.resolved.json`` (found via
the checkpoint manifest). This is what keeps model/loss changes in sync: no
manual ``--set`` is needed for evaluation. ``--config`` is only a fallback when
the original run directory is unavailable.
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

from adapters.torch import TorchCheckpointCodec  # noqa: E402
from dlef.benchmark import BenchmarkSpec  # noqa: E402
from dlef.config import load_config, resolve_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.seed import seed_everything  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import build_benchmark, make_loader  # noqa: E402
from core.runtime import (  # noqa: E402
    build_policy,
    build_runner,
    parse_overrides,
    resolve_output_root,
)
from core.task import build_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="fallback config if run dir is missing")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="evaluate on val (model selection) or test (requires --benchmark)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="frozen benchmark JSON; required with --split test",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="optional overrides applied on top of the training run config",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    overrides = parse_overrides(args.overrides)
    checkpoint_payload = TorchCheckpointCodec().load(checkpoint)
    manifest = getattr(checkpoint_payload, "manifest", {}) or {}
    train_experiment = str(manifest.get("experiment", ""))
    train_run_id = str(manifest.get("run_id", ""))

    # The checkpoint lives at <run_dir>/checkpoints/{last,best}.pt; prefer the
    # run directory next to it so a custom output_root does not break evaluation.
    resolved_path = checkpoint.parent.parent / "config.resolved.json"
    if resolved_path.is_file():
        base_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        config = resolve_config(
            ProjectConfig,
            base_resolved,
            overrides=[overrides] if overrides else [],
        )
        source = f"training run {train_run_id}"
    else:
        if not args.config:
            print(
                "could not find the training run config; pass --config as fallback",
                file=sys.stderr,
            )
            return 2
        config = load_config(
            args.config,
            ProjectConfig,
            overrides=[overrides] if overrides else [],
        )
        source = args.config

    if not train_experiment:
        train_experiment = config.schema.experiment

    seed_everything(config.schema.seed)
    policy = build_policy(config.schema)
    policy.setup()
    device = policy.device

    benchmark_id = ""
    test_set_id = ""
    benchmark_name = ""
    if args.split == "test":
        if not args.benchmark:
            print("--benchmark is required with --split test", file=sys.stderr)
            return 2
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.is_file():
            print(f"benchmark not found: {benchmark_path}", file=sys.stderr)
            return 2
        eval_loader = make_loader(config.schema.data, "test", config.schema.seed)
        current = build_benchmark(config.schema, eval_loader.spec)
        try:
            stored = BenchmarkSpec.load(benchmark_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"invalid benchmark manifest: {exc}", file=sys.stderr)
            return 2
        if stored.benchmark_id != current.benchmark_id:
            print(
                "benchmark mismatch: the test set or evaluation protocol changed\n"
                f"stored:  {stored.benchmark_id}\n"
                f"current: {current.benchmark_id}\n"
                "create a new benchmark instead of reusing this one",
                file=sys.stderr,
            )
            return 2
        benchmark_id = stored.benchmark_id
        test_set_id = stored.test_set_id
        benchmark_name = stored.name
    else:
        eval_loader = make_loader(config.schema.data, "val", config.schema.seed)

    runner = build_runner(resolve_output_root(config.schema, ROOT), policy)
    task = build_task(config.schema, device=device)
    optimizer = task.configure_optimizer(task.get_model())
    policy.on_fit_start(task, optimizer, config.schema)
    runner.load_checkpoint(checkpoint, task, optimizer)

    run = runner.run_store.create(
        project=config.schema.project,
        experiment=f"{train_experiment}-eval",
        config_digest=config.digest,
        seed=config.schema.seed,
        data_versions={args.split: eval_loader.spec.version},
        tags={
            "kind": "eval",
            "split": args.split,
            **({"benchmark_id": benchmark_id} if benchmark_id else {}),
        },
        config_source=source,
    )
    run.save_json("config.resolved.json", config.resolved)
    try:
        record = runner.evaluate(
            task,
            eval_loader,
            EvalSpec(
                checkpoint=str(checkpoint),
                split=args.split,
                benchmark_id=benchmark_id,
                test_set_id=test_set_id,
            ),
            run,
        )
        run.finalize("completed")
    except BaseException as exc:  # noqa: BLE001
        run.finalize("failed", error=repr(exc))
        raise

    print(f"config: {source}")
    if benchmark_name:
        print(f"benchmark: {benchmark_name} ({benchmark_id})")
    print(f"eval run: {run.run_id}")
    print(f"metrics: {record.metrics}")
    print(f"eval record: {run.dir / 'evals' / (record.eval_id + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
