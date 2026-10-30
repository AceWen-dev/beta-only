"""Thin training entry point: config -> data/task -> Runner.fit -> eval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import load_config  # noqa: E402
from dlef.eval import EvalSpec  # noqa: E402
from dlef.seed import seed_everything  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import make_loaders  # noqa: E402
from core.runtime import (  # noqa: E402
    build_policy,
    build_runner,
    parse_overrides,
    resolve_output_root,
    write_trial_result,
)
from core.task import build_task  # noqa: E402


def _parse_tags(items: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        name = name.strip()
        if not name:
            raise SystemExit(f"invalid --tag: {item!r}")
        tags[name] = value if separator else "true"
    return tags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="typed override, e.g. --set model.name=linear --set loss.name=huber",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="attach a tag to this run, e.g. --tag baseline --tag debug; "
        "tags can protect runs from gc",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    config = load_config(
        args.config,
        ProjectConfig,
        overrides=[overrides] if overrides else [],
    )
    seed_everything(config.schema.seed)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    policy = build_policy(config.schema, ddp=ddp)
    policy.setup()
    device = policy.device

    runner = build_runner(resolve_output_root(config.schema, ROOT), policy)
    train_loader, val_loader = make_loaders(
        config.schema.data,
        config.schema.seed,
        distributed=ddp,
    )
    task = build_task(config.schema, device=device)
    run = runner.fit(
        task,
        train_loader,
        val_loader,
        config,
        tags=_parse_tags(args.tag),
    ) #训练的核心代码
    if run is None:
        print(f"rank {getattr(policy, '_rank', 0)}: training complete (non-main)")
        return 0

    best_checkpoint = run.best_checkpoint_path
    checkpoint = best_checkpoint if best_checkpoint.is_file() else run.checkpoint_path
    if checkpoint != run.checkpoint_path:
        optimizer = task.configure_optimizer(task.get_model())
        runner.load_checkpoint(checkpoint, task, optimizer)

    record = runner.evaluate(
        task,
        val_loader,
        EvalSpec(checkpoint=str(checkpoint), split="val"),
        run,
    )
    write_trial_result(run, record.metrics)
    print(f"run: {run.run_id}")
    print(f"metrics: {record.metrics}")
    print(f"manifest: {run.manifest_path}")
    print(f"checkpoint: {run.checkpoint_path}")
    if best_checkpoint.is_file():
        print(f"best checkpoint: {best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
