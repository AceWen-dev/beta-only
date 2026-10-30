"""Shared runtime wiring: policy, runner, subprocess trial result."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from adapters.torch import TorchCheckpointCodec, TorchTrainingPolicy
from dlef.run import ArtifactStore, Run, RunStore
from dlef.runner import Runner
from dlef.sweep import TrialResult

from .config import ProjectConfig


def parse_overrides(items: list[str]) -> dict[str, Any]: #允许你在命令行输入参数这样就能覆盖原来的config
    """Parse ``--set model.name=linear`` into a nested override mapping."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override must be KEY=VALUE: {item!r}")
        value = yaml.safe_load(raw)
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"invalid override key: {key!r}")
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return overrides


def build_policy(config: ProjectConfig, *, ddp: bool = False) -> TorchTrainingPolicy:
    return TorchTrainingPolicy(
        amp=config.amp, #混合精度训练
        grad_accum_steps=config.grad_accum_steps, #梯度累加
        grad_clip=config.grad_clip, #梯度裁减
        ddp=ddp,     #分布式数据并行
    )


def resolve_output_root(config: ProjectConfig, base: str | Path) -> Path:
    """Resolve ``config.output_root`` against the project root."""

    root = Path(config.output_root).expanduser()
    return root if root.is_absolute() else Path(base) / root


def build_runner(outputs: str | Path, policy: TorchTrainingPolicy) -> Runner:
    outputs = Path(outputs)
    run_store = RunStore(outputs / "runs")
    artifact_store = ArtifactStore(outputs / "artifacts")
    return Runner(
        run_store, #数据记录
        artifact_store,
        checkpoint_codec=TorchCheckpointCodec(),
        training_policy=policy,
    )


def write_trial_result(run: Run, metrics: Mapping[str, Any]) -> None:
    """If launched by SubprocessSweep, write the result file it expects."""
    path = os.environ.get("DLEF_TRIAL_RESULT")
    if not path:
        return
    gpu_ids = [
        int(item)
        for item in os.environ.get("DLEF_GPU_IDS", "").split(",")
        if item
    ]
    result = TrialResult(
        trial_id=os.environ.get("DLEF_TRIAL_ID", "unknown"),
        status="completed",
        run_id=run.run_id,
        metrics=dict(metrics),
        gpu_ids=tuple(gpu_ids),
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle)
