"""TensorBoard callback: an optional mirror, not the experiment record.

RunStore / ``metrics.jsonl`` / ``logs/train.log`` remain the source of truth.
This callback only writes scalars and the resolved config under
``<run_dir>/<tensorboard.subdir>/`` so TensorBoard can plot them.

If TensorBoard is not installed the callback prints one warning and becomes a
no-op; training continues. Set ``tensorboard.enabled: false`` to skip it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dlef.contracts import CallbackBase, RunContext, StepOutput, to_float


class TensorBoardCallback(CallbackBase):
    def __init__(self, *, subdir: str = "tensorboard", flush_secs: int = 10) -> None:
        self.subdir = subdir
        self.flush_secs = flush_secs
        self.writer: Any | None = None

    def on_run_start(self, ctx: RunContext) -> None:
        if not ctx.run_dir:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print(
                "warning: tensorboard is enabled but not installed; "
                "run `pip install tensorboard` or set tensorboard.enabled=false",
                file=sys.stderr,
            )
            return

        log_dir = Path(ctx.run_dir) / self.subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=self.flush_secs)
        self.writer.add_text(
            "run/config",
            json.dumps(ctx.config.to_dict(), indent=2, ensure_ascii=False, default=str),
            0,
        )
        if ctx.manifest is not None:
            self.writer.add_text(
                "run/manifest",
                json.dumps(
                    ctx.manifest.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                0,
            )

    def on_step_end(self, ctx: RunContext, step: StepOutput) -> None:
        if self.writer is None:
            return
        if step.loss is not None:
            self.writer.add_scalar("train/loss", to_float(step.loss), ctx.step)
        for name, value in step.terms.items():
            self.writer.add_scalar(f"train/{name}", to_float(value), ctx.step)

    def on_epoch_end(self, ctx: RunContext) -> None:
        if self.writer is None:
            return
        for name, value in ctx.metrics.items():
            try:
                self.writer.add_scalar(f"val/{name}", float(value), ctx.epoch)
            except (TypeError, ValueError):
                continue

    def on_run_end(self, ctx: RunContext) -> None:
        self._close()

    def on_exception(self, ctx: RunContext, error: BaseException) -> None:
        self._close()

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.flush()
        self.writer.close()
        self.writer = None
