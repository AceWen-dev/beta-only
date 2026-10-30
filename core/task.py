"""Composition root: wire components selected by config.

Keep this file thin. Model/loss/optimizer/scheduler/metric implementations live
in their own packages; this file only composes them into a TorchTask.
"""

from __future__ import annotations

from typing import Any

from adapters.torch import TorchTask

from .config import ProjectConfig

# Keep imports and construction in training-flow order:
# model -> loss -> optimizer -> scheduler -> metrics.
from .models import build_model
from .losses import build_loss
from .optimizers import build_optimizer
from .schedulers import build_scheduler
from .metrics import build_metrics
from .tensorboard import TensorBoardCallback


def build_task(config: ProjectConfig, device: Any = "cpu") -> TorchTask:
    model = build_model(config.model, device=device)
    loss = build_loss(config.loss)

    def optimizer_fn(model):
        return build_optimizer(config.optim, model)

    def scheduler_fn(optimizer):
        return build_scheduler(config.scheduler, optimizer, epochs=config.epochs)

    metrics = build_metrics(config.metrics)

    callbacks = []
    if config.tensorboard.enabled:
        callbacks.append(
            TensorBoardCallback(
                subdir=config.tensorboard.subdir,
                flush_secs=config.tensorboard.flush_secs,
            )
        )

    return TorchTask(
        model=model,
        loss=loss,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        metrics=metrics,
        callbacks=callbacks,
        device=device,
    )
