"""Data composition root.

``data.name`` selects a dataset builder, ``data.split`` selects a split
strategy, and ``data.train_transform`` / ``data.eval_transform`` select the
transforms. The actual implementations live in ``datasets/`` and
``transforms/``; this file only wires them into train/val loaders.
"""

from __future__ import annotations

from typing import Callable

from ..config import DataConfig
from .datasets.regression import build_regression
from .datasets.v11_frames import build_v11_frames
from .loaders import build_loader
from .spec import DataBundle
from .transforms.base import Transform
from .transforms.identity import build_identity

DatasetBuilder = Callable[[DataConfig, str, Transform], DataBundle]
TransformBuilder = Callable[[], Transform]

DATASETS: dict[str, DatasetBuilder] = {
    "regression": build_regression,
    "v11_frames": build_v11_frames,
}

TRANSFORMS: dict[str, TransformBuilder] = {
    "identity": build_identity,
}


def build_transform(name: str) -> Transform:
    try:
        builder = TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown transform: {name!r}; available: {sorted(TRANSFORMS)}"
        ) from exc
    return builder()


def build_dataset(
    config: DataConfig,
    role: str,
    transform_name: str,
) -> DataBundle:
    try:
        builder = DATASETS[config.name]
    except KeyError as exc:
        raise ValueError(
            f"unknown data.name: {config.name!r}; available: {sorted(DATASETS)}"
        ) from exc

    transform = build_transform(transform_name)
    return builder(config, role, transform)


def make_loader(
    config: DataConfig,
    role: str,
    seed: int,
    *,
    transform_name: str | None = None,
    shuffle: bool | None = None,
    distributed: bool = False,
):
    if transform_name is None:
        transform_name = (
            config.train_transform if role == "train" else config.eval_transform
        )
    if shuffle is None:
        shuffle = role == "train"

    bundle = build_dataset(config, role, transform_name)
    return build_loader(
        bundle,
        config.batch_size,
        shuffle=shuffle,
        seed=seed,
        distributed=distributed,
    )


def make_loaders(
    config: DataConfig,
    seed: int,
    *,
    distributed: bool = False,
):
    train_loader = make_loader(config, "train", seed, distributed=distributed)
    val_loader = make_loader(config, "val", seed)
    return train_loader, val_loader
