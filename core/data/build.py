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
from .samplers import case_bin_weights
from .spec import DataBundle
from .transforms.base import Transform
from .transforms.identity import build_identity, build_identity_448
from .transforms.light import build_light

DatasetBuilder = Callable[[DataConfig, str, Transform], DataBundle]
TransformBuilder = Callable[[], Transform]

DATASETS: dict[str, DatasetBuilder] = {
    "regression": build_regression,
    "v11_frames": build_v11_frames,
}

TRANSFORMS: dict[str, TransformBuilder] = {
    "identity": build_identity,
    "identity448": build_identity_448,
    "light": build_light,
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
    weights = None
    sampler = "shuffle"
    if role == "train" and config.pair_sampling:
        sampler = "pair_mixed" if config.pair_batches > 0 else "pair"
    elif role == "train" and config.sampler == "case_bin":
        rows = getattr(bundle.dataset, "rows", None)
        if rows is None:
            raise ValueError(
                f"data.sampler='case_bin' requires dataset rows; "
                f"{type(bundle.dataset).__name__} does not expose .rows"
            )
        weights = case_bin_weights(rows)
        sampler = "case_bin"
    return build_loader(
        bundle,
        config.batch_size,
        shuffle=shuffle,
        seed=seed,
        distributed=distributed,
        sampler=sampler,
        weights=weights,
        pair_min_span_ratio=config.pair_min_span_ratio,
        pair_batches=config.pair_batches,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
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
