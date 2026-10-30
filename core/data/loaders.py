"""Loader assembly: torch Dataset -> TorchDataLoader with its DataSpec."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from adapters.torch import TorchDataLoader

from .spec import DataBundle


def build_loader(
    bundle: DataBundle,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    distributed: bool = False,
) -> TorchDataLoader:
    dataset = bundle.dataset
    if distributed and shuffle:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )
    return TorchDataLoader(loader, bundle.spec)
