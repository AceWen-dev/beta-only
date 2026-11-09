"""Loader assembly: torch Dataset -> TorchDataLoader with its DataSpec."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from adapters.torch import TorchDataLoader

from .samplers import MixedPairSampler, PairSampler
from .spec import DataBundle


def build_loader(
    bundle: DataBundle,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    distributed: bool = False,
    sampler: str = "shuffle",
    weights=None,
    pair_min_span_ratio: float = 1.0,
    pair_batches: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> TorchDataLoader:
    dataset = bundle.dataset
    loader_kwargs: dict = {"num_workers": num_workers}
    if pin_memory and torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    if distributed and shuffle:
        sampler_obj = DistributedSampler(dataset, shuffle=True, seed=seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler_obj, **loader_kwargs
        )
    elif sampler == "pair_mixed":
        rows = getattr(dataset, "rows", None)
        if rows is None:
            raise ValueError("data.pair_batches requires dataset rows")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=MixedPairSampler(
                rows,
                seed=seed,
                batch_size=batch_size,
                pair_batches=pair_batches,
                min_span_ratio=pair_min_span_ratio,
            ),
            **loader_kwargs,
        )
    elif sampler == "pair":
        rows = getattr(dataset, "rows", None)
        if rows is None:
            raise ValueError(
                f"data.pair_sampling requires dataset rows; "
                f"{type(dataset).__name__} does not expose .rows"
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=PairSampler(rows, seed=seed, min_span_ratio=pair_min_span_ratio),
            **loader_kwargs,
        )
    elif sampler == "case_bin" and weights is not None:
        generator = torch.Generator().manual_seed(seed)
        sampler_obj = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler_obj, **loader_kwargs
        )
    else:
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            **loader_kwargs,
        )
    return TorchDataLoader(loader, bundle.spec)
