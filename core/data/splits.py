"""Materialized split strategies.

A split is an explicit mapping ``role -> list[int]``. It is created once and
recorded in ``DataSpec.splits``; loaders never call ``random_split`` at runtime.

The synthetic placeholder generates each role from its own data seed, so the
test membership does not drift when ``n_train`` changes or when a run uses a
different training seed. A real project should load a materialized membership
manifest from ``$DLEF_DATA_ROOT`` instead.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

SplitIndices = dict[str, list[int]]

ROLE_SEED_OFFSET = {"train": 0, "val": 1, "test": 2}


def holdout(n_train: int, n_val: int, n_test: int, seed: int) -> SplitIndices:
    return {
        "train": list(range(n_train)),
        "val": list(range(n_val)),
        "test": list(range(n_test)),
    }


def manifest(n_train: int, n_val: int, n_test: int, seed: int) -> SplitIndices:
    """v11: membership 由 manifest 决定，这里只占位。"""
    return {"train": [], "val": [], "test": []}


SPLITTERS: dict[str, Callable[[int, int, int, int], SplitIndices]] = {
    "holdout": holdout,
    "manifest": manifest,
}


def make_split(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> SplitIndices:
    try:
        splitter = SPLITTERS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown data.split: {name!r}; available: {sorted(SPLITTERS)}"
        ) from exc
    return splitter(n_train, n_val, n_test, seed)


def role_seed(seed: int, role: str) -> int:
    try:
        return seed + ROLE_SEED_OFFSET[role]
    except KeyError as exc:
        raise ValueError(
            f"unknown data role: {role!r}; available: {sorted(ROLE_SEED_OFFSET)}"
        ) from exc


def membership_ids(
    split_name: str,
    indices: SplitIndices,
    seed: int,
    dataset_version: str,
) -> dict[str, str]:
    ids: dict[str, str] = {}
    for role, role_indices in indices.items():
        payload = {
            "split": split_name,
            "seed": seed,
            "dataset_version": dataset_version,
            "role": role,
            "indices": role_indices,
        }
        data = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()[:16]
        ids[role] = f"{split_name}@{digest}/{role}"
    return ids
