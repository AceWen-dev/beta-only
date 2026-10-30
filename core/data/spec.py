"""Dataset bundle + immutable DataSpec construction.

For the synthetic placeholder, the recipe digest is built from the dataset
name, split strategy, role, seed and transform. A real project should replace
it with ``dataset_version + split_id + transform_id`` from its data registry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from dlef.contracts import DataSpec

from ..config import DataConfig


@dataclass(frozen=True)
class DataBundle:
    dataset: Any
    spec: DataSpec


def make_spec(
    config: DataConfig,
    role: str,
    transform_name: str,
    n: int,
    splits: Mapping[str, str],
) -> DataSpec:
    recipe = {
        "dataset": config.name,
        "dataset_version": config.version,
        "split": config.split,
        "role": role,
        "data_seed": config.seed,
        "n": n,
        "noise": config.noise,
        "transform": transform_name,
    }
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return DataSpec(
        name=config.name,
        version=f"{config.name}@{digest}",
        schema={"x": "float32[n,1]", "y": "float32[n,1]"},
        splits=dict(splits),
        transform=transform_name,
        stats={"n": n},
        provenance={"recipe": recipe, "role": role, "data_seed": config.seed},
    )
