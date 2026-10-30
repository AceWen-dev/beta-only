"""External data root.

Dataset bytes (images, audio, tables, ...) live outside this repository.
Only code and small manifests are committed. Set ``DLEF_DATA_ROOT`` to the
external directory; the project resolves logical dataset/version names below
that root.

Standard layout::

    $DLEF_DATA_ROOT/
      raw/<source>/...
      curated/<dataset>/<version>/...
      splits/<dataset>/<version>/<split>.json

A split manifest stores membership only (sample ids or relative paths), so one
copy of the data serves every split.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "DLEF_DATA_ROOT"


def data_root() -> Path:
    value = os.environ.get(ENV_VAR)
    if not value:
        raise RuntimeError(
            f"{ENV_VAR} is not set; point it at the external dataset directory"
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"{ENV_VAR} is not a directory: {root}")
    return root


def curated_dir(dataset_id: str, version: str) -> Path:
    return data_root() / "curated" / dataset_id / version


def split_dir(dataset_id: str, version: str) -> Path:
    return data_root() / "splits" / dataset_id / version


def split_manifest_path(dataset_id: str, version: str, split_name: str) -> Path:
    return split_dir(dataset_id, version) / f"{split_name}.json"
