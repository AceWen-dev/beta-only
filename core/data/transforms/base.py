from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

Sample = Mapping[str, Any]
TransformFn = Callable[[Sample], Sample]


@dataclass(frozen=True)
class Transform:
    """A named sample transform.

    The name goes into ``DataSpec.transform`` and therefore into the run's
    data version; the callable does the actual work.
    """

    name: str
    fn: TransformFn
