from __future__ import annotations

from .base import Transform


def build_identity() -> Transform:
    def identity(sample):
        return sample

    return Transform(name="identity", fn=identity)
