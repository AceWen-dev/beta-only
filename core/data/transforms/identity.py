from __future__ import annotations

from .base import Transform


def build_identity() -> Transform:
    def identity(sample):
        return sample

    return Transform(name="identity", fn=identity)


def build_identity_448() -> Transform:
    """No-op transform whose name pins the 448 evaluation protocol (D-0080).

    The actual resolution is handled by ``data.input_size``; a distinct
    transform name makes the 448 benchmark_id differ from the 224 one.
    """

    def identity(sample):
        return sample

    return Transform(name="identity448", fn=identity)
