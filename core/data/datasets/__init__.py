"""One dataset per file; register the builder in ``data/build.py``."""

from .regression import build_regression

__all__ = ["build_regression"]
