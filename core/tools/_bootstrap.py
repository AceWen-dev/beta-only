"""Resolve the core for in-repo template runs.

A copied project uses the installed ``dlef`` package. When the template is run
in place inside the DLEF repo, the core lives in an ancestor directory; add it
to ``sys.path`` so `python commands/<name>.py` works without a pip install.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_core() -> None:
    try:
        import dlef  # noqa: F401
        return
    except ImportError:
        pass

    for parent in Path(__file__).resolve().parents:
        if (parent / "dlef" / "__init__.py").is_file():
            sys.path.insert(0, str(parent))
            return
