"""Freeze a new benchmark manifest from a config.

Wraps ``build_benchmark`` with a ``--name`` override and refuses to overwrite
an existing manifest. The manifest records test membership, label version,
eval transform and metric protocol; changing any of them is a new benchmark.

After creating it, set the same name in ``PROJECT.yaml`` and the v11 config
you use; ``make check`` verifies the three agree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import ConfigError, load_config  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import build_benchmark, make_loader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
        ),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = args.name.strip()
    if not name or "/" in name or "\\" in name:
        print("--name must be a simple file name", file=sys.stderr)
        return 2

    try:
        config = load_config(
            args.config,
            ProjectConfig,
            overrides=[{"benchmark": {"name": name}}],
        )
        loader = make_loader(config.schema.data, "test", config.schema.seed)
        spec = build_benchmark(config.schema, loader.spec)
    except (OSError, ConfigError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot build benchmark: {exc}", file=sys.stderr)
        return 1

    out = (
        Path(args.out)
        if args.out
        else ROOT / "input" / "benchmarks" / f"{name}.json"
    )
    if out.exists() and not args.force:
        print(f"benchmark already exists: {out}; pass --force to overwrite", file=sys.stderr)
        return 2
    spec.save(out)
    print(f"benchmark: {out}")
    print(f"name: {spec.name}")
    print(f"benchmark_id: {spec.benchmark_id}")
    print(f"test_set_id: {spec.test_set_id}")
    print("next: set benchmark.name in PROJECT.yaml and the training config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
