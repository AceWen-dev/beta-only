"""Create or verify a frozen benchmark manifest.

The benchmark is the unit of comparability: test membership + labels + eval
transform + metric protocol. Once created, treat the JSON as immutable; create a
new file/version instead of editing it in place.
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

from dlef.benchmark import BenchmarkSpec  # noqa: E402
from dlef.config import load_config  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data import build_benchmark, make_loader  # noqa: E402
from core.runtime import parse_overrides  # noqa: E402


def _load_config(path: str, overrides: list[str]) -> ProjectConfig:
    config = load_config(
        path,
        ProjectConfig,
        overrides=[parse_overrides(overrides)] if overrides else [],
    )
    return config.schema


def _current(config: ProjectConfig) -> BenchmarkSpec:
    test_loader = make_loader(config.data, "test", config.seed)
    return build_benchmark(config, test_loader.spec)


def create(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.overrides)
    spec = _current(config)
    out = Path(args.out)
    if out.exists() and not args.force:
        try:
            existing = BenchmarkSpec.load(out)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"existing benchmark is invalid: {exc}", file=sys.stderr)
            return 2
        if existing.benchmark_id == spec.benchmark_id:
            print(f"benchmark already exists and matches: {out}")
            return 0
        print(
            f"benchmark file already exists with a different id: {out}\n"
            f"stored:  {existing.benchmark_id}\n"
            f"current: {spec.benchmark_id}\n"
            "create a new benchmark file instead of overwriting this one",
            file=sys.stderr,
        )
        return 2
    spec.save(out)
    print(f"benchmark: {spec.name}")
    print(f"test_set_id: {spec.test_set_id}")
    print(f"benchmark_id: {spec.benchmark_id}")
    print(f"manifest: {out}")
    return 0


def verify(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.overrides)
    current = _current(config)
    try:
        stored = BenchmarkSpec.load(args.benchmark)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"invalid benchmark manifest: {exc}", file=sys.stderr)
        return 2
    if stored.benchmark_id != current.benchmark_id:
        print(
            "benchmark mismatch: the current test set or evaluation protocol "
            "differs from the frozen manifest\n"
            f"stored:  {stored.benchmark_id}\n"
            f"current: {current.benchmark_id}",
            file=sys.stderr,
        )
        return 1
    print(f"benchmark ok: {stored.benchmark_id}")
    print(f"test_set_id: {stored.test_set_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    create_parser = sub.add_parser("create", help="freeze a benchmark manifest")
    create_parser.add_argument("--config", required=True)
    create_parser.add_argument("--out", required=True)
    create_parser.add_argument("--force", action="store_true")
    create_parser.add_argument("--set", dest="overrides", action="append", default=[])
    create_parser.set_defaults(func=create)

    verify_parser = sub.add_parser("verify", help="verify against a frozen benchmark")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--benchmark", required=True)
    verify_parser.add_argument("--set", dest="overrides", action="append", default=[])
    verify_parser.set_defaults(func=verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
