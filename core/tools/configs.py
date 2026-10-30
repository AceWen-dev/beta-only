"""List / validate experiment configs.

This is a catalog, not a registry: it scans the documented config layout and
validates each file against the typed schema. It does not duplicate values.
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

from dlef.config import load_config  # noqa: E402
from core.config import ProjectConfig  # noqa: E402


def _paths() -> list[Path]:
    paths: list[Path] = []
    base = ROOT / "input" / "configs" / "base.yaml"
    if base.is_file():
        paths.append(base)
    experiments = ROOT / "input" / "configs" / "experiments"
    if experiments.is_dir():
        paths.extend(sorted(experiments.rglob("*.yaml")))
    return paths


def _load(path: Path):
    return load_config(str(path), ProjectConfig)


def _expected_experiment(path: Path) -> str:
    relative = path.relative_to(ROOT / "input" / "configs")
    parts = relative.parts
    if len(parts) >= 3 and parts[0] == "experiments":
        return parts[1]
    return ""


def list_configs(check: bool) -> int:
    paths = _paths()
    if not paths:
        print("no configs found")
        return 0

    rows = []
    failed = False
    for path in paths:
        relative = path.relative_to(ROOT)
        try:
            config = _load(path)
            schema = config.schema
            expected = _expected_experiment(path)
            if (
                expected
                and schema.experiment != expected
                and not schema.experiment.startswith(f"{expected}-")
            ):
                raise ValueError(
                    f"experiment field {schema.experiment!r} does not match "
                    f"directory group {expected!r}"
                )
            rows.append(
                (
                    str(relative),
                    schema.experiment,
                    schema.data.name,
                    schema.data.version,
                    schema.model.name,
                    schema.loss.name,
                    schema.benchmark.name,
                )
            )
        except Exception as exc:  # noqa: BLE001 - show the bad config, keep scanning
            failed = True
            rows.append((str(relative), "INVALID", str(exc), "", "", "", ""))

    headers = ("config", "experiment", "data", "version", "model", "loss", "benchmark")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if check and failed:
        return 1
    return 0


def show_config(path_text: str) -> int:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"config not found: {path}", file=sys.stderr)
        return 2
    config = _load(path)
    print(f"config: {path}")
    print(f"digest: {config.digest}")
    print(config.dump())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list and validate configs")
    list_parser.add_argument("--check", action="store_true")
    list_parser.set_defaults(func=lambda args: list_configs(args.check))

    show_parser = sub.add_parser("show", help="show a resolved config")
    show_parser.add_argument("config")
    show_parser.set_defaults(func=lambda args: show_config(args.config))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
