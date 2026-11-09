"""Create one experiment-variant config from the v11 baseline.

Keeps the file layout consistent:

    input/configs/experiments/<experiment>/<variant>.yaml

One config per variant; seeds are run dimensions, not new files. The generated
``experiment`` field is ``<experiment-group>-<name>`` so directory
``experiments/v11/lr.yaml`` becomes ``experiment: v11-lr``. The tool validates
the result with ``ProjectConfig`` before writing.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.config import ConfigError, resolve_config  # noqa: E402
from core.config import ProjectConfig  # noqa: E402


def _set_dotted(payload: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if any(not part for part in parts):
        raise SystemExit(f"invalid --set key: {key!r}")
    cursor = payload
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="variant file name")
    parser.add_argument("--experiment", default="v11")
    parser.add_argument(
        "--from",
        dest="source",
        default=str(
            ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
        ),
        help="base config to copy",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted override, e.g. --set optim.lr=0.005",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    name = args.name.strip()
    if name.endswith(".yaml"):
        name = name[:-5]
    experiment = args.experiment.strip()
    if not name or "/" in name or "\\" in name:
        print("--name must be a simple file name", file=sys.stderr)
        return 2
    if not experiment or "/" in experiment or "\\" in experiment:
        print("--experiment must be a simple path segment", file=sys.stderr)
        return 2

    source = Path(args.source)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"cannot read base config {source}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print(f"base config is not a mapping: {source}", file=sys.stderr)
        return 2

    data = copy.deepcopy(payload)
    data["experiment"] = f"{experiment}-{name}"
    for item in args.overrides:
        key, separator, raw = item.partition("=")
        if not separator or not key:
            print(f"invalid --set: {item!r}", file=sys.stderr)
            return 2
        _set_dotted(data, key, yaml.safe_load(raw))

    try:
        resolve_config(ProjectConfig, data)
    except (ConfigError, ValueError) as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 1

    out = (
        Path(args.out)
        if args.out
        else ROOT
        / "input"
        / "configs"
        / "experiments"
        / experiment
        / f"{name}.yaml"
    )
    if out.exists() and not args.force:
        print(f"config already exists: {out}; pass --force to overwrite", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"config: {out}")
    print(f"next: python core/commands/train.py --config {out} --tag baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
