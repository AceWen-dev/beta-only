"""Initialization gate for a DLEF project.

Checks the stable project brief (`PROJECT.yaml`), the typed config, the frozen
benchmark, the data root, and the project memory (`STATE.md` / `DECISIONS.md`).
It is intentionally stricter than `make test`: the test suite must pass on the
synthetic template, while `make check` must fail until the project is real.

This tool is read-only. It never edits a config, run, or manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.benchmark import BenchmarkSpec  # noqa: E402
from dlef.config import ConfigError, load_config  # noqa: E402
from dlef.run import RunStore  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data.build import DATASETS, TRANSFORMS  # noqa: E402
from core.data.splits import SPLITTERS  # noqa: E402
from core.losses.build import BUILDERS as LOSSES  # noqa: E402
from core.metrics.build import BUILDERS as METRICS  # noqa: E402
from core.models.build import BUILDERS as MODELS  # noqa: E402
from core.optimizers.build import BUILDERS as OPTIMIZERS  # noqa: E402
from core.schedulers.build import BUILDERS as SCHEDULERS  # noqa: E402

PHASES = {"setup", "data", "baseline", "ablation", "benchmark", "report"}
PLACEHOLDER_VALUES = {
    "",
    "<...>",
    "YYYY-MM-DD",
    "my-project",
}
REQUIRED = (
    ("project",),
    ("question",),
    ("unit",),
    ("input",),
    ("output",),
    ("primary_metric",),
    ("success",),
    ("data", "root"),
    ("data", "name"),
    ("data", "version"),
    ("data", "sample_id"),
    ("data", "labels"),
    ("data", "units"),
    ("splits", "train"),
    ("splits", "val"),
    ("splits", "test"),
    ("splits", "group_by"),
    ("benchmark", "name"),
    ("benchmark", "label_version"),
    ("benchmark", "metric_protocol"),
)


def _get(payload: Any, path: tuple[str, ...]) -> Any:
    cursor = payload
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _placeholder(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return text in PLACEHOLDER_VALUES or (
        text.startswith("<") and text.endswith(">")
    )


def _front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", text, flags=re.DOTALL | re.MULTILINE)
    if not match:
        raise ValueError("missing YAML front matter")
    payload = yaml.safe_load(match.group(1)) or {}
    if not isinstance(payload, dict):
        raise ValueError("front matter is not a mapping")
    return payload


def _check_brief(
    brief: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    for path in REQUIRED:
        value = _get(brief, path)
        if _placeholder(value):
            errors.append(f"PROJECT.yaml: {'.'.join(path)} is missing or a placeholder")
    if not isinstance(brief.get("non_goals", []), list):
        errors.append("PROJECT.yaml: non_goals must be a list")
    data_root = _get(brief, ("data", "root"))
    if isinstance(data_root, str) and Path(data_root).is_absolute():
        errors.append("PROJECT.yaml: data.root must be relative to DLEF_DATA_ROOT")
    if _get(brief, ("splits", "test_frozen")) is not True:
        errors.append("PROJECT.yaml: splits.test_frozen must be true")
    if _placeholder(_get(brief, ("created",))):
        warnings.append("PROJECT.yaml: created is not set")
    if _placeholder(_get(brief, ("dlef_commit",))):
        warnings.append("PROJECT.yaml: dlef_commit is not set")


def _check_config(
    brief: dict[str, Any],
    config_path: Path,
    errors: list[str],
    warnings: list[str],
) -> Any:
    try:
        config = load_config(str(config_path), ProjectConfig)
    except (OSError, ConfigError, ValueError) as exc:
        errors.append(f"config: cannot load {config_path}: {exc}")
        return None
    schema = config.schema

    if schema.project != _get(brief, ("project",)):
        errors.append(
            f"config.project ({schema.project!r}) != PROJECT.yaml.project "
            f"({_get(brief, ('project',))!r})"
        )
    if schema.data.name != _get(brief, ("data", "name")):
        errors.append(
            f"config.data.name ({schema.data.name!r}) != PROJECT.yaml.data.name "
            f"({_get(brief, ('data', 'name'))!r})"
        )
    if schema.data.version != _get(brief, ("data", "version")):
        errors.append(
            f"config.data.version ({schema.data.version!r}) != "
            f"PROJECT.yaml.data.version ({_get(brief, ('data', 'version'))!r})"
        )
    if schema.benchmark.name != _get(brief, ("benchmark", "name")):
        errors.append(
            f"config.benchmark.name ({schema.benchmark.name!r}) != "
            f"PROJECT.yaml.benchmark.name "
            f"({_get(brief, ('benchmark', 'name'))!r})"
        )
    primary = _get(brief, ("primary_metric",))
    if primary not in schema.metrics:
        errors.append(f"PROJECT.yaml.primary_metric {primary!r} is not in config.metrics")
    if primary not in schema.benchmark.metric_versions:
        errors.append(
            f"PROJECT.yaml.primary_metric {primary!r} is not in "
            "config.benchmark.metric_versions"
        )
    if not schema.output_root.strip():
        errors.append("config.output_root must not be empty")

    selectors = (
        ("data.name", schema.data.name, DATASETS),
        ("data.split", schema.data.split, SPLITTERS),
        ("data.train_transform", schema.data.train_transform, TRANSFORMS),
        ("data.eval_transform", schema.data.eval_transform, TRANSFORMS),
        ("model.name", schema.model.name, MODELS),
        ("loss.name", schema.loss.name, LOSSES),
        ("optim.name", schema.optim.name, OPTIMIZERS),
        ("scheduler.name", schema.scheduler.name, SCHEDULERS),
    )
    for label, value, registry in selectors:
        if value not in registry:
            errors.append(f"{label}={value!r} is not registered")
    for name in schema.metrics:
        if name not in METRICS:
            errors.append(f"metric {name!r} is not registered")

    return config


def _check_benchmark(
    brief: dict[str, Any],
    config: Any,
    errors: list[str],
    warnings: list[str],
) -> BenchmarkSpec | None:
    name = _get(brief, ("benchmark", "name"))
    if _placeholder(name):
        return None
    path = ROOT / "input" / "benchmarks" / f"{name}.json"
    if not path.is_file():
        errors.append(f"benchmark file not found: {path}")
        return None
    try:
        spec = BenchmarkSpec.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"benchmark: cannot load {path}: {exc}")
        return None
    if spec.name != name:
        errors.append(f"benchmark name {spec.name!r} != PROJECT.yaml {name!r}")
    if config is not None:
        if spec.dataset != config.schema.data.name:
            errors.append(
                f"benchmark.dataset {spec.dataset!r} != "
                f"config.data.name {config.schema.data.name!r}"
            )
        if spec.dataset_version != config.schema.data.version:
            errors.append(
                f"benchmark.dataset_version {spec.dataset_version!r} != "
                f"config.data.version {config.schema.data.version!r}"
            )
        primary = _get(brief, ("primary_metric",))
        if primary not in spec.metrics:
            errors.append(f"benchmark.metrics does not contain {primary!r}")
    label_version = _get(brief, ("benchmark", "label_version"))
    if not _placeholder(label_version) and label_version != spec.label_version:
        errors.append(
            f"benchmark.label_version {spec.label_version!r} != "
            f"PROJECT.yaml {label_version!r}"
        )
    print(f"[ok] benchmark: {path} ({spec.benchmark_id[:20]}...)")
    return spec


def _check_data_root(
    brief: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    data_root = _get(brief, ("data", "root"))
    if _placeholder(data_root):
        return
    env = os.environ.get("DLEF_DATA_ROOT")
    if not env:
        errors.append("DLEF_DATA_ROOT is not set")
        return
    path = Path(env).expanduser() / str(data_root)
    if not path.exists():
        errors.append(f"data root does not exist: {path}")
    else:
        print(f"[ok] data root: {path}")


def _check_memory(
    brief: dict[str, Any], errors: list[str], warnings: list[str]
) -> dict[str, Any] | None:
    state_path = ROOT / "STATE.md"
    decisions_path = ROOT / "DECISIONS.md"
    front: dict[str, Any] | None = None
    if not state_path.is_file():
        errors.append("STATE.md is missing")
    else:
        try:
            front = _front_matter(state_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"STATE.md: {exc}")
        else:
            if front.get("project") != _get(brief, ("project",)):
                errors.append(
                    f"STATE.md.project ({front.get('project')!r}) != "
                    f"PROJECT.yaml.project ({_get(brief, ('project',))!r})"
                )
            if _placeholder(front.get("updated")):
                errors.append("STATE.md.updated is missing")
            if front.get("phase") not in PHASES:
                errors.append(
                    f"STATE.md.phase must be one of {sorted(PHASES)}"
                )
            if front.get("primary_metric") != _get(brief, ("primary_metric",)):
                errors.append(
                    "STATE.md.primary_metric does not match PROJECT.yaml.primary_metric"
                )
    if not decisions_path.is_file():
        errors.append("DECISIONS.md is missing")
    else:
        text = decisions_path.read_text(encoding="utf-8")
        ids = re.findall(r"^### (D-\d+)\b", text, flags=re.MULTILINE)
        if not ids:
            errors.append("DECISIONS.md has no D-xxxx entries")
        if len(ids) != len(set(ids)):
            errors.append("DECISIONS.md has duplicate decision ids")
        if "D-0001" not in ids:
            warnings.append("DECISIONS.md has no D-0001 entry")
    return front


def _check_phase(
    config: Any,
    front: dict[str, Any],
    benchmark: BenchmarkSpec | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    phase = front.get("phase")
    if phase not in PHASES or phase in {"setup", "data"}:
        return

    output_root = Path(config.schema.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    runs_root = output_root / "runs"
    runs = RunStore(runs_root).iter_runs() if runs_root.is_dir() else []
    completed = [run for run in runs if run.manifest.status == "completed"]
    baseline = [run for run in completed if "baseline" in run.manifest.tags]
    others = [
        run
        for run in completed
        if not ({"baseline", "debug", "smoke"} & set(run.manifest.tags))
    ]

    if phase in {"baseline", "ablation", "benchmark", "report"} and not baseline:
        errors.append(
            "STATE.md.phase is beyond setup/data but there is no completed run "
            "tagged 'baseline'; run train.py --tag baseline"
        )
    if phase in {"ablation", "benchmark", "report"} and not others:
        errors.append(
            f"STATE.md.phase={phase!r} requires at least one completed non-baseline, "
            "non-debug run"
        )

    best_run = front.get("best_run")
    if best_run not in {None, "", "-"} and not any(
        run.run_id == best_run for run in runs
    ):
        errors.append(f"STATE.md.best_run {best_run!r} is not in the run store")
    if phase in {"baseline", "ablation", "benchmark", "report"} and best_run in {
        None,
        "",
        "-",
    }:
        warnings.append("STATE.md.best_run is not set")

    if phase in {"benchmark", "report"}:
        if front.get("benchmark") in {None, "", "-"}:
            errors.append("STATE.md.benchmark must be set for phase benchmark/report")
        elif benchmark is not None and front["benchmark"] not in {
            benchmark.benchmark_id,
            benchmark.name,
        }:
            errors.append(
                f"STATE.md.benchmark {front['benchmark']!r} does not match "
                f"{benchmark.name!r} / {benchmark.benchmark_id}"
            )
    if phase == "report":
        claims = list((output_root / "claims").glob("*.json"))
        report = output_root / "reports" / "report.md"
        if not claims:
            errors.append("phase=report requires at least one output/claims/*.json")
        if not report.is_file():
            errors.append("phase=report requires output/reports/report.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(ROOT / "PROJECT.yaml"))
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "input" / "configs" / "experiments" / "v11" / "baseline.yaml"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    project_path = Path(args.project)
    config_path = Path(args.config)
    errors: list[str] = []
    warnings: list[str] = []

    if not project_path.is_file():
        errors.append(f"PROJECT.yaml is missing: {project_path}")
        brief: dict[str, Any] = {}
    else:
        try:
            payload = yaml.safe_load(project_path.read_text(encoding="utf-8"))
            brief = payload if isinstance(payload, dict) else {}
            if not brief:
                errors.append("PROJECT.yaml is empty or not a mapping")
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"PROJECT.yaml: {exc}")
            brief = {}

    if brief:
        _check_brief(brief, errors, warnings)
        config = _check_config(brief, config_path, errors, warnings)
        benchmark = _check_benchmark(brief, config, errors, warnings)
        _check_data_root(brief, errors, warnings)
        front = _check_memory(brief, errors, warnings)
        if config is not None and front is not None:
            _check_phase(config, front, benchmark, errors, warnings)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": not errors,
                    "project": str(project_path),
                    "errors": errors,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"CHECK {project_path}")
        for warning in warnings:
            print(f"WARN: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        print(f"RESULT: {'OK' if not errors else 'FAIL'} ({len(errors)} errors)")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
