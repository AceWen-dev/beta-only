"""One-screen project status for a human or coding agent.

Reads PROJECT.yaml, STATE.md, the run index and git status. It is read-only and
does not require torch. Use `make status` at session start, `make handoff` at
session end.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.run import RunStore  # noqa: E402


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _front_matter(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", text, flags=re.DOTALL | re.MULTILINE)
    if not match:
        return {}
    payload = yaml.safe_load(match.group(1)) or {}
    return payload if isinstance(payload, dict) else {}


def _section_bullets(path: Path, heading: str) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^##[^\n]*{re.escape(heading)}[^\n]*\n(.*?)(?=^##\s+|\Z)",
        text,
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return []
    bullets = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
    return bullets


def _output_root() -> Path:
    env = os.environ.get("DLEF_OUTPUT_ROOT")
    if env:
        root = Path(env).expanduser()
        return root if root.is_absolute() else ROOT / root
    base = _read_yaml(
        ROOT / "input" / "configs" / "experiments" / "v11" / "baseline-best.yaml"
    )
    value = str(base.get("output_root", "output"))
    root = Path(value).expanduser()
    return root if root.is_absolute() else ROOT / root


def _git_summary() -> tuple[str, int]:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        return branch or "no-git", len(status)
    except OSError:
        return "no-git", 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", action="store_true")
    args = parser.parse_args()

    project = _read_yaml(ROOT / "PROJECT.yaml")
    state_path = ROOT / "STATE.md"
    front = _front_matter(state_path)
    runs_root = _output_root() / "runs"
    runs = RunStore(runs_root).iter_runs() if runs_root.is_dir() else []
    statuses = Counter(run.manifest.status for run in runs)
    experiments = Counter(run.manifest.experiment for run in runs)
    latest = max(runs, key=lambda run: run.manifest.started_at or "") if runs else None
    branch, dirty = _git_summary()

    print(f"project: {project.get('project', front.get('project', '-'))}")
    print(f"phase: {front.get('phase', '-')}")
    print(f"question: {project.get('question', '-')}")
    print(f"primary metric: {project.get('primary_metric', front.get('primary_metric', '-'))}")
    print(f"best run: {front.get('best_run', '-')}")
    print(f"benchmark: {front.get('benchmark', '-')}")
    print(
        "runs: "
        f"{len(runs)} total"
        + "".join(f" | {name}={count}" for name, count in sorted(statuses.items()))
    )
    if latest is not None:
        print(
            f"latest: {latest.run_id} ({latest.manifest.experiment}, "
            f"{latest.manifest.status})"
        )
    if experiments:
        print(
            "experiments: "
            + ", ".join(f"{name}={count}" for name, count in sorted(experiments.items()))
        )
    else:
        print("experiments: -")

    next_actions = _section_bullets(state_path, "Next actions")
    if next_actions:
        print("next actions:")
        for item in next_actions[:8]:
            print(f"  - {item}")
    else:
        print("next actions: -")

    if args.handoff:
        checklist = _section_bullets(state_path, "Handoff checklist")
        if checklist:
            print("handoff checklist:")
            for item in checklist[:10]:
                print(f"  - {item}")
        print("before handoff: make check; update STATE.md; append DECISIONS.md; commit")

    print(f"git: {branch}; {dirty} changed file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
