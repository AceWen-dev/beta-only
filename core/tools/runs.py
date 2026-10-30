"""List runs with their config source and latest eval metric.

RunStore is the source of truth; this command is only a readable index. It
never edits run data.

Views:
  summary  one row per project/experiment (default)
  runs     one row per run (default when --experiment is set)

Examples:
  python core/tools/runs.py
  python core/tools/runs.py --experiment baseline
  python core/tools/runs.py --all --limit 100
  python core/tools/runs.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.tools._bootstrap import ensure_core  # noqa: E402

ensure_core()

from dlef.eval import EvalRecord  # noqa: E402
from dlef.run import Run, RunStore  # noqa: E402

_HIDDEN_TAGS = {"framework"}
_DEFAULT_OUTPUT = Path(os.environ.get("DLEF_OUTPUT_ROOT", str(ROOT / "output")))


def _evals(run: Run) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for path in run.eval_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(EvalRecord.from_dict(payload))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return records


def _latest_eval(records: list[EvalRecord]) -> EvalRecord | None:
    if not records:
        return None
    return max(records, key=lambda record: record.created_at)


def _benchmarks(evals: list[EvalRecord]) -> str:
    ids = sorted({record.benchmark_id for record in evals if record.benchmark_id})
    return ", ".join(value[:12] for value in ids) if ids else "-"


def _metric(record: EvalRecord | None, key: str) -> float | None:
    if record is None:
        return None
    value = record.metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def _short(value: str | None, width: int) -> str:
    text = str(value or "-")
    return text if len(text) <= width else text[: width - 1] + "…"


def _started(run: Run) -> str:
    return run.manifest.started_at or ""


def _timestamp(value: str) -> str:
    """Render an ISO timestamp in local time, matching run_id / train.log."""
    if not value:
        return "-"
    try:
        moment = datetime.fromisoformat(value)
        if moment.tzinfo is not None:
            moment = moment.astimezone()
        return moment.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value[:19].replace("T", " ")


def _tags(run: Run) -> str:
    names = sorted(tag for tag in run.manifest.tags if tag not in _HIDDEN_TAGS)
    return ",".join(names) if names else "-"


def _matches(run: Run, args: argparse.Namespace) -> bool:
    manifest = run.manifest
    if args.project and manifest.project != args.project:
        return False
    if args.experiment and manifest.experiment != args.experiment:
        return False
    if args.status and manifest.status != args.status:
        return False
    if args.config and args.config not in (manifest.config_source or ""):
        return False
    if args.tag and args.tag not in manifest.tags:
        return False
    if args.since and (manifest.started_at or "") < args.since:
        return False
    if args.benchmark:
        if not any(
            (record.benchmark_id or "").startswith(args.benchmark)
            for record in _evals(run)
        ):
            return False
    return True


def _summary_rows(runs: list[Run], args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[(run.manifest.project, run.manifest.experiment)].append(run)

    rows: list[dict[str, Any]] = []
    for (project, experiment), items in groups.items():
        items = sorted(items, key=_started)
        latest = items[-1]
        statuses = Counter(run.manifest.status for run in items)
        last_eval = _latest_eval(_evals(latest))
        best: float | None = None
        for run in items:
            value = _metric(_latest_eval(_evals(run)), args.metric_key)
            if value is None:
                continue
            if best is None or (
                value < best if args.metric_mode == "min" else value > best
            ):
                best = value
        tags = sorted(
            {
                tag
                for run in items
                for tag in run.manifest.tags
                if tag not in _HIDDEN_TAGS
            }
        )
        rows.append(
            {
                "project": project,
                "experiment": experiment,
                "runs": len(items),
                "latest": _timestamp(_started(latest)),
                "status": " ".join(
                    f"{name}={count}" for name, count in sorted(statuses.items())
                ),
                "last_metric": _metric(last_eval, args.metric_key),
                "best_metric": best,
                "tags": ",".join(tags) if tags else "-",
                "last_config": _short(latest.manifest.config_source, 48),
            }
        )
    rows.sort(key=lambda row: row["latest"], reverse=True)
    return rows


def _run_rows(runs: list[Run], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(runs, key=_started, reverse=True):
        evals = _evals(run)
        latest = _latest_eval(evals)
        rows.append(
            {
                "run_id": run.run_id,
                "status": run.manifest.status,
                "project": run.manifest.project,
                "experiment": run.manifest.experiment,
                "seed": run.manifest.seed,
                "started": _timestamp(_started(run)),
                "benchmark": _benchmarks(evals),
                "split": latest.split if latest else "-",
                "metric": _metric(latest, args.metric_key),
                "tags": _tags(run),
                "config": _short(run.manifest.config_source, 48),
            }
        )
    return rows


def _print_table(
    headers: tuple[str, ...],
    keys: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells: list[str] = []
        for key in keys:
            value = row.get(key, "")
            if key in {"last_metric", "best_metric", "metric"}:
                cells.append(_format_metric(value if value is not None else None))
            else:
                cells.append(str(value))
        print("| " + " | ".join(cells) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(_DEFAULT_OUTPUT / "runs"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--config", default=None, help="substring of config_source")
    parser.add_argument("--status", default=None)
    parser.add_argument("--tag", default=None, help="only runs carrying this tag")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="only runs with an eval benchmark id starting with this value",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="only runs started at/after this ISO date (e.g. 2026-10-29)",
    )
    parser.add_argument("--metric-key", default="mae")
    parser.add_argument("--metric-mode", choices=("min", "max"), default="min")
    parser.add_argument(
        "--view",
        choices=("summary", "runs"),
        default=None,
        help="summary per experiment (default), or one row per run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="rows to show; 0 = no limit (default: 20, or all with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="show all runs (equivalent to --view runs --limit 0)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    view = args.view or ("runs" if (args.experiment or args.all) else "summary")
    limit = args.limit if args.limit is not None else (0 if args.all else 20)

    store = RunStore(Path(args.runs))
    runs = [run for run in store.iter_runs() if _matches(run, args)]

    rows = _summary_rows(runs, args) if view == "summary" else _run_rows(runs, args)
    total = len(rows)
    if limit:
        rows = rows[:limit]

    if args.json:
        print(
            json.dumps(
                {
                    "view": view,
                    "metric_key": args.metric_key,
                    "metric_mode": args.metric_mode,
                    "total": total,
                    "shown": len(rows),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if view == "summary":
        _print_table(
            (
                "project",
                "experiment",
                "runs",
                "latest",
                "status",
                f"last {args.metric_key}",
                f"best {args.metric_key}",
                "tags",
                "last config",
            ),
            (
                "project",
                "experiment",
                "runs",
                "latest",
                "status",
                "last_metric",
                "best_metric",
                "tags",
                "last_config",
            ),
            rows,
        )
    else:
        _print_table(
            (
                "run_id",
                "status",
                "project",
                "experiment",
                "seed",
                "started",
                "benchmark",
                "split",
                args.metric_key,
                "tags",
                "config",
            ),
            (
                "run_id",
                "status",
                "project",
                "experiment",
                "seed",
                "started",
                "benchmark",
                "split",
                "metric",
                "tags",
                "config",
            ),
            rows,
        )

    if not rows:
        print("(no runs)")
    elif total > len(rows):
        print(f"\n(showing {len(rows)} of {total}; use --limit 0 for all)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
