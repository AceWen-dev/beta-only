#!/usr/bin/env python3
"""Append steps 5/6 after the running D-0079 queue: big-scratch + 448 screen.

Waits for the predecessor queue PID, then:
  1. company-v1-big from scratch (40 ep, diranchor recipe)
  2. 448 smoke for base448 + diranchor15-448
  3. base448 (15 ep)
  4. diranchor15-448 (15 ep)

448 uses batch=4 x grad_accum=4 (effective 16, same optimizer steps as the
224 batch-16 runs); it is a resolution screen, not a 40-epoch confirmation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python")
LOG_DIR = ROOT / "output" / "logs"
STEPS = LOG_DIR / "overnight_append_20261105.steps.log"
CHAIN = LOG_DIR / "overnight_append_20261105.chain.log"
STATUS = LOG_DIR / "overnight_append_20261105.status.json"
PREDECESSOR = 3697898
TAG = "d0080"
BENCH = "input/benchmarks/v11-company-test-v1.json"
BENCH448 = "input/benchmarks/v11-company-test-v1-448.json"

ENV = dict(os.environ)
ENV.update(
    {
        "DLEF_DATA_ROOT": "/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data",
        "PYTHONPATH": "/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF",
        "DLEF_SKIP_POST_REPORT": "1",
        "PYTHONUNBUFFERED": "1",
    }
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{stamp()}] {message}"
    print(line, flush=True)
    with CHAIN.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def status(stage: str, **extra) -> None:
    payload = {"updated": stamp(), "stage": stage, **extra}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)


def wait_pid(pid: int) -> None:
    while Path(f"/proc/{pid}").exists():
        time.sleep(30)
    log(f"predecessor {pid} exited; append queue starts")


def wait_gpu_idle() -> None:
    while True:
        out = subprocess.run(
            ["pgrep", "-f", r"core/commands/(train|eval)\.py"],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            return
        log("GPU 上仍有训练/评测进程，继续等待")
        time.sleep(60)


def step(name: str, args: list[str]) -> bool:
    log("STEP " + name + " :: " + " ".join(str(a) for a in args))
    with STEPS.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(
            [str(a) for a in args], cwd=ROOT, env=ENV,
            stdout=fh, stderr=subprocess.STDOUT,
        )
    if proc.returncode == 0:
        log(f"OK   {name}")
        return True
    log(f"FAIL {name} (exit {proc.returncode})")
    return False


def find_run(experiment: str, seed: int) -> Path | None:
    base = ROOT / "output" / "runs" / "beta-regression" / experiment
    best: tuple[str, Path] | None = None
    if not base.is_dir():
        return None
    for manifest in base.glob("*/manifest.json"):
        data = json.loads(manifest.read_text())
        if data.get("status") != "completed" or data.get("seed") != seed:
            continue
        if data.get("tags", {}).get("debug") == "true":
            continue
        started = str(data.get("started_at", ""))
        if best is None or started > best[0]:
            best = (started, manifest.parent)
    return best[1] if best else None


def eval_args(run: Path, big: bool = False, bench: str = BENCH) -> list:
    args = [PY, "core/commands/eval.py", "--checkpoint",
            run / "checkpoints" / "final_ema.pt", "--split", "test",
            "--benchmark", bench]
    if big:
        # 训练在 company-v1-big；测试帧与 company-v1 完全相同，冻结 benchmark 不变。
        args += ["--set", "data.version=company-v1",
                 "--set", "data.test_set=company",
                 "--set", "benchmark.name=v11-company-test-v1"]
    return args


def chart_audit(run: Path, name: str, suite: str) -> None:
    step(f"chart {name}", [PY, "scripts/plot_experiment.py", "--run", run,
                           "--suite", suite, "--name", name])
    step(f"audit {name}", [PY, "scripts/smallbin_audit.py", "--run", run,
                           "--cohorts", "company_test,B,R", "--device", "cuda",
                           "--out", f"output/reports/smallbin_audit_{name}"])


def run_one(label: str, config: str, experiment: str, seed: int,
            big: bool = False, bench: str = BENCH,
            suite: str = "input/benchmarks/benchmark2.json") -> None:
    status("train", label=label, seed=seed)
    if not step(f"train {label}", [PY, "core/commands/train.py",
                                   "--config", config,
                                   "--set", f"seed={seed}", "--tag", TAG]):
        return
    run = find_run(experiment, seed)
    if run is None:
        log(f"no completed run for {experiment} seed={seed}")
        return
    log(f"run {label} -> {run}")
    step(f"eval {label}", eval_args(run, big=big, bench=bench))
    chart_audit(run, f"{experiment}-s{seed}", suite)


def smoke(config: str, label: str) -> bool:
    return step(f"smoke {label}", [PY, "core/commands/train.py",
                                   "--config", config, "--tag", "debug",
                                   "--set", f"experiment=v11-smoke-{label}",
                                   "--set", "max_steps=1",
                                   "--set", "data.max_samples=4",
                                   "--set", "data.num_workers=0",
                                   "--set", "data.pin_memory=false",
                                   "--set", "early_stopping.enabled=false"])


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STEPS.write_text("", encoding="utf-8")
    os.environ.update({k: v for k, v in ENV.items() if k.startswith("DLEF") or k == "PYTHONPATH"})
    status("等待前序队列", predecessor=PREDECESSOR)
    wait_pid(PREDECESSOR)
    wait_gpu_idle()

    # Step 5: big from scratch
    run_one("big-scratch", "input/configs/experiments/v11/company-big-scratch.yaml",
            "v11-company-big-scratch", 20261020, big=True,
            suite="input/benchmarks/benchmark2-big.json")

    # Step 6: 448 screen
    for name, config, experiment in (
        ("base448", "input/configs/experiments/v11/company-base448.yaml",
         "v11-company-base448"),
        ("diranchor15-448", "input/configs/experiments/v11/company-diranchor15-448.yaml",
         "v11-company-diranchor15-448"),
    ):
        if not smoke(config, name):
            log(f"skip {name}: smoke failed")
            continue
        run_one(name, config, experiment, 20261020, bench=BENCH448)

    status("done")
    log("=== overnight append done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
