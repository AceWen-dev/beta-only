#!/usr/bin/env python3
"""D-0077/D-0078 过夜队列：选模口径 + span 旁路 + β-only。

顺序（配对控制）：
  1. diranchor40-sel seed 20261021（d40 s20 的第二 seed / s21 控制）
  2. spaninput seed 20261020
  3. betaonly  seed 20261020
  4. spaninput seed 20261021
  5. betaonly  seed 20261021

每个 run：40 轮 → 官方 company eval → benchmark2 出图（R 已排除 1–2mm）
→ 小档审计。失败只记录、不中断。
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
STEPS = LOG_DIR / "overnight_20261105.steps.log"
CHAIN = LOG_DIR / "overnight_20261105.chain.log"
STATUS = LOG_DIR / "overnight_20261105.status.json"
TAG = "d0077"
BENCH = "input/benchmarks/v11-company-test-v1.json"
SUITE = "input/benchmarks/benchmark2.json"

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


def step(name: str, args: list[str], timeout: int | None = None) -> bool:
    log("STEP " + name + " :: " + " ".join(str(a) for a in args))
    with STEPS.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(
            [str(a) for a in args], cwd=ROOT, env=ENV,
            stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
        )
    if proc.returncode == 0:
        log(f"OK   {name}")
        return True
    log(f"FAIL {name} (exit {proc.returncode})")
    return False


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


def run_one(
    label: str,
    config: str,
    experiment: str,
    seed: int,
) -> None:
    log(f"=== {label} (seed={seed}) ===")
    status("train", label=label, seed=seed)
    if not step(
        f"train {label}",
        [PY, "core/commands/train.py", "--config", config,
         "--set", f"seed={seed}", "--tag", TAG],
    ):
        return
    run = find_run(experiment, seed)
    if run is None:
        log(f"no completed run for {experiment} seed={seed}")
        return
    log(f"run {label} -> {run}")
    checkpoint = run / "checkpoints" / "final_ema.pt"
    status("eval", label=label, run=run.name)
    step(
        f"eval {label}",
        [PY, "core/commands/eval.py", "--checkpoint", checkpoint,
         "--split", "test", "--benchmark", BENCH],
    )
    status("chart", label=label)
    step(
        f"chart {label}",
        [PY, "scripts/plot_experiment.py", "--run", run,
         "--suite", SUITE, "--name", f"{experiment}-s{seed}"],
    )
    status("audit", label=label)
    step(
        f"audit {label}",
        [PY, "scripts/smallbin_audit.py", "--run", run,
         "--cohorts", "company_test,B,R", "--device", "cuda",
         "--out", f"output/reports/smallbin_audit_{experiment}_s{seed}"],
    )


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STEPS.write_text("", encoding="utf-8")
    log("=== overnight 20261105 start (D-0077/D-0078) ===")
    wait_gpu_idle()
    plan = [
        ("d40-sel-s21", "input/configs/experiments/v11/company-diranchor40-sel.yaml",
         "v11-company-diranchor40-sel", 20261021),
        ("spaninput-s20", "input/configs/experiments/v11/company-diranchor40-spaninput.yaml",
         "v11-company-diranchor40-spaninput", 20261020),
        ("betaonly-s20", "input/configs/experiments/v11/company-diranchor40-betaonly.yaml",
         "v11-company-diranchor40-betaonly", 20261020),
        ("spaninput-s21", "input/configs/experiments/v11/company-diranchor40-spaninput.yaml",
         "v11-company-diranchor40-spaninput", 20261021),
        ("betaonly-s21", "input/configs/experiments/v11/company-diranchor40-betaonly.yaml",
         "v11-company-diranchor40-betaonly", 20261021),
    ]
    for label, config, experiment, seed in plan:
        run_one(label, config, experiment, seed)
    status("done", plan=len(plan))
    log("=== overnight 20261105 done ===")
    log("report: output/reports/<experiment>-s<seed>_evalmode.json 与 smallbin_audit_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
