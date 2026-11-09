#!/usr/bin/env python3
"""D-0083/D-0084 conditional chain after the 448 screen.

Rule from owner:
- if beta-only@448 beats beta-only@224 -> STOP and report; do not start the two.
- otherwise: run beta-only + EndoStreamDepth, then pure-image regression.
- after both: if beta-only still wins and Endo vs DA is close, select Endo beta-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path("/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python")
LOG_DIR = ROOT / "output" / "logs"
STEPS = LOG_DIR / "conditional_20261106.steps.log"
CHAIN = LOG_DIR / "conditional_20261106.chain.log"
STATUS = LOG_DIR / "conditional_20261106.status.json"
PREDECESSOR = 3705367
TAG = "d0083"
BENCH_DA = "input/benchmarks/v11-company-test-v1.json"
BENCH_ENDO = "input/benchmarks/v11-company-test-v1-endo.json"
SUITE_DA = "input/benchmarks/benchmark2.json"
SUITE_ENDO = "input/benchmarks/benchmark2-endo.json"

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
    log(f"predecessor {pid} exited")


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


def run_one(label: str, config: str, experiment: str, seed: int,
            bench: str, suite: str) -> None:
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
    status("eval", label=label, run=run.name)
    step(f"eval {label}", [PY, "core/commands/eval.py",
                           "--checkpoint", run / "checkpoints/final_ema.pt",
                           "--split", "test", "--benchmark", bench])
    status("chart", label=label)
    step(f"chart {label}", [PY, "scripts/plot_experiment.py", "--run", run,
                            "--suite", suite, "--name", f"{experiment}-s{seed}"])
    status("audit", label=label)
    step(f"audit {label}", [PY, "scripts/smallbin_audit.py", "--run", run,
                            "--cohorts", "company_test,B,R", "--device", "cuda",
                            "--out", f"output/reports/smallbin_audit_{experiment}_s{seed}"])


def read_evalmode(name: str) -> dict | None:
    path = ROOT / "output" / "reports" / f"{name}_evalmode.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    final = data.get("final_ema.pt")
    if isinstance(final, list):
        final = final[0]
    if not isinstance(final, dict):
        return None
    out = {}
    for cohort in ("company", "b"):
        item = final.get(cohort)
        if not item:
            return None
        out[cohort] = {
            "lt10": float(item["lt10"]),
            "worst": float(max(item["per_bin"].values())),
        }
    out["s1"] = max(out["company"]["worst"], out["b"]["worst"])
    return out


def decide_448() -> tuple[bool, dict]:
    """Return (hold, numbers). hold=True means do not start the two."""
    m448 = read_evalmode("v11-company-diranchor15-448-s20261020")
    m20 = read_evalmode("v11-company-diranchor40-betaonly-s20261020")
    m21 = read_evalmode("v11-company-diranchor40-betaonly-s20261021")
    nums = {"448": m448, "betaonly_s20": m20, "betaonly_s21": m21}
    if m448 is None or m20 is None or m21 is None:
        return True, nums
    s1_224 = min(m20["s1"], m21["s1"])
    c_224 = min(m20["company"]["lt10"], m21["company"]["lt10"])
    b_224 = min(m20["b"]["lt10"], m21["b"]["lt10"])
    better = (m448["s1"] < s1_224) or (
        m448["company"]["lt10"] <= c_224 and m448["b"]["lt10"] <= b_224
    )
    return better, nums


def write_report(path: Path, title: str, nums: dict, extra: str = "") -> None:
    lines = [f"# {title}", "", "```json",
             json.dumps(nums, ensure_ascii=False, indent=2), "```", extra]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STEPS.write_text("", encoding="utf-8")
    status("等待 448 队列", predecessor=PREDECESSOR)
    wait_pid(PREDECESSOR)
    wait_gpu_idle()

    hold, nums = decide_448()
    write_report(ROOT / "output/reports/conditional_448_20261106.md",
                 "β-only@448 vs β-only@224", nums)
    if hold:
        status("held_448_better" if nums.get("448") else "held_missing_448_eval",
               numbers=nums)
        log("448 结果优于 224 β-only（或缺失评估）；按 owner 指令停在等待决定，"
            "不启动 Endo/纯图像实验")
        return 0

    status("448_not_better_launch_two")
    log("448 未优于 224 β-only；启动 Endo β-only 与纯图像回归")
    run_one("endo-betaonly", "input/configs/experiments/v11/company-betaonly-endo.yaml",
            "v11-company-betaonly-endo", 20261020, BENCH_ENDO, SUITE_ENDO)
    run_one("imageonly", "input/configs/experiments/v11/company-imageonly.yaml",
            "v11-company-imageonly", 20261020, BENCH_DA, SUITE_DA)

    endo = read_evalmode("v11-company-betaonly-endo-s20261020")
    image = read_evalmode("v11-company-imageonly-s20261020")
    da = read_evalmode("v11-company-diranchor40-betaonly-s20261020")
    result = {"endo": endo, "imageonly": image, "betaonly_da": da}
    beta_wins = bool(
        endo and image and da
        and min(endo["s1"], da["s1"]) < image["s1"]
    )
    endo_close = bool(
        endo and da
        and abs(endo["s1"] - da["s1"]) <= 0.15
        and abs(endo["company"]["lt10"] - da["company"]["lt10"]) <= 0.10
        and abs(endo["b"]["lt10"] - da["b"]["lt10"]) <= 0.10
    )
    chosen = (
        "endo_betaonly" if beta_wins and endo_close
        else ("da_betaonly" if beta_wins else "owner_review")
    )
    result["beta_wins"] = beta_wins
    result["endo_close"] = endo_close
    result["chosen"] = chosen
    write_report(ROOT / "output/reports/conditional_final_20261106.md",
                 "条件实验最终比较", result)
    status("done", chosen=chosen, numbers=result)
    log(f"conditional done; chosen={chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
