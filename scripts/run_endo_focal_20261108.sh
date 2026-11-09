#!/usr/bin/env bash
# Endo 米制焦距头（D-0091）：train -> official company eval -> Endo chart -> audit.
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

EXP=v11-company-betaonly-endo-focal
CFG=input/configs/experiments/v11/company-betaonly-endo-focal.yaml
TAG=d0091
LOG=output/logs/endo_focal_20261108.chain.log
STEPS=output/logs/endo_focal_20261108.steps.log
mkdir -p output/logs
: > "$STEPS"

log() { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG"; }
step() {
  local n="$1"; shift
  log "STEP $n :: $*"
  if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi
}

log "=== Endo focal start ==="
step "train" "$PY" core/commands/train.py --config "$CFG" --set seed=20261020 --tag "$TAG"

R=$("$PY" - <<'PY'
import glob
import json
import os

cands = sorted(
    glob.glob(
        "output/runs/beta-regression/v11-company-betaonly-endo-focal/*/manifest.json"
    ),
    key=os.path.getmtime,
    reverse=True,
)
for path in cands:
    data = json.load(open(path))
    tags = data.get("tags") or {}
    if data.get("status") != "completed":
        continue
    if str(tags.get("debug", "")).lower() == "true":
        continue
    run = os.path.dirname(path)
    if os.path.isfile(os.path.join(run, "checkpoints", "final_ema.pt")):
        print(run)
        break
PY
)
R=${R%/}
if [ -z "$R" ] || [ ! -f "$R/checkpoints/final_ema.pt" ]; then
  log "no completed run with final_ema checkpoint; stop"
  exit 1
fi
log "run $R"

step "eval" "$PY" core/commands/eval.py \
  --checkpoint "$R/checkpoints/final_ema.pt" \
  --split test --benchmark input/benchmarks/v11-company-test-v1-endo.json
step "chart" "$PY" scripts/plot_experiment.py \
  --run "$R" --suite input/benchmarks/benchmark2-endo.json \
  --name "${EXP}-s20261020"
step "audit" "$PY" scripts/smallbin_audit.py \
  --run "$R" --cohorts company_test,B,R --device cuda \
  --out "output/reports/smallbin_audit_${EXP}-s20261020"
log "=== Endo focal done ==="
