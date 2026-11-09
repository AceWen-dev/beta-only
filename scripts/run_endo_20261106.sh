#!/usr/bin/env bash
# EndoStreamDepth beta-only: train -> official Endo eval -> Endo chart -> audit.
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

EXP=v11-company-betaonly-endo
CFG=input/configs/experiments/v11/company-betaonly-endo.yaml
TAG=d0085
LOG=output/logs/endo_20261106.chain.log
STEPS=output/logs/endo_20261106.steps.log
mkdir -p output/logs
: > "$STEPS"

log() { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG"; }
step() {
  local n="$1"; shift
  log "STEP $n :: $*"
  if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi
}

log "=== Endo beta-only start ==="
step "train" "$PY" core/commands/train.py --config "$CFG" --set seed=20261020 --tag "$TAG"

R=$(ls -dt "output/runs/beta-regression/${EXP}"/*/ 2>/dev/null | head -1)
R=${R%/}
if [ -z "$R" ] || [ ! -f "$R/checkpoints/final_ema.pt" ]; then
  log "no final_ema checkpoint; stop"
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
log "=== Endo beta-only done ==="
