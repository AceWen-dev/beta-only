#!/usr/bin/env bash
# 过夜队列（D-0066）：company-base 补 seed 20261021/20261022 → 官方 eval → 三 seed 均值带图 → 汇总。
#
# 启动：
#   nohup bash scripts/run_overnight_20261104.sh > output/logs/overnight_20261104.log 2>&1 &
#
# 约定：
# - 一个 seed 一个 run（AGENTS.md：seed 是 run 维度，不是 config 变体）→ 只用 --set seed=
# - 训练期间 DLEF_SKIP_POST_REPORT=1：三个 seed 同属 experiment v11-company-base，
#   train.py 的自动出图会用同名 --name 互相覆盖 output/reports/v11-company-base_evalmode.json
# - 每步失败只记录、不中断（夜里没人看着，宁可少一步也别整晚白跑）
# - 队列运行期间不要改 core/*.py：否则第二个 run 与第一个不可比
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

CONFIG=input/configs/experiments/v11/company-base.yaml
BASE=output/runs/beta-regression/v11-company-base/20261104-103104_d8f11691_s20261020_035a6f
BENCH=input/benchmarks/v11-company-test-v1.json
SUITE=input/benchmarks/benchmark2.json
TAG=d0066
SEEDS="20261021 20261022"
STEPS=output/logs/overnight_20261104.steps.log
mkdir -p output/logs
: > "$STEPS"

log()  { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$STEPS"; }
step() { local n="$1"; shift
         log "STEP $n :: $*"
         if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi; }

find_run() {  # $1=seed → 最新的 completed run 目录
  local s="$1" d
  for d in $(ls -dt output/runs/beta-regression/v11-company-base/*_s${s}_*/ 2>/dev/null); do
    if grep -q '"status": "completed"' "${d}manifest.json" 2>/dev/null; then
      echo "${d%/}"; return 0
    fi
  done
  return 1
}

log "=== overnight queue start (D-0066) ==="
log "base run: $BASE"
NEW=""
for S in $SEEDS; do
  step "train seed=$S" "$PY" core/commands/train.py \
      --config "$CONFIG" --set "seed=$S" --tag "$TAG"
  if R=$(find_run "$S"); then
    log "run seed=$S -> $R"
    NEW="${NEW:+$NEW,}$R"
    step "official eval (company test) seed=$S" "$PY" core/commands/eval.py \
        --checkpoint "$R/checkpoints/final_ema.pt" --split test --benchmark "$BENCH"
  else
    log "no completed run for seed=$S（跳过其 eval）"
  fi
done

ORDER="${NEW:+$NEW,}$BASE"
log "plot order: $ORDER"
if [ -n "$NEW" ]; then
  step "3-seed chart+json" "$PY" scripts/plot_experiment.py \
      --runs "$ORDER" --suite "$SUITE" --name v11-company-base-3seed \
      --note "company-base 三 seed 噪声底（D-0066）"
else
  log "SKIP 3-seed chart（没有新的 completed run）"
fi

step "tb symlink sync" "$PY" scripts/sync_tb_links.py
step "summary md" "$PY" scripts/summarize_overnight.py \
    --runs-order "$ORDER" \
    --out output/reports/overnight_20261104.md \
    --steps-log "$STEPS"
log "=== queue done ==="
