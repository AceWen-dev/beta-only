#!/usr/bin/env bash
# D-0067 解压缩两臂队列：等 D-0066 主队列 + 标定尾巴都跑完（GPU 空闲）后，
# 依次训练 company-gatefloor（门控下界 0.5）与 company-diranchor（direct 分支锚定 D·S），
# 每臂：官方公司 test eval → benchmark2 三面板逐档图 → 与三 seed 噪声底对比汇总。
#
# 启动（在主队列/尾巴之后串进来，独立 nohup）：
#   nohup bash scripts/run_overnight_arms_20261105.sh > output/logs/arms_20261105.log 2>&1 &
#
# 约定：
# - 每臂只有一个变量（相对 company-base），seed 20261020 与 base 同 seed
# - 训练期间 DLEF_SKIP_POST_REPORT=1，出图统一在训练后做，避免与主队列同名覆盖
# - 每步失败只记录、不中断
# - 队列运行期间不要改 core/*.py
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

BENCH=input/benchmarks/v11-company-test-v1.json
SUITE=input/benchmarks/benchmark2.json
SEED=20261020
TAG=d0067
ARMS="v11-company-gatefloor v11-company-diranchor"
STEPS=output/logs/arms_20261105.steps.log
LOG=output/logs/arms_20261105.chain.log
REPORT=output/reports/overnight_20261104.md
mkdir -p output/logs
: > "$STEPS"

log()  { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG"; }
step() { local n="$1"; shift
         log "STEP $n :: $*"
         if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi; }

# $1=experiment → 该 seed 最新的 completed 且非 debug run 目录
find_run() {
  local exp="$1" d
  for d in $(ls -dt "output/runs/beta-regression/${exp}"/*_s${SEED}_*/ 2>/dev/null); do
    if grep -q '"status": "completed"' "${d}manifest.json" 2>/dev/null \
       && ! grep -q '"debug": "true"' "${d}manifest.json" 2>/dev/null; then
      echo "${d%/}"; return 0
    fi
  done
  return 1
}

TAIL_PID="${TAIL_PID:-3424030}"   # bash scripts/run_overnight_tail_20261104.sh
log "=== arms queue start (D-0067)：等待标定尾巴 pid=$TAIL_PID 退出 ==="
while kill -0 "$TAIL_PID" 2>/dev/null; do sleep 60; done
log "尾巴 pid=$TAIL_PID 已退出"
# 兜底：任何 company-* 训练/评测进程都退干净再上 GPU
while pgrep -f "core/commands/(train|eval).py .*company-" >/dev/null 2>&1; do
  log "仍有 company-* 训练/评测进程占用 GPU，继续等待"; sleep 60
done
sleep 30
log "GPU 空闲，开始两臂训练"

SPEC=""
for EXP in $ARMS; do
  CFG="input/configs/experiments/v11/${EXP#v11-}.yaml"
  step "train $EXP" "$PY" core/commands/train.py --config "$CFG" --tag "$TAG"
  if R=$(find_run "$EXP"); then
    log "run $EXP -> $R"
    SPEC="$SPEC --arm ${EXP#v11-company-}=${EXP}:${R}"
    step "official eval $EXP" "$PY" core/commands/eval.py \
        --checkpoint "$R/checkpoints/final_ema.pt" --split test --benchmark "$BENCH"
    step "chart $EXP (benchmark2)" "$PY" scripts/plot_experiment.py \
        --run "$R" --suite "$SUITE" --name "$EXP" \
        --note "D-0067 解压缩臂（${EXP#v11-company-}）"
  else
    log "no completed run for $EXP（跳过其 eval/出图）"
    SPEC="$SPEC --arm ${EXP#v11-company-}=${EXP}:output/runs/beta-regression/${EXP}/MISSING"
  fi
done

step "tb symlink sync" "$PY" scripts/sync_tb_links.py
# shellcheck disable=SC2086
step "arms summary" "$PY" scripts/summarize_arms.py \
    $SPEC \
    --steps-log "$STEPS" \
    --out output/reports/arms_20261105.md \
    --append-to "$REPORT"

log "=== arms queue done ==="
