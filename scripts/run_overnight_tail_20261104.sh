#!/bin/bash
# D-0066 第 4 步（尾巴）：等主队列跑完、GPU 空出来后，做单调标定诊断并追加进过夜报告。
# 不改主队列脚本；独立 nohup 启动。
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF

BASE=output/runs/beta-regression/v11-company-base/20261104-103104_d8f11691_s20261020_035a6f
SEEDS="20261021 20261022"
STEPS=output/logs/overnight_20261104.steps.log
TAIL=output/logs/overnight_20261104.tail.log

log() { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$TAIL"; }
step() {
  local n="$1"; shift
  log "STEP $n :: $*"
  if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi
}
find_run() {
  local s="$1" d
  for d in $(ls -dt output/runs/beta-regression/v11-company-base/*_s${s}_*/ 2>/dev/null); do
    if grep -q '"status": "completed"' "${d}manifest.json" 2>/dev/null; then
      echo "${d%/}"; return 0
    fi
  done
  return 1
}

DRIVER_PID="${DRIVER_PID:-3420052}"   # bash scripts/run_overnight_20261104.sh
log "=== tail start：等待主队列 pid=$DRIVER_PID 退出 ==="
while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 60; done
log "主队列 pid=$DRIVER_PID 已退出"
# 兜底：若还有 company-base 训练进程占着 GPU，继续等
while pgrep -f "core/commands/train.py .*company-base.yaml" >/dev/null 2>&1; do
  log "仍有训练进程占用 GPU，继续等待"; sleep 60
done
sleep 30   # 让主队列的汇总/图表落盘
log "GPU 空闲，开始标定诊断"

RUNS="$BASE"
for S in $SEEDS; do
  if R=$(find_run "$S"); then RUNS="$RUNS $R"; else log "WARN 找不到 seed=$S 的 completed run，跳过"; fi
done
log "诊断 run：$RUNS"

for R in $RUNS; do
  step "recal $(basename "$R")" \
    "$PY" scripts/bias_recal_diagnosis.py --run "$R" --device cuda
done

step "recal summary" \
  "$PY" scripts/summarize_recal.py \
    --dir output/reports/recal_diagnosis \
    --append-to output/reports/overnight_20261104.md \
    --out output/reports/overnight_20261104_recal.md

log "=== tail done ==="
