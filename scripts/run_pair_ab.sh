#!/usr/bin/env bash
# D-0050 配对训练 A/B：等基线队列结束后串行跑两臂。
# 两臂除 loss.pair_beta_weight 外完全相同（帧、采样、步数、seed、fold）。
#   A: dev3-pair.yaml       配对采样，无 β 一致性
#   B: dev3-pair-beta.yaml  配对采样 + 0.5·L_pair,beta
set -u
cd "$(dirname "$0")/.." || exit 1
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF:${PYTHONPATH:-}"
LOG=output/logs/pair_ab.log
mkdir -p output/logs

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "=== waiting for pid ${WAIT_PID} $(date '+%F %T') ===" | tee -a "$LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "=== baseline queue done, starting A/B $(date '+%F %T') ===" | tee -a "$LOG"
fi

for cfg in dev3-pair dev3-pair-beta; do
  echo "=== ${cfg} start $(date '+%F %T') ===" | tee -a "$LOG"
  "${PY}" core/commands/train.py \
    --config "input/configs/experiments/v11/${cfg}.yaml" \
    --tag pair --tag d0050 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== ${cfg} exit ${rc} $(date '+%F %T') ===" | tee -a "$LOG"
  if [ "${rc}" -ne 0 ]; then
    echo "${cfg} failed, stopping chain" | tee -a "$LOG"
    exit "${rc}"
  fi
done
echo "=== pair A/B done $(date '+%F %T') ===" | tee -a "$LOG"
