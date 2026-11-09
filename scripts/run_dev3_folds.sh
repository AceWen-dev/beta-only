#!/usr/bin/env bash
# D-0048 Step 2：用同一配方（RankSim+logloss γ=2，40 epoch，第40轮 EMA）
# 补 dev_fold=1、dev_fold=2 两次训练。fold0 已存在。
# 串行执行，单卡；每折结束检查 manifest 状态。
set -u
cd "$(dirname "$0")/.." || exit 1
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export PATH="$(dirname "$PY"):$PATH"
export PYTHONPATH="/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF:${PYTHONPATH:-}"
LOG=output/logs/dev3_folds.log
mkdir -p output/logs

for fold in 1 2; do
  echo "=== fold ${fold} start $(date '+%F %T') ===" | tee -a "$LOG"
  "${PY}" core/commands/train.py \
    --config input/configs/experiments/v11/dev3-g20.yaml \
    --tag dev3 --tag step2 \
    --set "data.dev_fold=${fold}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== fold ${fold} exit ${rc} $(date '+%F %T') ===" | tee -a "$LOG"
  if [ "${rc}" -ne 0 ]; then
    echo "fold ${fold} failed, stopping chain" | tee -a "$LOG"
    exit "${rc}"
  fi
done
echo "=== all folds done $(date '+%F %T') ===" | tee -a "$LOG"
