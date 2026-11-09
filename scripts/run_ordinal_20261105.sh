#!/usr/bin/env bash
# D-0069 分档头队列：company-ordinal 训练 → 官方公司 test eval → benchmark2 逐档图
# → 小档审计（逐帧分支/几何）→ 分档准确率诊断 → 与三 seed 噪声底 + 两臂对比汇总。
#
# 启动（GPU 空闲时直接跑；约 2 小时训练 + 10 分钟诊断）：
#   nohup bash scripts/run_ordinal_20261105.sh > output/logs/ordinal_20261105.log 2>&1 &
#
# 约定：
# - 单变量：相对 company-base 只改 model.ordinal_head + loss.name（log_space 被 mm_ordinal
#   取代，属同一改动的必然结果）；seed 20261020、15 轮、从零、RankSim γ=2
# - 每步失败只记录、不中断
# - 队列运行期间不要改 core/*.py
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

EXP=v11-company-ordinal
CFG=input/configs/experiments/v11/company-ordinal.yaml
SEED=20261020
TAG=d0069
BENCH=input/benchmarks/v11-company-test-v1.json
SUITE=input/benchmarks/benchmark2.json
FLOOR=output/reports/v11-company-base-3seed_evalmode.json
BASE_RUN=output/runs/beta-regression/v11-company-base/20261104-103104_d8f11691_s20261020_035a6f
DIRANCHOR_RUN=output/runs/beta-regression/v11-company-diranchor/20261105-025342_36433ff6_s20261020_631bbc
GATEFLOOR_RUN=output/runs/beta-regression/v11-company-gatefloor/20261105-005349_dc4296d5_s20261020_4dfb35
STEPS=output/logs/ordinal_20261105.steps.log
LOG=output/logs/ordinal_20261105.chain.log
REPORT=output/reports/ordinal_20261105.md
mkdir -p output/logs
: > "$STEPS"

log()  { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG"; }
step() { local n="$1"; shift
         log "STEP $n :: $*"
         if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi; }

# 该 seed 最新的 completed 且非 debug run（CPU 冒烟 run 同 seed、同目录，必须排除）
find_run() {
  local d
  for d in $(ls -dt "output/runs/beta-regression/${EXP}"/*_s${SEED}_*/ 2>/dev/null); do
    if grep -q '"status": "completed"' "${d}manifest.json" 2>/dev/null \
       && ! grep -q '"debug": "true"' "${d}manifest.json" 2>/dev/null; then
      echo "${d%/}"; return 0
    fi
  done
  return 1
}

while pgrep -f "core/commands/(train|eval).py" >/dev/null 2>&1; do
  log "GPU 上仍有训练/评测进程，继续等待"; sleep 60
done

log "=== ordinal queue start (D-0069) ==="
step "train $EXP" "$PY" core/commands/train.py --config "$CFG" --tag "$TAG"

if ! R=$(find_run); then
  log "FAIL 没有 completed run，后续步骤全部跳过"
  log "=== ordinal queue aborted ==="
  exit 1
fi
log "run -> $R"

step "official eval (company test)" "$PY" core/commands/eval.py \
    --checkpoint "$R/checkpoints/final_ema.pt" --split test --benchmark "$BENCH"
step "chart (benchmark2)" "$PY" scripts/plot_experiment.py \
    --run "$R" --suite "$SUITE" --name "$EXP" --note "D-0069 分档头（ordinal head）"
step "smallbin audit" "$PY" scripts/smallbin_audit.py \
    --run "$R" --cohorts company_test,B,R --device cuda \
    --out output/reports/smallbin_audit_ordinal
step "bin accuracy" "$PY" scripts/bin_accuracy.py \
    --frames output/reports/smallbin_audit/frames.csv --label base \
    --frames output/reports/smallbin_audit_diranchor/frames.csv --label diranchor \
    --frames output/reports/smallbin_audit_gatefloor/frames.csv --label gatefloor \
    --frames output/reports/smallbin_audit_ordinal/frames.csv --label ordinal \
    --cohorts company_test,B,R --out output/reports/ordinal_bin_accuracy.md
step "tb symlink sync" "$PY" scripts/sync_tb_links.py
step "vs noise floor" "$PY" scripts/summarize_arms.py \
    --arm "ordinal=${EXP}:${R}" \
    --arm "diranchor=v11-company-diranchor:${DIRANCHOR_RUN}" \
    --arm "gatefloor=v11-company-gatefloor:${GATEFLOOR_RUN}" \
    --base "$FLOOR" --steps-log "$STEPS" \
    --title "1. 分档头（D-0069）对比 company-base 三 seed 噪声底与解压缩两臂" \
    --out "$REPORT"

log "=== ordinal queue done ==="
log "报告：$REPORT、output/reports/ordinal_bin_accuracy.md"
