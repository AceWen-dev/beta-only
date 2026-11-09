#!/usr/bin/env bash
# D-0073 diranchor 跟进队列：
#   ① 第二个 seed（20261021，15 轮）复核「company 全档 <2mm」是否稳定；
#   ② 40 轮长 schedule（seed 20261020）——val 曲线证明 15 轮时仍在单调改善（D-0072）。
# 每步：训练 → 官方公司 test eval → benchmark2 逐档图 → 小档审计 → 分档准确率 → 汇总。
#
# 启动：
#   nohup bash scripts/run_diranchor_followup_20261105.sh > output/logs/diranchor_followup.log 2>&1 &
#
# 约定：
# - 除 epochs/min_epochs（②）与 seed（①）外，与 `company-diranchor.yaml` 逐字段一致
# - `data.num_workers=4/pin_memory=true`（D-0071）：只改加载并行度，样本内容与 batch 顺序不变；
#   ① 用 --set 传入以保持与原 diranchor run 同一份 config 文件
# - 选模全程看 val；test 每个 run 只评一次
# - 每步失败只记录、不中断；队列运行期间不要改 core/*.py
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1

PY=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/PolypMeasure/.venv/bin/python
export DLEF_DATA_ROOT=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/Betaregression_data
export PYTHONPATH=/media/twj/e81f2936-73e0-419c-9ade-f6d88f02dd93/DLEF
export DLEF_SKIP_POST_REPORT=1

BENCH=input/benchmarks/v11-company-test-v1.json
SUITE=input/benchmarks/benchmark2.json
FLOOR=output/reports/v11-company-base-3seed_evalmode.json
TAG=d0073
FAST="--set data.num_workers=4 --set data.pin_memory=true"

DIRANCHOR_S20=output/runs/beta-regression/v11-company-diranchor/20261105-025342_36433ff6_s20261020_631bbc
ORDINAL_RUN=output/runs/beta-regression/v11-company-ordinal/20261105-130723_e4c55d52_s20261020_bfeda1
STEPS=output/logs/diranchor_followup.steps.log
LOG=output/logs/diranchor_followup.chain.log
REPORT=output/reports/diranchor_followup_20261105.md
BINACC=output/reports/diranchor_bin_accuracy.md
mkdir -p output/logs
: > "$STEPS"

log()  { local m="[$(date '+%F %T')] $*"; echo "$m"; echo "$m" >> "$LOG"; }
step() { local n="$1"; shift
         log "STEP $n :: $*"
         if "$@" >> "$STEPS" 2>&1; then log "OK   $n"; else log "FAIL $n (exit $?)"; fi; }

# $1=experiment $2=seed → 最新的 completed 且非 debug run（CPU 冒烟 run 同 seed 同目录，必须排除）
find_run() {
  local exp="$1" s="$2" d
  for d in $(ls -dt "output/runs/beta-regression/${exp}"/*_s${s}_*/ 2>/dev/null); do
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
log "=== diranchor follow-up start (D-0073) ==="

# ---------- ① 第二个 seed ----------
step "train diranchor seed=20261021" \
  "$PY" core/commands/train.py --config input/configs/experiments/v11/company-diranchor.yaml \
    --set seed=20261021 $FAST --tag "$TAG"
S21=""
if S21=$(find_run v11-company-diranchor 20261021); then
  log "run diranchor s20261021 -> $S21"
  step "official eval diranchor s20261021" "$PY" core/commands/eval.py \
      --checkpoint "$S21/checkpoints/final_ema.pt" --split test --benchmark "$BENCH"
  step "chart diranchor 2-seed" "$PY" scripts/plot_experiment.py \
      --runs "$S21,$DIRANCHOR_S20" --suite "$SUITE" --name v11-company-diranchor-2seed \
      --note "diranchor 两 seed（D-0073 复核）"
  step "audit diranchor s20261021" "$PY" scripts/smallbin_audit.py \
      --run "$S21" --cohorts company_test,B,R --device cuda \
      --out output/reports/smallbin_audit_diranchor_s21
else
  log "no completed run for diranchor seed=20261021（跳过其后续步骤）"
fi

# ---------- ② 40 轮 ----------
step "train diranchor40 seed=20261020" \
  "$PY" core/commands/train.py --config input/configs/experiments/v11/company-diranchor40.yaml \
    --tag "$TAG"
L40=""
if L40=$(find_run v11-company-diranchor40 20261020); then
  log "run diranchor40 -> $L40"
  step "official eval diranchor40" "$PY" core/commands/eval.py \
      --checkpoint "$L40/checkpoints/final_ema.pt" --split test --benchmark "$BENCH"
  step "chart diranchor40" "$PY" scripts/plot_experiment.py \
      --run "$L40" --suite "$SUITE" --name v11-company-diranchor40 \
      --note "diranchor 40 轮（D-0072：15 轮时 val 仍在改善）"
  step "audit diranchor40" "$PY" scripts/smallbin_audit.py \
      --run "$L40" --cohorts company_test,B,R --device cuda \
      --out output/reports/smallbin_audit_diranchor40
else
  log "no completed run for diranchor40（跳过其后续步骤）"
fi

# ---------- 分档准确率（D-0070 指定的首要读点） ----------
ACC=("--frames" "output/reports/smallbin_audit/frames.csv" "--label" "base"
     "--frames" "output/reports/smallbin_audit_diranchor/frames.csv" "--label" "diranchor-s20"
     "--frames" "output/reports/smallbin_audit_ordinal/frames.csv" "--label" "ordinal")
[ -n "$S21" ] && ACC+=("--frames" "output/reports/smallbin_audit_diranchor_s21/frames.csv"
                       "--label" "diranchor-s21")
[ -n "$L40" ] && ACC+=("--frames" "output/reports/smallbin_audit_diranchor40/frames.csv"
                       "--label" "diranchor40")
step "bin accuracy" "$PY" scripts/bin_accuracy.py "${ACC[@]}" \
    --cohorts company_test,B,R --out "$BINACC"

step "tb symlink sync" "$PY" scripts/sync_tb_links.py

ARMS=("--arm" "ordinal=v11-company-ordinal:${ORDINAL_RUN}")
[ -n "$S21" ] && ARMS+=("--arm" "diranchor-2seed=v11-company-diranchor-2seed:${S21}")
[ -n "$L40" ] && ARMS+=("--arm" "diranchor40=v11-company-diranchor40:${L40}")
step "vs noise floor" "$PY" scripts/summarize_arms.py "${ARMS[@]}" \
    --base "$FLOOR" --steps-log "$STEPS" \
    --title "1. diranchor 跟进（D-0073）：第二 seed 复核 + 40 轮，对比 company-base 三 seed 噪声底" \
    --out "$REPORT"

log "=== diranchor follow-up done ==="
log "报告：$REPORT、$BINACC"
