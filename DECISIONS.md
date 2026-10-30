# DECISIONS.md — project decision log

<!--
Append-only. One entry per hard-to-reverse or surprising decision.
Do not rewrite or delete old entries; supersede them with a new entry.
Reversible choices (a normal experiment variant, a hyperparameter, a local
refactor) belong in the config or a run, not here.
-->

## When to write

- data source/version, split membership, grouping or leakage rule;
- frozen benchmark: test membership, labels, eval transform, metric protocol;
- baseline model/loss/optimizer/scheduler choice;
- output root, logging, retention or artifact policy future runs depend on;
- anything a future agent would otherwise re-litigate.

## When not to write

- code style, typos, local refactors;
- a normal experiment variant (it is a run);
- a reversible hyperparameter (sweep it).

IDs are sequential (`D-0001`, `D-0002`, ...) and are never reused.

## Entry template

```markdown
### D-000X — <title>
- Date: YYYY-MM-DD
- Status: proposed | accepted | superseded by D-000Y
- Context:
- Decision:
- Alternatives:
- Consequences:
- Evidence: config path, run id, eval id, benchmark id, commit
- Owner:
```

## Entries

<!-- Keep the template entries; the first project-specific decision is D-0003. -->

### D-0001 — Start from the DLEF research-project template

- Date: 2026-10-30
- Status: accepted
- Context: new project; need config resolution, run manifests, eval records and
  a frozen benchmark from day one.
- Decision: start from `templates/research-project`; keep `core/task.py` as the
  composition root; install `dlef` as a package instead of vendoring it.
- Alternatives: an ad-hoc training repo; copying `dlef/` into the project.
- Consequences: every run has a manifest/config/checkpoint; test evaluation
  requires a frozen benchmark; project code stays in `core/`.
- Evidence: `AGENTS.md`, `PROJECT.yaml`, `TUTORIAL.md`
- Owner: <name>

### D-0002 — Synthetic regression is smoke-only

- Date: 2026-10-30
- Status: accepted
- Context: the template ships a synthetic 1-D regression placeholder to prove
  the wiring.
- Decision: use it only for smoke tests; replace data/model/loss/metric before
  running or reporting a real experiment.
- Alternatives: treat the placeholder as a baseline.
- Consequences: synthetic runs are tagged `smoke`/`debug` and never reported.
- Evidence: `input/configs/base.yaml`, `core/tests/test_smoke.py`
- Owner: <name>

### D-0003 — v11 数据口径：镜下单值标签、病例级划分

- Date: 2026-10-30
- Status: accepted
- Context: 迁移 PolypMeasure 的 v11 与衍生实验；数据必须与既有评测口径一致。
- Decision: 标签只用 `内镜下息肉大小` 的单值（mm），不使用范围中点、不使用病理尺寸；
  划分按 `case_id`（train 2,044 帧 / val 205 帧）；B/D/R/清洗公司只作冻结测试。
- Alternatives: 范围取中点（会让 0–3 档没有样本）、病理尺寸（不同物理量，且用户明确不用）。
- Consequences: 1–2 mm 无真实开发病例；2–3 mm 仅 30 例。这一缺口在报告里必须写明。
- Evidence: `$DLEF_DATA_ROOT/curated/v11_frames/single-v1/manifest.json`、`STATE.md` §2
- Owner: twj

### D-0004 — 四个冻结 benchmark 与验收指标

- Date: 2026-10-30
- Status: accepted
- Context: 用户指定测试集为 B / D / R / 清洗公司四个，验收要求是 <10 mm 每个 1 mm 区间 MAE <2 mm。
- Decision: 冻结四个 benchmark：`v11-b-test-v1`、`v11-d-test-v1`、`v11-r-test-v1`、
  `v11-clean-test-v1`；指标为逐区间 MAE、`worst_bin`（2–9 mm 最差档）、`mae_lt10`、`acc10`。
  测试集不参与训练与选模；`worst_bin` 只用于 dev 选模。
- Alternatives: 只冻结 D；用总体 MAE 作唯一指标。
- Consequences: 改成员/标签/指标必须新建 benchmark 版本，不能原地改。
- Evidence: `input/benchmarks/v11-*-test-v1.json`
- Owner: twj

### D-0005 — v11 组件在 DLEF 中的映射

- Date: 2026-10-30
- Status: accepted
- Context: 需要把 v11 与衍生实验（RankSim、投影、2.5× 第三视图）接入框架，同时保留原权重。
- Decision:
  - 数据：`data.name=v11_frames`，manifest 驱动，`case_id` 分组；
  - 模型：`model.name=v11`，`views` 2/3，`proj` 开关，`init_checkpoint` 可加载旧权重；
  - 损失：`mm_case_bin`（毫米域、病例/区间等权）与 `mm_rank`（官方 RankSim，读 `ModelOutput.aux`）；
  - 指标：`metrics: [v11]`，一次产出逐区间 MAE / worst_bin / acc10；
  - 优化器：`v11_adamw`（backbone/head 两组学习率）。
- Alternatives: 把训练循环写进 `core/commands/`（违反框架禁令）；直接在 DLEF core 里加项目代码。
- Consequences: `core/task.py` 与 `core/commands/*` 保持原样；新组件都走注册表与配置。
- Evidence: `core/models/v11.py`、`core/losses/mm_case_bin.py`、`core/losses/mm_rank.py`、`core/metrics/v11.py`
- Owner: twj

### D-0006 — 迁移范围只含 baseline 与 RankSim 相关

- Date: 2026-10-30
- Status: accepted
- Context: 用户要求不要把 v11 的所有衍生实验都迁移进来。
- Decision: 本轮只迁移 `v11-baseline`（毫米损失）与 RankSim 相关（`v11-ranksim` 直接约束、
  `v11-proj` 独立投影）；2.5× 第三视图（`wide`）不迁移，配置已删除。
- Alternatives: 一次性迁移全部衍生实验（被用户否决）。
- Consequences: `model.views` 仍保留 2/3 的代码能力，但没有对应配置；后续若要恢复需新建 variant。
- Evidence: `input/configs/experiments/v11/{baseline,ranksim,proj}.yaml`
- Owner: twj

### D-0007 — 早停保存 best checkpoint，统一按 worst_bin 选模

- Date: 2026-10-30
- Status: accepted
- Context: 旧 Runner 只保存 `last.pt`，早停不回滚最佳 epoch；基线用 `v11/worst_bin`
  选模，backbone variants 却用 `v11/mae_lt10`，test 实际评的是最后一轮而不是 best。
- Decision: 采用 DLEF `937f11e`；早停指标提升时保存 `checkpoints/best.pt`；
  所有 v11 配置的 `early_stopping.monitor` 统一为 `v11/worst_bin`；正式 test eval
  使用 `best.pt`，`last.pt` 仅用于 resume。
- Alternatives: 继续用 `last.pt`；按 `mae_lt10` 选模；每 epoch 存 checkpoint 后手选。
- Consequences: 旧 run 只有 `last.pt`，全部降为 exploratory；baseline/full/partial/frozen
  用新代码重跑，旧 run 在新 run 成功后删除。
- Evidence: DLEF `937f11e`、`scripts/rerun_all.py`、`output/reports/rerun_*.json`
- Owner: twj
