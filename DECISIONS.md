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

### D-0008 — 修 manifest 的框：v1 作废，改用 single-v2

- Date: 2026-10-30
- Status: accepted（其中“旧 run 保留不删”由 D-0009 取代）
- Context: DLEF 的 v1 manifest 在生成时没有复制 `bbox_content`，后续补 `content_box` 的脚本又被 `zip` 截断，导致
  train 938 行、val 6 行、test_clean 206 行把“整幅内容区”当成了框；这使 v1 上的 baseline 与 A/B/C 结果无效。
- Decision:
  - 框按来源逐行回填：v2 公司表 757、lesion-query 840、A/V 掩码 104、B/D/R/clean 测试表 751；
  - 新建 `curated/v11_frames/single-v2`，v1 保留只为追溯；
  - 指标改为 0–10 mm 全区间，空区间报 `NaN`（无法评价），`worst_bin` 只对非空区间取最大；
  - 新建四个 v2 benchmark（`v11-{b,d,r,clean}-test-v2`）。
- Alternatives: 原地改 v1（违反数据集版本规则）。
- Consequences: v1 上的 DLEF baseline 与 A/B/C 全部作废，需在 v2 重跑；旧 run 保留不删。
- Evidence: 校验 0 个全内容框、0 个越界框、span 与源表逐行一致；`input/benchmarks/v11-*-test-v2.json`
- Owner: twj

### D-0009 — 删除 v1 run，只保留 v2 证据

- Date: 2026-10-30
- Status: accepted
- Context: v1 输入无效，旧 run 只保留追溯价值；用户要求删除，避免实验目录干扰观察。
- Decision: 删除全部 `single-v1` train/eval run 及未被 v2 run 引用的 artifact；保留 v1 数据和
  v1 benchmark JSON 作为不可变历史，不删除。
- Alternatives: 继续保留 v1 run（被用户否决）；只把 v1 run 移到外部归档（当前不需要）。
- Consequences: RunStore 里只保留 v2 证据；v1 数字只在文档/决策历史中出现，不能与 v2 混用。
- Evidence: 删除 21 个 v1 run、28 个 artifact；`output/runs` 只剩 v2 baseline/ranksim/proj。
- Owner: twj

### D-0010 — RankSim 系列从官方 ImageNet 权重起训

- Date: 2026-10-30
- Status: accepted
- Context: 早期 `v11-ranksim` / `v11-proj` 配置仍加载
  `pm/runs/size/beta-image-v10-r50-da-20261024/best.pt`，起点不是官方 ImageNet
  权重；用户要求所有 RankSim 实验从官方权重开始。
- Decision: `ranksim.yaml` 和 `proj.yaml` 的 `model.init_checkpoint` 设为 `""`；
  `V11Net` 使用 torchvision `ResNet50_Weights.IMAGENET1K_V1` 作为 backbone 起点。
  旧的 pm 初始化 RankSim runs 删除后重训，新 run 使用 `v2-imagenet` tag。
- Alternatives: 保留旧 RankSim runs（起点不一致，结果不能作为本次对照）；
  只改文档不改配置（不满足可复现要求）。
- Consequences: RankSim 系列的模型初始化与 v10 checkpoint 解耦；与仍使用
  v10 初始化的 baseline 不是严格单变量对照，报告时必须注明。
- Evidence: `input/configs/experiments/v11/{ranksim,proj}.yaml`、
  `core/tests/test_config_contract.py::test_ranksim_variants_start_from_imagenet`
- Owner: twj

### D-0011 — v11 训练配方升级：80 轮 + warmup + EMA + 档位均衡采样

- Date: 2026-10-30
- Status: accepted
- Context: 官方 RankSim 在 PolypMeasure 侧跑 40 轮得 B 1.183 / D 1.133，但第 40 轮的 dev
  仍在单调下降（`val` 1.225、`worst` 1.761 均为全程最低），说明 40 轮不是收敛点。
  DLEF 迁移版还缺官方配方里的两件东西：权重 EMA 与病例/档位均衡采样，所以 24 轮的
  DLEF RankSim（B 1.37 / D 1.22）不能与官方数字直接并列。
- Decision:
  - 训练 80 轮：前 3 轮线性 warmup，其余余弦退火到 0；
  - EMA decay 0.999，逐 step 更新；验证与选模都用 EMA 权重（指标名 `ema/...`）；
  - 采样按病例尺寸档均衡：3–6 mm 档权重 1，稀有档 `min(5, base/count)`，再按病例帧数归一；
  - 选模看 3 点平滑的 `ema/v11/worst_bin`；`min_epochs=80`，本轮不早停，先看满 80 轮曲线。
- Alternatives: 只把轮数加到 80（缺 EMA/采样，复现不了官方量级）；只加 EMA 跑 40 轮
  （官方曲线已否证，40 轮没收敛）。
- Consequences: 这是一次**配方整体升级**，与 24 轮 DLEF 结果、A/B/C 结果都不是单变量对照；
  唯一可比的对照是 PolypMeasure 官方 40 轮。若 80 轮仍在末轮创新低，下一轮加到 120。
- Evidence: `input/configs/experiments/v11/ranksim-best.yaml`、`core/data/samplers.py`、
  `adapters/torch/ema.py`、`DLEF/docs/TUTORIAL.md`；采样份额 2–3 mm 4.0%→13.2%、
  7–8 mm 6.9%→12.7%（`core/tests/test_case_bin_sampler.py` 固定算法）。
- Owner: twj

### D-0012 — 拆分 80 epoch RankSim exploratory run

- Date: 2026-10-30
- Status: accepted
- Context: 另一个 80 epoch、关闭 early stopping、`checkpoint_every=5` 的实验被用同一个
  `experiment: v11-ranksim` 启动，和 official 40 epoch run 混在同一目录；进程已结束，
  manifest 仍残留 `running`。
- Decision: 新建 `ranksim-long80.yaml`（`experiment: v11-ranksim-long80`）；恢复
  `ranksim.yaml` 为 official 40 epoch 配置；把 run `20261030-211500...` 移到
  `v11-ranksim-long80/`，标记 killed，并保留原始 `config.resolved.json` 作为历史。
- Alternatives: 删除该 run（用户要求保留）；继续混在 `v11-ranksim`（目录不可读）。
- Consequences: `v11-ranksim` 只保留 official run；`v11-ranksim-long80` 是独立的
  exploratory experiment，不参与 official 对照。
- Evidence: `input/configs/experiments/v11/ranksim-long80.yaml`、
  `output/runs/beta-regression/v11-ranksim-long80/20261030-211500_20c29ceb_s20261020_6841a2`
- Owner: twj
