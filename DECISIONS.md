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

### D-0013 — 重置为 best80 训练策略，删除旧策略实验

- Date: 2026-10-30
- Status: accepted
- Context: 用户确认旧的 40 轮/早停/普通采样策略不再使用；当前唯一正确的策略是
  80 轮 + warmup + EMA + case-bin sampler，并且正在训练的 `v11-ranksim-best` 要保留。
- Decision: 删除所有其他 run（baseline、旧 ranksim、proj、long80、smoke）及未引用
  artifact；只保留当前 `v11-ranksim-best`；后续 baseline/proj 使用同一 best80 recipe。
- Alternatives: 保留旧 run 作对比（用户明确否决，认为策略错误会污染观察）。
- Consequences: 旧策略结果不再出现在 RunStore；当前只有 best80 ranksim 在训练；
  baseline/proj 需要按新 recipe 重新建立。
- Evidence: 删除 19 个 run、29 个 artifact；保留 run
  `20261030-212317_09bdfb94_s20261020_17e52f`
- Owner: twj

### D-0014 — 对照臂改名去掉轮数后缀，并统一为 40 epoch

- Date: 2026-10-30
- Status: accepted
- Context: 参考实验 `v11-ranksim-best` 跑到 epoch 50 仍无改善（val 从 epoch 12–20
  起 plateau），继续跑满 80 轮收益很低；原计划 5 个 80 轮对照臂会再耗约 7.5 h。
- Decision: 5 个对照臂（baseline、backbone-full/partial/frozen、proj）改为 40 epoch
  （`epochs` 与 `early_stopping.min_epochs` 同步），文件名/实验名去掉轮数后缀：
  `*-best80.yaml` → `*-best.yaml`，`v11-*-best80` → `v11-*-best`；队列脚本改名
  `scripts/run_best_queue.py`，tag `best80` → `best`。参考实验 `v11-ranksim-best`
  保持 80 epoch 不变。
- Alternatives: 继续 80 轮（时间成本高且 plateau 证据充分）；只改配置不改名
  （记录会误导为 80 轮实验）。
- Consequences: 对照臂与参考实验轮数不同，比较时需注明；已有 `v11-ranksim-best`
  run 的目录和标签不变；旧 DECISIONS 条目中的 `best80` 作为历史保留。
- Evidence: `input/configs/experiments/v11/*-best.yaml`、`Makefile`、`AGENTS.md`、
  `README.md`、`TUTORIAL.md`、`core/tools/*`、`core/tests/*`
- Owner: twj

### D-0015 — 队列移除与 baseline 重复的 backbone-full 臂

- Date: 2026-10-30
- Status: accepted
- Context: `v11-backbone-full-best` 解析后与 `v11-baseline-best` 完全相同
  （`ModelConfig.backbone` 默认就是 `full`），队列会跑两个同配置实验。
- Decision: 从 `scripts/run_best_queue.py` 的 TARGETS 中移除 backbone-full；
  对照臂变为 4 个（baseline、backbone-partial、backbone-frozen、proj）。
  配置文件 `backbone-full-best.yaml` 保留，但不再排队。
- Alternatives: 保留该臂作冗余校验（无信息增益，浪费约 45 min GPU）。
- Consequences: D-0014 中“5 个对照臂”按本条修正为 4 个。
- Evidence: `scripts/run_best_queue.py` TARGETS
- Owner: twj

### D-0016 — 修复选模：raw/EMA 分开选、分开存，EMA 加 warmup

- Date: 2026-10-30
- Status: accepted
- Context: `v11-ranksim-best` 用 `monitor: ema/v11/worst_bin` 选模，`best.pt` 存的是
  EMA 权重。`ModelEMA` 从初始权重起步、按 0.999/step 平均（视野约 1000 步 ≈ 7.8 epoch），
  epoch 20 时仍含约 7.7% 初始（随机 head）权重，EMA 指标早期严重失真。结果是
  `best.pt` = epoch 63 EMA（val worst_bin 2.454），而原始权重 epoch 20 的
  val worst_bin 是 2.156，选模丢掉了真正更好的 checkpoint。
- Decision（DLEF 仓库）:
  - runner 维护两条独立选模轨迹：`best.pt` = 原始权重，按 monitor 去掉 `ema/` 前缀
    后的指标选；`best_ema.pt` = EMA 权重，按 `ema/...` 指标选；两者都写入 artifacts。
  - `Run.best_checkpoint_path` 只认 stem 恰为 `best` 的文件，避免误取 `best_ema.pt`。
  - `ModelEMA` 增加 warmup：默认在 `1/(1-decay)` 步内把 decay 从 0 线性升到配置值。
  - runner 的 EMA 验证改用 `deepcopy` 保存原始 state，支持非 torch Task。
- Alternatives: 只把 monitor 改成 raw（丢失 EMA 轨迹）；只调 decay（治标）。
- Consequences: `best.pt` 语义变为“原始权重的最优”。EMA 指标与旧 run 不可直接逐值比较；
  已完成的参考 run 其 `best.pt` 仍是 EMA@63，与新 run 比较时需用 raw checkpoint 或注明。
  旧代码启动的 baseline 臂已终止并重跑，保证 4 个对照臂使用同一版选模逻辑。
- Evidence: DLEF `dlef/runner.py`、`dlef/run.py`、`adapters/torch/ema.py`、
  `tests/test_early_stopping.py`、`tests/test_ema.py`；项目 `scripts/run_best_queue.py`。
- Owner: twj

### D-0017 — 复现调查：schedule 是主因，队列改为复现优先

- Date: 2026-10-30
- Status: accepted
- Context: 80 轮 `v11-ranksim-best` test 全面差于预期（worst_bin 3.23–4.81），且
  「case_bin 采样有害公开集」的结论来自与 24/40 轮 shuffle 旧 run 的跨配方对比。
  调查 PolypMeasure 侧原始实验发现：`v11-rank2-official-20261030` 已在**当前同一份
  single-v2 数据**上复现好结果（40 轮、官方 RankSim γ=1.0/interp=1.0、EMA 选模，
  test `<10` MAE B 1.183 / D 1.133 / R 2.366）。DLEF 迁移版与其逐项一致
  （数据成员/bbox/模型/rank 损失/采样器/优化器/EMA/seed 全部核对相同），唯一实质
  差异是 cosine 80 轮 vs 40 轮：中后期 LR 偏高（ep20 约 1.75e-4 vs 1.06e-4），
  raw 权重震荡、EMA 卡在 1.61（PM 同期降到 1.23–1.35）。次要差异：manifest
  `content_box` 约 38% 行与现算差 1–3px、`D` 约 36% 行差 ≤1.8% 相对（影响存疑）。
- Decision: 新建 `ranksim-repro.yaml`（`v11-ranksim-repro`，40 轮、patience 10、
  min_epochs 0，其余与 ranksim-best 相同）和 `baseline-shuffle.yaml`
  （`v11-baseline-shuffle`，仅 `data.sampler=shuffle`）；队列顺序改为
  baseline-best（在跑，自动评）→ ranksim-repro（评 best.pt + best_ema.pt）→
  baseline-shuffle → backbone-partial/frozen → proj；每次启动前等待 GPU 空闲，
  评估按 checkpoint digest 幂等复用。
- Alternatives: 直接按原计划跑 shuffle/backbone 臂（结论仍被 schedule 混淆）；
  在 PolypMeasure 侧继续实验（DLEF 是正式记录框架，需要可追溯的复现）。
- Consequences: 「case_bin 有害」降级为待验证假设；若 repro 对齐 rank2-official，
  80 轮参考 run 的失败归因于 schedule 而非采样器；新 arms 与参考 run 的比较需注明
  选模版本（D-0016 前后）。
- Evidence: PolypMeasure `runs/size/v11-rank2-{official,gaussian,proj}-20261030/`、
  `runs/size/v11-mm-bin{,-ranksim}-20261030/`、`outputs/training/v11-mmrank-dev-20261030/da/`；
  本项目 `input/configs/experiments/v11/{ranksim-repro,baseline-shuffle}.yaml`、
  `scripts/run_best_queue.py`
- Owner: twj

### D-0018 — 修复 content 裁剪：数据集现算 detect_image_area

- Date: 2026-10-31
- Status: accepted
- Context: 24 样本张量级对比发现 DLEF 与 PM 管线 8/24 样本 RGB 实质差异（max 3.92
  归一化单位），根因是 manifest 冻结的 `content_box` 与现算
  `detect_image_area` 在约 38% 样本上差 1–57px（集中在 recover/low_confidence 行）；
  bbox/depth/scalars/truth/letterbox/损失/模型全部逐值一致。40 轮同配方重跑
  （ranksim-repro）仍远差于 PM rank2-official（B lt10 2.25 vs 1.18），而旧管线上
  无 RankSim 的 baseline-best 反而 B 1.37——管线差异甚至翻转了 RankSim 的结论方向。
- Decision: 把 `detect_image_area`（PM `internal_samples.py` 逐字移植）放进
  `core/data/datasets/v11_frames.py`，`__getitem__` 一律现算内容裁剪；manifest
  的 `content_box` 保留但不再用于像素。验收：修复后同 24 样本 24/24 张量一致
  （max diff 0.00000）；`make check` 37 测试全过。实测检测 40ms/帧（含 imread），
  可接受。修复前的 baseline-shuffle 旧管线 run 终止（failed），随队列重跑。
- Alternatives: 重建 manifest（触碰冻结数据）；跨仓库 import
  `polyp_measurement`（项目耦合 PM 环境）。
- Consequences: 2026-10-31 00:47 之前的所有 run 用旧管线（像素与 PM 不同），
  跨修复点的数字不可直接比较；epoch 耗时略增（现算检测）；逐帧结果与 PM 逐像素对齐。
- Evidence: `core/data/datasets/v11_frames.py`、本会话 24 样本对比脚本输出、
  `runs v11-ranksim-repro 20261030-235746`（旧管线 40ep 复现结果）
- Owner: twj

### D-0019 — B/D 达标路径：尺度异质性诊断与 scale-consistency 增广

- Date: 2026-10-31
- Status: accepted
- Context: 合格标准收窄为 **B+D 全部 1mm 档 MAE<2mm**（R 降为补充评测）。当前
  最优配方（baseline-shuffle）未达标档：D 2–3（2.46,n=8）/ D 8–9（2.17,n=39）/
  B 2–3（2.93,n=15）/ B 7–8（3.12,n=5）/ B 8–9（2.1）。诊断发现：
  (1) 所有超标档 bias==mae（100% 单向）；(2) B 7–8 的 5 帧是**同一息肉**
  （polyp-size-04），其隐含 k=truth/S≈0.039 是公司域同档（≈0.020）的 2 倍——
  公开域放大率不同导致系统性低估；(3) val 拟合的 isotonic 标定在 test 小档
  恶化（域+尺寸条件偏差，非全局标定）；(4) **训练集四个来源的表观尺度
  k=truth/S 互差 2–15 倍**（company≈0.021、newdata≈0.011、phantom≈0.05、
  virtual≈0.15）——早前「小档框偏大 35%」的归因是来源混合的假象，
  company 内部只有约 10–15%；(5) 公开训练数据不可用：Polyp-Size 全部 42 视频
  就是 B 测试集本身，同视频帧入训=病例级泄漏。
- Decision: 放弃全局 `span_calibration`（其 k(S) 拟合被 virtual/phantom 小 S 行
  污染，factor 高达 3.0，不可用；代码保留、默认关闭）；改用
  **scale-consistency 增广**：`data.scale_aug≥1` 时训练侧随机以 α∈[1/aug,aug]
  同步缩放内容图与 bbox（S 同步变化），标签不变——模拟不同放大率下的同一
  息肉，教模型表观尺寸无关的测量。首个实验 `baseline-scaleaug.yaml`
  （scale_aug=2.0，其余同 baseline-shuffle）。
- Alternatives: 按 case 前缀做 per-source span 归一化（company 内部机制弱，
  收益小）；紧视图提分辨率（紧视图已 ~1.25× 放大，非瓶颈）；并入公开帧
  （泄漏墙，排除）。
- Consequences: 增广只在 train 生效，eval 不变；`span_calibration` 作为
  typed config 保留但需先解决多源污染才有意义；若 scale aug 无效，
  B/D 2–3mm 的剩余手段只有新增公司域小息肉病例。
- Evidence: `output/reports/calibration_diagnosis.md`、来源×档位 k 表、
  `data/raw/public/polyp-size/`（8,196 帧=42 视频）、B 集构成核查
- Owner: twj

### D-0021 — 公司域标签为整数 mm 粒度；D 8-9 档受标签粒度限制

- Date: 2026-10-31
- Status: accepted
- Context: valmerge 后 D 8-9 恶化（2.34→2.78）触发逐样本诊断：该档 39 个样本
  **标签全部恰为 8.00**；进一步核查发现 D 测试集 356 帧 100% 为整数 mm 标签
  （5.0/6.0/8.0/10.0 为主），train 亦 95% 整数；B 集（公开域）0% 整数
  （连续值如 7.48）。模型在 D 8-9 档预测 5–7（系统性低估），其中一部分可能是
  「连续预测 vs 整数化镜下目测」的口径差，而非纯模型错误。
- Decision: 该发现作为 D 集 8-9/相邻档评测解读的固定前提记录：公司域标签
  粒度为 1mm，这些档的 MAE 天花板受标签粒度限制。不做预测取整等 post-hoc
  技巧（会破坏 B 域连续标签、且属 test-informed）。valmerge seed 复核过夜运行。
- Alternatives: 预测量化到整数/半整数（破坏 B 域、test-informed）；
  立即修订标签（无金标准，不可行）。
- Consequences: 向用户/报告解读 D 集大档 MAE 时需注明标签粒度；
  公司域未来标注建议记录半毫米粒度。
- Evidence: 标签粒度统计（test_d 100%/train 95%/test_b 0% 整数百分位）、
  D 8-9 逐样本表（39/39 truth=8.00）
- Owner: twj

### D-0022 — 收窄两条过宽结论；dev 池病例级三折 + 混合损失预注册对照

- Date: 2026-11-01
- Status: accepted
- Context: 外部审查指出三点：(1) valmerge 同时改变训练成员与选模集，且新 val
  只含 4–6mm——选模看不见 2–3 与 7–10 档，收益归因（「9 帧小样本即达标」）与
  终局判断（「现有数据不可达」）均为过早结论；(2) D 8–9 的「标签粒度限制」
  说法过强——量化误差至多 ~0.5mm，解释不了 2.6–2.9 的 MAE，且无独立证据前
  不能把该档移出验收；(3) B 7–8 单病例失败不代表整类形态失败。
- Decision:
  1. **收窄结论**：D 2–3 写「接近达标」（一个 seed 仍 2.08）；撤回「现有数据
     不可达」；D 8–9 保持验收内，标签粒度仅作解读前提；B 7–8 为单病例失败。
  2. **新选模协议**：dev 池（train+val，2,249 帧/605 例）病例级三折
     （按来源×中位尺寸档分层轮转，`data.dev_folds/dev_fold`），每折 val 覆盖
     全部尺寸档；三折轮换，稀缺病例既有训练机会也有一次诚实留出预测。
  3. **预注册对照**：A=纯相对损失 vs B=混合损失
     （0.75·|err|+0.25·m·|err|/max(d,1)，m=5.0 与 w=0.25 预先固定），
     2 损失 × 3 折 = 6 次训练，seed 固定 20261020，不碰 test。
     推进条件：B 的 dev 7–10mm 最差档（帧平均）改善 ≥0.5mm 且 2–3mm 退化
     ≤0.2mm；报告全部档位。满足才补 seed。
  4. 评估补充**病例等权 MAE**（病例内帧误差均值再病例间均值），与帧平均并列
     报告；最终产品结论需未参与决策的独立病例。
- Alternatives: 立即采数据（审查判断为时过早）；换模型架构（无证据支持）；
  扩展参数扫描（违背预注册纪律）。
- Consequences: valmerge 的收益归因待 6 run 对照后重写；`v11-baseline-valmerge`
  的结果保留但标注协议局限；若对照仍无改善，下一步才是检查 β/直接/门控分支
  输出，仍不先改架构。
- Evidence: `core/data/datasets/v11_frames.py:assign_dev_folds`、
  `core/losses/mm_case_bin.py`、`core/tests/test_dev_folds_and_hybrid.py`、
  `scripts/run_fold_compare.py`、`input/configs/experiments/v11/dev3-{relloss,hybrid}.yaml`
- Owner: twj

### D-0023 — 框架级 bug：验证/评估从未进入 eval 模式（BN train 模式）

- Date: 2026-11-01
- Status: accepted
- Context: 三折对照的交叉验证发现同一 checkpoint 的 fold0 val 指标不一致：
  run 内验证 8–9=2.577 vs post-hoc 推理 1.969。排查（双数据路径比对 →
  train/eval 模式对照）确认根因：`Runner._validate` 从未调用 `model.eval()`、
  也无 no_grad——BN 在验证时使用批统计并用**验证数据污染 running stats**
  （一次 val pass ≈51 批、momentum 0.1 → running stats 几乎完全被 val 数据
  重写并随 checkpoint 保存）。影响范围：**所有 DLEF run 的 in-run val 指标、
  best.pt 选模依据、以及 eval.py 的全部 test 评估记录**（预测依赖批组成、
  非部署口径）；训练本身不受影响；PolypMeasure 侧不受影响（其 evaluate
  显式 .eval()）。
- Decision: ①`Runner._validate` 修复：验证前 `model.eval()`、结束后恢复
  train，并用新的 `policy.eval_context()`（Torch 实现为
  `torch.inference_mode()`，NoOp 为 nullcontext）包裹验证循环；
  ②回归测试 `tests/test_validate_mode.py`（验证期间模式必须为 eval、结束后
  恢复 train）；③所有历史 DLEF val/test 数字标注为「train 模式 BN 口径」，
  最终报告的数字（含 valmerge/shuffle/relloss 的 B/D 成绩）须以修复后的
  eval 模式重新评测。
- Alternatives: 仅做 BN running stats 重估（post-hoc 补救，不改根因）；
  保留旧口径继续对照（组内自洽但非部署口径，且 BN 统计带评估集泄漏）。
- Consequences: 修复后 val/test 数字与历史值不可直接比较（方向：fold0 上
  eval 模式普遍优于 train 模式 0.2–1.8mm）；运行中的六次三折对照放弃已完成的
  2 次并以修复后代码重启；in-run 选模指标同步变为 eval 口径。
- Evidence: 同 checkpoint 双模式对照（train: 8-9=2.38/9-10=3.75；eval:
  1.96/1.92；run 内记录 2.58/2.19）、`dlef/runner.py:_validate`、
  `dlef/policy.py:eval_context`、`adapters/torch/policy.py:eval_context`、
  `tests/test_validate_mode.py`
- Owner: twj

### D-0024 — tools 与 tests 移出 core

- Date: 2026-11-02
- Status: accepted
- Context: `core/` 同时包含运行时、维护 CLI 和测试；`pyproject.toml` 的
  `include = ["core*"]` 会把 `core.tools`/`core.tests` 也当作包内容。项目变大后
  职责不清，也增加打包和导航噪声。
- Decision: `core/tools/` → `tools/`，`core/tests/` → `tests/`；`core/` 只保留
  运行时库。更新 `tools/_bootstrap` 的 ROOT、commands 的 bootstrap import、
  Makefile、测试发现路径、AGENTS/README/TUTORIAL。
- Alternatives: 保持现状；只移动 tests（tools 仍有争议）。
- Consequences: `tools`/`tests` 不再被打进项目包；`make test`/`make check` 使用
  `tests/`；命令行改为 `python tools/...`。结构偏离 DLEF 模板的 `core/tools`，
  但职责边界更清晰。
- Evidence: `tools/`、`tests/`、`Makefile`、`AGENTS.md`；迁移后 52 个测试通过。
- Owner: twj

### D-0025 — tools 回到 core/tools；tests 仍留在项目根

- Date: 2026-11-02
- Status: accepted
- Context: 把 `tools/` 放在项目根后，工具与项目源码分属两个包路径，维护和 import
  边界反而变复杂；这些工具是项目 `core` 的维护入口，不是独立应用。
- Decision: `tools/` 回到 `core/tools/`；`tests/` 继续留在项目根，和 `scripts/`
  并列。更新 `core/commands/*.py` 的 bootstrap import、`Makefile`、工具 ROOT
  路径、测试 import 和文档。
- Alternatives: 保持 `tools/` 在根目录（D-0024，被部分取代）；把 tests 也移回
  `core/tests`（本次不做）。
- Consequences: `core/tools/` 与 `core/commands/` 同属项目包；`tests/` 仍从项目根
  运行。D-0024 中“tools 移出 core”的部分由本条目取代。
- Evidence: `core/tools/`、`tests/`、`Makefile`、`AGENTS.md`、`README.md`
- Owner: twj

### D-0026 — 选模改为 EMA 优先；训练结束保存 final_ema

- Date: 2026-11-02
- Status: accepted
- Context: `v11-ranksim-log` 三个 checkpoint 的 eval 模式实测：`last.pt` 里的
  **ep40 EMA** 全面优于 raw 与早停选出的 best/best_ema（B 1.130 / D 1.073 /
  R 2.379 vs raw 1.266/1.150/2.611）。原因是 valmerge 的验证集只含 4–6mm，
  best/best_ema 在 ep7 就被选中；而 EMA 抹平了余弦退火末期的权重震荡，
  与「输出尺度压缩」的偏差分解结论一致。PM 侧配方本就是 EMA 选模。
- Decision: ①Runner 训练结束总是保存 `final_ema.pt`（末轮 EMA，写入 artifact
  `checkpoint:final_ema`）；②`Run.best_checkpoint_path` 改为 EMA 优先：
  final_ema → best_ema → best → best.json，无 EMA 产物的旧 run 自动回落；
  ③`scripts/plot_experiment.py` 默认绘制 final_ema + best 两条；
  ④回归测试同步 EMA 优先语义。
- Alternatives: 维持 raw 优先（D-0016）——已被本轮实测否定；仅靠 dev 全档 val
  选 EMA（需要重训，收益与 final_ema 重叠）。
- Consequences: eval/queue/chart 在 EMA 存在时默认使用 EMA checkpoint；报告需注明
  用的是 final_ema 还是 best_ema；`best.pt`（raw）仍保留以便对照。
- Evidence: `DLEF/dlef/runner.py`（final_ema 保存）、`DLEF/dlef/run.py`
  （EMA-first 查询）、`DLEF/tests/test_early_stopping.py`、
  `output/reports/v11-ranksim-log_evalmode.json`
- Owner: twj

### D-0027 — 尺寸显式输入（size_input）：让 β 头看见 S 与 D

- Date: 2026-11-02
- Status: accepted
- Context: 误差分解显示 2–3mm 档在两种损失下都系统性高估 2.2–2.9mm，而训练集
  `truth/(D·S)` 在 2–3mm 档比 4–6mm 低约 36%——β 头需要学尺寸相关修正，但
  **S 只作为乘数、从未进入网络输入**，θ 头无法感知「框有多大」。
- Decision: `model.size_input: bool = False`；开启时在 β 头输入加零初始化旁路
  `size_proj([log S − 5, log D + 1])`（2 → 2·d），训练起点与原模型严格等价。
- Alternatives: 改乘数形式；分档独立头（更重、样本更少）。
- Consequences: 新实验 `v11-ranksim-log-size`（log + rank + valmerge + size_input）；
  旧 checkpoint 不含该参数，加载时 strict=False 兼容。
- Evidence: `core/models/v11.py`、`core/config.py`、
  `input/configs/experiments/v11/ranksim-log-size.yaml`
- Owner: twj

### D-0028 — 每个 run 只保留三件套 checkpoint；周期性点不再复制进 artifact

- Date: 2026-11-02
- Status: accepted
- Context: 每 run 曾有 9 个 checkpoint（best/epoch-0005…0040/last），且每个都被
  `save_artifact` **复制进内容寻址存储**，单 run ≈ 3.8 GiB、全库 33 GiB run +
  30 GiB artifacts（系统盘曾因此写满）。
- Decision: ①配置统一 `checkpoint_every: 0`（不再写周期点）；②`save_best_ema`
  默认 False（不再写 best_ema，EMA 由 final_ema 承担）；③保留三件套
  **best.pt（raw 选模）/ final_ema.pt（末轮 EMA）/ last.pt（恢复用）**；
  ④runner 对周期点加 `checkpoint_keep`（默认 3）兜底轮转，且周期点不再入 artifact；
  ⑤已按此清理历史 run：删除 49 个周期点/best_ema 文件 + 40 个 artifact 副本，
  回收 25.7 GiB（manifest 中对应 artifact 条目保留为历史，不再有文件，校验工具不检查）。
- Alternatives: 用 `dlef gc`（其策略按 run 保留、artifact TTL 30 天，dry-run 会
  删除 75 个 run 记录而只回收 0.9 GiB，不适合本目标）。
- Consequences: 历史 run 无法再从中途 epoch 恢复（best/last/final_ema 仍在）；
  新 run 单 run 体积降到约 1.1–1.4 GiB；早期实验的 `epoch-0020` 等中间点已不可用。
- Evidence: `DLEF/dlef/config.py`、`DLEF/dlef/runner.py`、`input/configs/experiments/v11/*.yaml`
- Owner: twj

### D-0029 — 所有评价指标写入 TensorBoard（含 eval）

- Date: 2026-11-02
- Status: accepted
- Context: 训练期逐 epoch 指标（`val/v11/mae_*`、`ema/*`）已由 TensorBoardCallback
  写入，但 `eval.py` 的测试/验证评估只写 `evals/*.json` 与 metrics.jsonl，
  TensorBoard 里看不到 B/D/R/clean 成绩。
- Decision: ①新增 `core/tensorboard.write_eval_scalars`；②`eval.py` 评估完成后
  写 `eval/<split>/<benchmark>/<metric>`（step=0）；③`train.py` 的最终 val 评估
  写 `eval/val/final/<metric>`；④`scripts/backfill_tensorboard.py` 把历史 146 条
  eval 记录回填进各自 run 的 TB 目录；⑤TensorBoard 仍是镜像，metrics.jsonl /
  evals/*.json 才是权威记录。
- Alternatives: 只保留 JSON（用户需要一处查看全部指标）。
- Consequences: `tensorboard --logdir output/runs` 可一次看到训练曲线 + 全部评测档位；
  新 eval 自动带 TB 标量；历史 eval 已完成回填。
- Evidence: `core/tensorboard.py`、`core/commands/eval.py`、`core/commands/train.py`、
  `scripts/backfill_tensorboard.py`
- Owner: twj

### D-0030 — benchmark 指标以逐档曲线写 TensorBoard，并挂到训练 run

- Date: 2026-11-02
- Status: accepted
- Context: 用户要求在 TensorBoard 查看 benchmark（B/D/R/clean）评价指标。首版写成
  `eval/<split>/<bench>/<metric>` 单点，TB 里呈平线；且只写在 `*-eval` 子目录，
  打开训练 run 时看不到。另有三个实验（ranksim-relloss/log/log-size）是离线评测，
  根本没有 eval 记录。
- Decision: ①新增 `write_benchmark_scalars`，逐档写
  `benchmark/<cohort>/<checkpoint>/mae_by_bin`（step=1mm 档序号 → 真曲线），
  汇总指标写 `.../summary/<metric>`；②`eval.py` 同时写入 eval run 与**训练 run**
  的 TB 目录；③`plot_experiment.py`（离线评测）也写同样结构，弥补无 eval 记录的 run；
  ④`scripts/backfill_tensorboard.py` 按 manifest artifacts 的 digest→checkpoint 名，
  把 146 条历史 eval 记录挂回训练 run。
- Alternatives: 只用逐区间 PNG（用户要求统一在 TB 看）；用 step=epoch 把测试点叠到
  val 曲线上（同一 run 多个 checkpoint 会撞 tag）。
- Consequences: `tensorboard --logdir output/runs` 下每个训练 run 都能看到
  `benchmark/B|D|R|clean/<ckpt>/mae_by_bin`；eval 子目录仍保留一份。
- Evidence: `core/tensorboard.py`、`core/commands/eval.py`、
  `scripts/plot_experiment.py`、`scripts/backfill_tensorboard.py`
- Owner: twj

### D-0031 — 逐区间折线图本身也写入 TensorBoard（Images 页 + 自定义布局）

- Date: 2026-11-02
- Status: accepted
- Context: 用户希望「之前画的折线图」在 TensorBoard 里也能看到（TB 标量画不出
  2mm 参考线和四面板布局）。
- Decision: ①新增 `core/tensorboard.write_chart_image`：用 `add_image` 把
  `per_bin_mae.png` 写入 run 的 Images 页（tag `chart/per_bin_mae`），并用
  `add_custom_scalars_multilinechart` 建一个
  「benchmark per-bin MAE（B/D/R/clean）」的 Scalars 布局；②`plot_experiment.py`
  出图后自动写入；③已回填 14 张历史逐区间图。
- Alternatives: 只靠标量曲线（缺 2mm 线与面板布局）。
- Consequences: TB 的 Images 页 = 原图；Scalars 自定义页 = 四 cohort 逐档曲线并排。
- Evidence: `core/tensorboard.py`、`scripts/plot_experiment.py`
- Owner: twj

### D-0032 — 移除 TensorBoard 的 eval/ 标签（保留 train/val/benchmark/图片）

- Date: 2026-11-02
- Status: accepted
- Context: `eval/` 是第一版写入方式（单点汇总），与 `benchmark/summary` 重复，
  用户要求从 TB 中删除。
- Decision: `scripts/prune_tb_eval.py` 重写每个 run 的 TB 事件文件：读全部
  scalars/images → 丢弃 `eval/` 前缀 → 其余按原 step/wall_time 重写 →
  重建 run/config 与 run/manifest 文本 → 替换旧事件文件。共处理 150 个目录，
  保留 3116 个 tag、删除 240 个 `eval/` tag。
- Alternatives: 不重写（TB 无法删单个 tag）；清空整个 TB（会丢训练曲线与 benchmark）。
- Consequences: TB 只剩 train/、val/、benchmark/ 和 Images；权威记录
  （metrics.jsonl / evals/*.json）未受影响；新运行的 `eval.py` 仍会写
  `benchmark/`，不再写 `eval/`（已在 D-0030 中改过写入路径，本脚本只清理历史）。
- Evidence: `scripts/prune_tb_eval.py`
- Owner: twj

### D-0033 — TensorBoard 只保留一张四面板图；用常数曲线画 2mm 参考线

- Date: 2026-11-02
- Status: accepted
- Context: 曾按 cohort 拆成 4 张单面板图写进 Images 页，用户反馈「不如只要一张四面板图」；
  同时 TB 原生只支持标量曲线（y vs step），画不了参考线。
- Decision: ①撤掉单面板 image（代码与已写事件均清理，仅保留 `chart/per_bin_mae`
  四面板图，写图时等比缩到 ≤1600px 宽）；②`write_benchmark_scalars` 额外写
  `benchmark/reference/2mm` 常数曲线，并用 `add_custom_scalars_multilinechart`
  把 B/D/R/clean 的 `mae_by_bin` 与参考线放进同一张 Scalars 图。
- Alternatives: 只保留 PNG（无法在 TB 里看标量）；只用原生标量（无参考线）。
- Consequences: TB Images 页只有一张四面板图；Scalars 自定义页
  「benchmark per-bin MAE」可看到四 cohort + 2mm 线。
- Evidence: `scripts/plot_experiment.py`、`scripts/prune_tb_images.py`、`core/tensorboard.py`
- Owner: twj

### D-0036 — 训练结束自动同步 TensorBoard 并出图

- Date: 2026-11-02
- Status: accepted
- Context: 用户要求「以后实验都要给我 TensorBoard 数据和图片」，只看 TensorBoard。
  此前新 run 需手工 `make tb-sync` + `make chart`，容易遗漏（`rellog` 就因缺链接
  在 TB 里看不到）。
- Decision: `core/commands/train.py` 在主进程训练结束、最终 val 评估之后自动：
  ①`scripts/sync_tb_links.py`（output/tb 符号链接）；②`scripts/plot_experiment.py
  --run <run_dir>`（eval 模式评测 B/D/R/clean → `benchmark/` 标量 + 图 + Images）。
  冒烟 run（`max_steps` 或 `data.max_samples`）跳过；`DLEF_SKIP_POST_REPORT=1` 可关闭。
  失败只打印不阻断训练。
- Alternatives: 留在 Makefile 手工步骤（已证明会被漏掉）。
- Consequences: 每次完整训练多 2–4 分钟（四 cohort eval 推理）；离线评测不写
  test_access 记录（只有 eval.py 写），因此不增加正式 test 访问计数。
- Evidence: `core/commands/train.py`、`scripts/sync_tb_links.py`、`scripts/plot_experiment.py`
- Owner: twj

### D-0039 — 实验性把 R 测试集并入训练（R benchmark 随即失效）

- Date: 2026-11-02
- Status: accepted
- Context: 用户要求在最优配方（log + RankSim γ=2.0 + valmerge）上把 R 测试集
  （REAL-colon，185 帧/87 病灶）并入训练，观察 B/D 是否受益。
- Decision: 新增 `data.extra_train_splits`（白名单仅四个 test split），
  `v11-ranksim-log-g20-r` 设置 `extra_train_splits: test_r`；train 帧数
  2,076 → 2,261（+185）。**R benchmark 自此不可作为留出测试**：同一批帧进入
  训练，R 的任何数字都是训练集内表现，不得用于报告或选模。B/D 不受影响。
- Alternatives: 只用 R 的 train/val 部分（REAL-colon 无官方 split）；不做。
- Consequences: 若后续仍需 R 作为独立测试集，必须新建 v3 benchmark 并排除
  这 87 个病灶；本实验只看 B/D。自动上报仍会算 R——解读时必须标注失效。
- Evidence: `core/config.py`、`core/data/datasets/v11_frames.py`、
  `input/configs/experiments/v11/ranksim-log-g20-r.yaml`
- Owner: twj

### D-0040 — 轻量增广（水平翻转 + 亮度/对比度 + 噪声），只作用于训练集

- Date: 2026-11-02
- Status: accepted
- Context: 训练侧一直是 `train_transform: identity`（零增广）；此前只试过缩放增广
  （`scale_aug`，证伪）。用户提出「对少的病例做数据增强」。
- Decision: 新增 `core/data/transforms/light.py` 与 `data.train_transform: light`：
  ①水平翻转 p=0.5（`S` 为最大边长，翻转不变，无需改 bbox）；②对比度
  `1±0.15`、亮度 `±0.10`（ImageNet 归一化空间）；③高斯噪声 σ=0.02（p=0.3）。
  只作用于 `role="train"`（`make_loader` 按角色选 train/eval transform）。
  首个实验 `v11-ranksim-log-g20-aug`（γ=2.0 + valmerge + light）。
- Alternatives: 按档位加强稀有档增广——本质是重加权，`case_bin` 已证伪，先做均匀版；
  旋转/裁剪——预裁剪视图会引入黑角或破坏与 S 的尺度一致性。
- Consequences: 数据版本标记为 transform=light，与 identity 的 run 不可混算；
  eval/test 仍用 identity，评测口径不变。
- Evidence: `core/data/transforms/light.py`、`core/data/build.py`、`core/config.py`、
  `tests/test_light_transform.py`
- Owner: twj

### D-0041 — TorchModel 缺 eval()：D-0023 的修复对 torch 模型从未生效

- Date: 2026-11-03
- Status: accepted
- Context: 收尾时为 3 个 seed 补官方 eval 记录，发现 eval.py 的 B 成绩是 1.512，
  而同一 checkpoint 的离线 eval 模式测量是 1.066。排查确认：`runner._validate`
  的修复用 `hasattr(model, "eval")` 判断，而 `TorchModel` 只 passthrough
  `parameters/named_parameters/state_dict/load_state_dict`，**没有 train/eval/
  training**，因此验证与评估一直停留在 train 模式（BN 用批统计）。
  D-0023 当时的回归测试用的是带 eval() 的假模型，掩盖了这一点。
- Decision: ①`adapters/torch/components.py` 给 TorchModel 补 `train/eval/training`
  passthrough；②新增 `tests/test_torch_model_mode.py`（断言 BN 在 eval 下用
  running stats）；③用修复后的 eval.py 重新为最优配方 3 seed 生成官方 eval 记录
  并建 claim（数值与离线测量逐位一致：B 1.066/1.150/1.123）。
- Alternatives: 在 runner 里通过 `_inner_module()` 解包（adapter 缺方法时兜底），
  但补 passthrough 更通用。
- Consequences: **2026-11-03 之前的所有 eval.py 记录都是 train 模式 BN**，其数值
  不可用于报告；此前所有结论基于离线 eval 模式测量（net.eval()），不受影响。
  今后 in-run 验证也真正用 eval 模式，早停/选模行为会变（更可靠）。
- Evidence: `DLEF/adapters/torch/components.py`、`tests/test_torch_model_mode.py`、
  `output/reports/report.md`、`output/claims/best-*.json`
- Owner: twj

### D-0042 — 收尾定稿：最优配方冻结，γ 1.0/2.0 不可区分

- Date: 2026-11-03
- Status: accepted
- Context: 模型侧实验共 26 组（见 STATE §4）。γ 调优（0.5/1.0/2.0 + 三 seed 复核）
  显示 B 随 γ 单调改善、D 不敏感，但单 seed 的差异（0.02–0.06）落在 seed 散布
  （B 0.084）内；γ=2.0 三 seed 为 B 1.113 / D 1.111，γ=1.0 单 seed 为
  B 1.130 / D 1.073。
- Decision: ①**冻结配方** = 对数空间损失 + RankSim γ=2.0 + valmerge + shuffle +
  40 轮 + full backbone + EMA 优先选模；以 **3 seed 均值**作为正式基线：
  B 1.113 / D 1.111 / R 2.356。②γ=2.0 与其三 seed 证据保留为基线；γ=1.0 记为
  等价可选（D 略优、B 略差）。③已建 8 个 claim 并生成 `output/reports/report.md`。
- Alternatives: 只用 γ=1.0（需再补 2 seed）；继续调 γ（已证明在噪声内）。
- Consequences: 后续任何改动都以该 3-seed 基线为对照；B 2–3（2.32）、B 7–8（2.71）、
  D 2–3（2.71）三个未达标档归因为**数据缺口**（训练 2–3mm 仅 90 帧；B 7–8 为
  单一扁平病灶），不再靠损失/头/增广类改动推进。
- Evidence: `output/reports/g20_seeds_evalmode.json`、`output/claims/best-*.json`、
  `output/reports/report.md`、STATE §3/§4
- Owner: twj

### D-0043 — BN 统计重估：污染影响很小，不解释小档缺口

- Date: 2026-11-03
- Status: accepted
- Context: 三个最优 seed 都在 eval 模式修复（D-0041）之前训练，验证批更新了 BN
  running stats（实测 `last.pt` 的 `num_batches_tracked=5,640` = 5,200 训练批 +
  440 验证批）。
- Decision: `scripts/bn_recalibrate.py` 固定 `final_ema.pt` 学习参数，用该 run 的
  实际训练成员（含 valmerge 后的 2,076 帧）重新统计 BN，比较逐档误差。
- 结果（seed 20261020）: B `<10mm` 1.066→1.110（+0.044）、D 1.097→1.116（+0.019）、
  R 2.293→2.355（+0.062）；2–3mm：B 2.01→2.10（+0.10）、D 2.74→2.83（+0.09）。
  **污染统计在测试上略好，差异在噪声内；BN 污染不解释 2–3mm 缺口。**
- Consequences: 仍以修复后代码重训（`--tag trusted`）作为可信新基线；旧三个 seed
  的数字保留为"评估口径已修正、BN 统计含验证批"的基线，差异已量化。
- Evidence: `scripts/bn_recalibrate.py`、
  `output/reports/bn_recal_v11-ranksim-log-g20_20261102-182028.json`
- Owner: twj

### D-0044 — 分支诊断：2–3mm 是泛化问题不是拟合问题；修 dev 折分配

- Date: 2026-11-03
- Status: accepted
- Context: 用户/审查要求判断小档误差是「训练没学会」还是「留出泛化不好」，并检查
  dev 三折是否覆盖尾档。
- 结果（`scripts/branch_diagnosis.py`，seed 20261020 final_ema）:
  - **训练集内** 2–3mm：实际融合 MAE **0.09**（β 分支 1.14 / 直接分支 0.26）
    → 模型能拟合自己的小息肉；
  - **B 测试** 2–3mm：融合 2.01（β **3.47** / 直接 **1.73**）→ 泛化失败，且
    β 分支在小档最差、却在 2–3mm 拿到 **0.21** 的门控权重（全表均值 0.15），
    融合因此比纯直接分支更差；
  - **D 测试** 2–3mm：β 2.54 / 直接 2.80 / 融合 2.74（D 上 β 反而更好）。
  - 门控随尺寸单调下降（2–3mm 0.21 → 8–9mm 0.14），方向与小档需求相反。
- Decision: ①确认小档问题属**泛化/来源差异**，不是拟合不足；②**修正 dev 折分配**：
  `assign_dev_folds` 原来按层内 `i % n_folds` 轮转，单病例层（9–10mm 的三个来源各
  1 例）会全部落到 fold 0；改为**全局计数器**轮转，修后 9–10mm 帧分布 0/4/5、
  各折病例 202/202/201；③分支独立监督（Step 3）具备依据：β 分支在小档最差且被
  门控过度信任。
- Alternatives: 不动折分配（尾档无法在 dev 上评估）。
- Consequences: 后续 dev 实验重新按新折分配；此前 dev3 系列的折分配为旧版（尾档集中）。
- Evidence: `scripts/branch_diagnosis.py`、`output/reports/branch_diag_20261102-182028.json`、
  `core/data/datasets/v11_frames.py`
- Owner: twj

### D-0045 — 分支独立监督（预注册），用覆盖小档的 dev 折验收

- Date: 2026-11-03
- Status: accepted（预注册，结果待出）
- Context: 分支诊断（D-0044）显示：训练集内 2–3mm 融合误差仅 0.09（能拟合），
  而 B 测试 2–3mm 为 2.01，其中 **β 分支 3.47 / 直接分支 1.73**，且门控在 2–3mm
  给 β 的权重是**全表最高 0.21**（均值 0.15）——融合在小档过度信任最差的分支。
- Decision: ①模型 `forward(..., return_branches=True)` 暴露 `pred_beta`/`pred_direct`
  到 `aux`；②`mm_rank` 支持
  `L = L_fused + γ·L_rank + w_β·L_β + w_direct·L_direct`（三项同口径 log 损失，
  config: `loss.branch_beta_weight`/`branch_direct_weight`，默认 0，仅 mm_rank 可用）；
  ③对照实验 **dev_folds=3 / fold 0**（val 含 23 帧 2–3mm）：
  `v11-dev3-g20`（无分支监督）vs `v11-dev3-g20-branch`（0.25/0.25）。
- 推进线（预注册）：dev fold0 的 2–3mm MAE 改善 ≥0.5mm，且其余 <10mm 档退化
  ≤0.2mm；通过才在 B/D 上确认并补 seed。
- Alternatives: 直接调门控/后处理（会在 test 上调参，否决）。
- Consequences: 两个 dev3 run 与 valmerge 基线不同训练成员（2/3 dev），只用于
  门控决策；9–10mm 仅 3 个病例，无法在任何单折上验收。
- Evidence: `core/models/v11.py`、`core/losses/mm_rank.py`、`core/config.py`、
  `tests/test_dev_folds_and_hybrid.py`、`input/configs/experiments/v11/dev3-g20*.yaml`
- Owner: twj

### D-0046 — 分支独立监督：未越过预注册闸门，不采纳

- Date: 2026-11-03
- Status: accepted（负结果）
- Context: 按 D-0045 预注册，在 dev fold0（val 721 帧，2–3mm 23 帧）比较
  `v11-dev3-g20`（基线）与 `v11-dev3-g20-branch`（+0.25·L_β +0.25·L_direct）。
- 结果（ep40 EMA，dev fold0）:
  - **2–3mm：1.77 → 1.74（−0.04）**，远未达到「改善 ≥0.5mm」的闸门；
  - 1–2mm：2.22→1.63（−0.59）但该档只有 **1 帧**，是噪声，不计；
  - 其余档变化 −0.12…+0.13，全部在 ±0.2 内（闸门第二条通过）；
  - lt10：1.300→1.286（−0.014）；测试集 B 1.190→1.230（略差）、D 1.142→1.111、
    R 2.502→2.416（略好）——均在 seed 噪声内。
- Decision: **不采纳分支独立监督**；分支路线关闭。
- Consequences: 小档缺口不再由分支/损失/增广类改动推进；剩余手段只有数据
  （company/newdata 域 2–3mm 病例，训练侧现 90 帧）。dev 三折（新折分配）成为
  后续 dev 侧验收的标准工具。
- Evidence: `v11-dev3-g20`、`v11-dev3-g20-branch` 的 `metrics.jsonl`（ep40 ema/…）、
  `output/reports/v11-dev3-g20*_evalmode.json`
- Owner: twj

### D-0047 — 小息肉病例级审计：2–3mm 偏差是几何/标签分布问题，当前训练分布学不到

- Date: 2026-11-03
- Status: accepted
- Context: 按审查建议把 dev fold0 与 B/D 的误差拆到病例，并按来源统计；再用
  `k = truth/(D·S)` 检验「几何先验能否解释小档」。
- 结果（`v11-dev3-g20` final_ema；`scripts/case_error_audit.py`）:
  - **dev fold0 的 2–3mm MAE 1.77 是两种来源的混合**：company 5 例/13 帧
    **2.20**（偏差 +2.20）、newdata 5 例/10 帧 **1.13**（+1.13）；
  - 所有 2–3mm 病例偏差**全为正**、帧内预测标准差中位 **0.12**（最大 0.52）
    → 稳定地高估，不是抖动；
  - 385 个病例的帧内标签跨度全为 0（标签按病例给定，company 的 2mm 档全部恰为 2.0）；
  - D 集 company 2–3mm = **3.04**（6 例），B 集 polyp-size 2–3mm = 2.55（3 例）。
- **机制（关键）**：`k=truth/(D·S)` 在测试小档异常低——B 2–3mm 为 0.0339
  （全局中位 0.0732 的 0.46×）、D 为 0.0312（0.43×）；而训练 2–3mm 为 0.051
  （0.69×）。模型的隐含 k = pred/(D·S) = **0.0757 ≈ β_ref**，即它用的是全局先验。
  D 的 2–3mm 病例 D·S≈64.6、标签 2.00 → 用 β_ref 预测 4.87（实测 5.03）。
- 进一步检验：用训练集拟合 **k(D·S) 曲线**（20 个分位箱、中位 k）做纯几何预测，
  B/D 的 2–3mm 仍为 **2.76 / 3.13**——**不比模型好**。因为 D·S≈65 在训练分布里
  对应 k≈0.077（≈5mm），训练集根本没有「D·S≈65 且真值≈2mm（k≈0.031）」的样本。
- Decision: 结论——**2–3mm 缺口不是模型容量/损失/分支/增广问题，而是输入几何与标签
  关系在测试小档发生了训练集未覆盖的偏移**。下一步按审查建议：①补 dev fold1/2
  验证跨折重复性；②对小档病例做盲法标签复核 + 框宽审计（关键：这些框是否相对
  2mm 病灶过大，或标签是否为整数化名义值）。不为该档继续调模型。
- Evidence: `output/reports/case_audit_20261103-111009.csv`、`scripts/case_error_audit.py`、
  k(D·S) 曲线与纯几何预测（本会话计算，见 STATE §4 第 28 行）
- Owner: twj

### D-0049 — 取景漂移已证实：β 分支几乎完全跟随表观大小，直接分支与门控稳定

- Date: 2026-11-03
- Status: accepted（第 1 步诊断完成；配对训练/背景分支尚未实施）
- Context: 审查提出「同一息肉在远/近帧预测是否漂移」是可绕开标签争议的可检验问题。
  用 `scripts/framing_drift_audit.py` 在 **dev fold0**（该 run 未训练到的折，
  721 帧 / 202 病例）逐帧取 `p/W`、q、β 分支、直接分支、门控、融合输出，
  对每个病例做 `d log(pred)/d log(S)` 的 log-log 斜率（理想 0，1 = 完全跟随表观大小）。
  纳入标准：≥3 帧且 S 的 max/min ≥ 1.10×，共 **91 例**（S 变化中位 **2.18×**）。
- 结果（final_ema，`v11-dev3-g20` fold0）:
  - 融合输出斜率中位 **0.196**（IQR 0.05–0.34），**32%** 病例 > 0.3；
    近远预测绝对差中位 **0.87mm**，**44%** 病例 > 1mm。
  - **β 分支斜率中位 0.906**（几乎 1：完全按表观大小缩放）；
  - **直接分支斜率中位 0.021**（对取景基本不变）；
  - 门控均值 **0.21**（IQR 0.20–0.23）——融合主要走直接分支，这才把最终斜率压到 0.20。
  - 分解 `dlog(pred_β)/dlogS = dlogβ_eff + dlog q + 1`：
    **dlog q/dlog S = −0.25**（应有 −1）、**dlog β_eff/dlog S = +0.16**（应有 0），
    合计 0.91——深度项只提供 25% 的补偿，β 头还略微反向放大。
- 深度读数语义核实：PolypMeasure `align_relative_depth` 要求 `DepthKind.RELATIVE`
  + `DepthConvention.AXIAL_Z`，即**相对、距离式轴向读数**（非逆深度），单位任意、
  逐帧尺度不可比；`q` 是框内中位读数，`D=exp(scalars[6])`。相对深度原则上无法
  给出跨帧可比的距离，因此物理分支按 `β·q·S` 使用它时补偿不足是预期结果。
- Decision: **取景漂移成立，且来源已定位到 β 分支的距离补偿**（不是门控、不是直接分支）。
  下一步按审查建议做两件独立对照：
  ①**配对一致性训练**（`L_pair = SmoothL1(log d_near − log d_far)`，保留尺寸监督；
  优先补「距离变化大且当前仍在漂移」的真实帧，完全留出病例验收）；
  ②**整图 7×7 空间特征 + 框标记**（保留局部病灶分支，比较二者独立收益）。
  预注册推进门槛（审查建议，尚未实测）：远近漂移相对本基线降低 ≥30%，
  2–3mm MAE 改善 ≥0.5mm，其余 <10mm 档退化 ≤0.2mm。
  注意：直接分支「不漂移」不等于「准」——其 2–3mm/大档误差仍高，不能据此关掉物理分支。
- Evidence: `output/reports/framing_drift/{summary.json,per_case.csv,frames.csv,drift_cases.png,drift_slopes.png}`、`scripts/framing_drift_audit.py`、`scripts/plot_framing_drift.py`
- Owner: twj

### D-0050 — 配对训练 A/B 设计定稿：统一诊断变量、β 理想斜率为 b_βW=−(1+b_q)、先只做 β 分支一致性

- Date: 2026-11-03
- Status: accepted（实现完成，待基线队列 fold1/fold2 结束后启动）
- Context: D-0049 的结论被审查收窄三处，本条目固化更正后的口径：
  1. **β 的理想斜率取决于 q**：`d_β = β q S = (βW) q a`（a=p/W），
     尺寸不变要求 `b_(βW) + b_q + 1 = 0`。实测 `b_q=−0.25` → **β 应补 `−0.75`**，
     不是「保持常数」。所谓退化不是"对 S 不敏感"，而是所有病灶输出相近尺寸，
     要靠尺寸监督 + 留出病例评测识别。
  2. **RELATIVE 标记不等于每帧尺度任意**：现役配置加载的是经 SfM 米制监督微调的
     权重，跨帧响应以实测为准；能确认的只是这批临床图像上**组合补偿不足**
     （`b_q=−0.25`、`b_(βW)=+0.16`，合计 +0.91）。
  3. **直接分支中位斜率≈0 但存在个体漂移**：如 company-578_7380 的直接分支
     4.47→6.32mm。归因应为「β 普遍敏感 + 部分病例直接分支也漂」，而非"直接分支稳定"。
- 诊断口径统一（`scripts/framing_drift_audit.py` 已改）：
  - 回归变量改为 **a = p/W**（与图一致），保留 S_px 列；
  - 逐病例分解 `b_pred_β = b_(βW) + b_q + 1`，恒等式残差中位 **0.0**（分解自洽）；
  - 剔除边界截断帧（框距内容边 ≤2px 的帧不参与拟合，不足时回退）；
  - 记录框中心位移，供判断视角变化；
  - 单列 <3mm：91 例中仅 **3 例**（`company-578_856` 斜率 0.66、`newdata-467_1731` 0.22、
    `company-578_662` −0.21）→ 小档的漂移结论证据不足，需补真实远近帧。
- Decision: 第一项训练对照**只增加 β 分支的配对一致性**，两臂除该损失外完全相同：
  - A：当前损失 + 真实远近成对采样（`data.pair_sampling=true`, `loss.pair_beta_weight=0`）
  - B：同 A + `L_pair,β = SmoothL1(log d_β,near − log d_β,far, beta=1)`，权重 **0.5**（预注册）
  - 配对方式：每病例取表观跨度最大/最小的两帧成对，批内相邻；只保留跨度比
    ≥**1.3** 的病例（`data.pair_min_span_ratio`，取景几乎不变的配对无信号）；
    一轮内循环所有对
    直至样本量=行数（步数/epoch 与基线一致，均 96 批）；seed 20261020、fold 0。
  - 预注册门槛沿用 D-0049：漂移 slope 0.196→≤0.14（−30%）、2–3mm −0.5mm、其余档 +≤0.2mm；
    **β 分支变稳定但最终 mm 误差不改善 ⇒ 不接入**。分别报告 β/直接/最终漂移。
  - 背景 7×7 空间特征作为之后的独立对照，不与配对训练混在一起比较。
- Evidence: `core/data/samplers.py`(PairSampler)、`core/losses/mm_rank.py`(pair_beta_weight)、
  `input/configs/experiments/v11/dev3-pair{,-beta}.yaml`、`tests/test_pair_training.py`、
  `output/reports/framing_drift/`
- Owner: twj

### D-0051 — dev 三折 out-of-fold 汇总：2–3mm 失败跨折重复（病例等权 1.81mm）

- Date: 2026-11-03
- Status: accepted
- Context: 审查第 2 步——固定 RankSim+logloss γ=2、40 轮、第 40 轮 EMA 的配方，
  补 `dev_fold=1/2` 两次训练；每折用「没训练过该折病例」的模型预测，把三折并成
  覆盖全部 dev 池的 out-of-fold 预测，并按病例等权汇总。
- Runs: fold0 `20261103-111009_e41415c5_s20261020_f69d8f`、
  fold1 `20261103-132738_b816cad6_s20261020_5aedbc`、
  fold2 `20261103-142236_1534b6c1_s20261020_eae096`（均 completed，tag dev3/step2）
- 结果（605 病例 / 2,249 帧，`scripts/dev3_fold_aggregate.py`）:

  | 档 | 病例 | 帧 | 帧 MAE | 病例等权 MAE |
  |---|---:|---:|---:|---:|
  | 2–3 | 32 | 90 | 1.88 | **1.81** |
  | 3–4 | 124 | 431 | 1.73 | 1.54 |
  | 4–5 | 93 | 272 | 1.28 | 1.03 |
  | 5–6 | 103 | 348 | 0.72 | 0.62 |
  | 6–7 | 102 | 361 | 0.93 | 0.99 |
  | 7–8 | 30 | 159 | 1.23 | 1.21 |
  | 8–9 | 39 | 222 | 1.58 | 1.65 |

  <10mm 病例等权 **1.185**；按来源：company 1.137（437 例）、newdata 1.248（123）、
  phantom 1.548（15）、virtual 1.851（30）。
- Decision: **2–3mm 的失败可跨折重复**（fold0 单折 1.77 → 三折合并 1.81），
  不是某一折的偶然；与 D-0047 的几何/标签分布解释一致。该档的最终定性仍以
  盲法复核（Step 3）为准，模型侧不为该档继续单独调参。
- Evidence: `output/reports/dev3_folds/{oof_frames.csv,oof_cases.csv,oof_summary.json}`
- Owner: twj

### D-0052 — 配对一致性训练结果：β 漂移大幅下降（0.898→0.196），但精度不达标，**不接入**

- Date: 2026-11-03
- Status: accepted（负结果；机制有效，配方需改）
- Context: D-0050 的 A/B 两臂已跑完（各 40 轮、seed 20261020、fold0）：
  A `20261103-151549_c9cb82ee_s20261020_6afc83`（配对采样，无一致性损失）、
  B `20261103-160727_6886b602_s20261020_cfe8d3`（+0.5·L_pair,β）。
- 漂移（dev fold0，91 例，`scripts/framing_drift_audit.py`，a=p/W）：

  | 臂 | 融合 | β | βW | 直接 | 中位\|近远差\| |
  |---|---:|---:|---:|---:|---:|
  | 基线 dev3-g20 | 0.196 | 0.898 | +0.162 | 0.046 | 0.849mm |
  | A 配对采样 | 0.148 | 0.761 | +0.008 | 0.084 | 1.037mm |
  | **B 配对+β一致性** | **0.121** | **0.196** | **−0.532** | −0.030 | 0.781mm |

  **漂移门槛通过**：融合 0.196→0.121（−38%，门槛 −30%）；β 分支从"纯跟随表观大小"
  （0.898）降到 0.196，βW 从 +0.16 变成 −0.53（目标 −0.75）；小档 β 斜率 0.936→0.034
  （仅 3 例）。预测离散度上升（std 1.29→1.55），**没有出现"所有病灶输出相近"的塌缩**。
- 精度（final_ema 离线口径，同一脚本三臂一致；B/D/R 测试 + dev0）：

  | 臂 | B lt10 | D lt10 | R lt10 | dev0 lt10 | B 2–3 | D 2–3 | dev0 2–3 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 基线 | **1.190** | **1.142** | 2.502 | **1.300** | **2.55** | 3.10 | **1.77** |
  | A | 1.507 | 1.311 | 2.362 | 1.529 | 3.54 | 2.98 | 1.62 |
  | B | 1.630 | 1.226 | 2.520 | 1.630 | 3.48 | 3.09 | 1.86 |

  **精度门槛未过**：B 集 <10mm +0.44、dev0 +0.33、B 2–3mm +0.93（门槛是 −0.5）；
  A 与 B 的退化幅度相近，说明**主要损失来自配对采样带来的帧多样性下降**（每轮只看到
  每例 2 帧、重复填充到 96 批 ≈ 512 个不同帧，而基线是 2,044 帧），配对损失本身
  （B vs A）只再添少量代价。
- Decision: 按预注册规则——**β 变稳但最终 mm 误差不改善 ⇒ 不接入配方**。结论保留两条：
  ①β 分支配对一致性确实能教会距离补偿（这是 D-0049 病因的直接验证）；
  ②当前"只采极值帧"的训练方式牺牲了帧多样性，不能作为最终方案。
  下一步只做一个改法（C 臂）：**保留全帧 shuffle（100% 帧覆盖）+ 额外插入配对批**
  （同一对仍批内相邻、一致性损失不变），用于把"漂移收益"和"多样性损失"分开。
  C 臂仍用同一预注册门槛；若精度不达标则停止该路线，转向背景空间特征（7×7）。
- Evidence: `output/reports/framing_drift/{pairA_fold0,pairB_fold0}/`、各 run 的
  `metrics.jsonl`/`logs/train.log`、`scripts/framing_drift_audit.py`
- Owner: twj

### D-0053 — 剔除 ≥10mm 训练帧：小档明显改善、大档回退，worst_bin 反而变差（不直接采纳）

- Date: 2026-11-03
- Status: accepted（混合结果；需第二组折复核或改阈值）
- Context: 主指标是 0–10mm 逐档，训练池里有 354 帧 ≥10mm（76 例；其中恰为
  10.0mm 的 203 帧、15mm 以上 78 帧、最大 25.76）。新增 `data.train_max_truth_mm`
  （只过滤 train，val/test 不变），配置 `dev3-lt10.yaml`，其余同当前最优配方。
- Run: `20261103-172712_5751b9c5_s20261020_30ef38`，训练 1,297 帧（基线 1,528），
  最大训练真值 9.49mm；val 721 帧保持全尺寸分布。
- dev fold0（run 自身 val，帧级）：

  | 档 | 基线 | lt10 |
  |---|---:|---:|
  | 2–3 | 1.774 | **1.434** |
  | 3–4 | 1.712 | **1.210** |
  | 4–5 | 1.197 | **0.710** |
  | 5–6 | 0.584 | 0.737 |
  | 6–7 | 1.054 | 1.243 |
  | 7–8 | 1.197 | **1.676** |
  | 8–9 | 1.969 | **2.508** |
  | <10mm | 1.300 | **1.249** |
  | worst_bin | 2.215 | **2.508** |

- 测试集（final_ema 离线口径，同脚本）：

  | 集 | lt10 基线→lt10 | 2–3 | 7–8 | 8–9 | worst_bin |
  |---|---|---:|---:|---:|---:|
  | B | 1.190→**1.158** | 2.55→**1.97**（−0.58） | 2.72→3.29 | 1.73→2.55 | 2.72→**3.29** |
  | D | 1.142→1.215 | 3.10→2.77 | 0.89→1.51 | 1.97→2.55 | 1.97→**2.55** |
  | R | 2.502→**2.177** | 3.41→3.03 | 0.92→1.77 | 0.68→1.68 | 3.41→**3.03** |

- 漂移（fold0，91 例）：融合 0.196→**0.161**、β 0.898→0.823、βW +0.162→+0.119、
  小档 β 斜率 0.936→**0.442**（n=3）。
- Decision: **不直接采纳**。它把误差从"小档"搬到"大档"：2–3mm 在 B 上 −0.58（过了
  −0.5 门槛）、R 总 <10mm −0.33，但 7–8/8–9 三个测试集一致变差，**worst_bin 在
  B/D 变差 0.57/0.58**，与主指标冲突。可确认的是：**10–15mm 的样本在为上端提供
  标定**，整段拿掉会压缩大尺寸映射。
  下一步（按代价排序）：①阈值改 12 或 15（保留 10–11mm 锚点，只去掉 15mm 以上的
  极端样本 78 帧）——直接检验"是不是巨大型拉坏的"；②改为对 ≥10mm **降权**而不是
  删除；③若 ① 仍是大档换小档，则停止这条线。
- Evidence: `output/runs/beta-regression/v11-dev3-lt10/20261103-172712_5751b9c5_s20261020_30ef38`、
  `output/reports/framing_drift/lt10_fold0/`
- Owner: twj

### D-0054 — 只剔除 >15mm 极端样本：几乎无变化（证明 lt10 的代价来自 10–15mm 锚点）

- Date: 2026-11-03
- Status: accepted（负结果，关闭"过滤训练尺寸"这条线）
- Context: D-0053 提出用阈值 15 区分"巨大型有害"还是"10–15mm 有用"。
  配置 `dev3-lt15.yaml`（`data.train_max_truth_mm=15`）：训练 1,473 帧
  （基线 1,528，只去掉 55 帧 15–26mm，占 3.6%），最大训练真值 14.89。
- Run: `20261103-184249_9bbadcd5_s20261020_4f2fde`（completed）
- dev fold0（帧级）：lt10 **1.291**（基线 1.300）、2–3 **1.720**（1.774）、
  3–4 1.727（1.712）、4–5 1.183（1.197）、6–7 0.999（1.054）、
  7–8 **1.093**（1.197）、8–9 **1.936**（1.969）、worst_bin 1.936（2.215）。
- 测试集（final_ema 离线口径）：

  | 集 | lt10 基线→lt15 | 2–3 | 7–8 | 8–9 |
  |---|---|---:|---:|---:|
  | B | 1.190→1.223 | 2.55→2.55 | 2.72→2.81 | 1.73→1.87 |
  | D | 1.142→**1.126** | 3.10→3.04 | 0.89→1.00 | 1.97→**1.75** |
  | R | 2.502→**2.459** | 3.41→3.35 | 0.92→1.04 | 0.68→0.97 |

- 漂移（fold0，91 例）：融合 0.187（基线 0.196）、β 0.869、βW +0.114、
  小档 β 斜率 0.828——**与基线无实质差异**。
- Decision: **极端样本不是问题**。去掉 15–26mm 只有噪声级变化；D-0053 里
  lt10 的大档回退来自**去掉 10–15mm 锚点**（尤其恰好 10.0mm 的 203 帧），
  不是巨大型样本。**关闭"过滤训练尺寸"路线**，默认 `train_max_truth_mm=0`。
  小档若还要继续改善，只能走别的机制（配对一致性、背景上下文），不能靠删数据。
- Evidence: `output/runs/beta-regression/v11-dev3-lt15/20261103-184249_9bbadcd5_s20261020_4f2fde`、
  `output/reports/framing_drift/lt15_fold0/`
- Owner: twj

### D-0055 — 7×7 背景空间分支：β 补偿明显改善（0.898→0.472），但门控放大残余漂移、精度全面退化（不采纳）

- Date: 2026-11-03
- Status: accepted（负结果；机制部分成立）
- Context: D-0049 显示 β 缺距离线索；D-0052 的配对一致性证明 β 可被训练去补偿。
  本条目实现审查第 3 步：整图分支保留 ResNet50 **2048×7×7** 网格（avgpool→Identity，
  非空间路径用 mean 还原，已验证旧 checkpoint 结果不变），加 49 维位置编码、
  每格 `[框内, Δx, Δy, 框宽, 框高]` 框关系（由 S 与几何标量算出）、
  以 ROI token 为 query 的交叉注意力；零初始化注入 β 头。
  配置开关 `model.spatial_context`（默认 false），新增参数约 15 万。
- Run: `20261103-201244_62583ba8_s20261020_214580`（fold0，40 轮，其余同 dev3-g20）
- 漂移（fold0，91 例）：

  | 指标 | 基线 | ctx |
  |---|---:|---:|
  | 融合斜率 | 0.196 | **0.314** |
  | β 斜率 | 0.898 | **0.472** |
  | βW 斜率 | +0.162 | **−0.172** |
  | 直接分支 | 0.046 | 0.000 |
  | 中位\|近远差\| | 0.849mm | **1.655mm** |
  | 小档 β 斜率 | 0.936 | **0.178** |
  | 门控均值 | 0.21 | **0.71** |

  **关键机制**：β 的距离补偿确实改善了（βW 朝 −0.75 走了一半），但门控因此从 0.21
  开到 0.71，把 β 剩余的 0.47 漂移放大进最终输出——融合反而从 0.196 升到 0.314。
- 精度（final_ema 离线口径）：

  | 集 | lt10 基线→ctx | 2–3 | 7–8 | 8–9 |
  |---|---|---:|---:|---:|
  | B | 1.190→**1.451** | 2.55→3.07 | 2.72→3.02 | 1.73→**1.43** |
  | D | 1.142→**1.323** | 3.10→3.31 | 0.89→1.67 | 1.97→2.00 |
  | R | 2.502→**2.604** | 3.41→3.12 | 0.92→2.12 | 0.68→1.48 |
  | dev0 | 1.300→**1.453** | **1.77→1.44** | 1.20→1.52 | 1.97→**1.86** |

- Decision: **不采纳**。两条门槛都未过：漂移变差（0.314 vs 0.196），三个测试集的
  <10mm 全面退化（+0.26/+0.18/+0.10），只有 dev0 的 2–3mm 与 B 的 8–9 改善。
  结论保留：**空间上下文确实给到了 β 距离信息**（βW 从 +0.16 走到 −0.17、小档 β
  漂移 0.94→0.18），但单靠它不够，且会通过门控放大残余误差。
  唯一还没试过的组合是：**ctx + β 一致性**，且用"全帧 shuffle + 额外配对批"
  （C 臂，见 D-0052）避免 D-0052 的帧多样性损失。若仍不达标，则这条路线整体关闭。
- Evidence: `output/runs/beta-regression/v11-dev3-ctx/20261103-201244_62583ba8_s20261020_214580`、
  `output/reports/framing_drift/ctx_fold0/`、`tests/test_spatial_context.py`
- Owner: twj

### D-0056 — ctx + β 一致性 + 混合采样：仍不达标，距离补偿路线整体关闭

- Date: 2026-11-03
- Status: accepted（负结果；关闭 D-0049–D-0056 这条线）
- Context: D-0052 证明配对一致性能把 β 漂移压到 0.196，但纯配对采样丢帧；
  D-0055 证明 7×7 背景分支能改善 β 补偿（βW +0.16→−0.17），但门控放大残余漂移。
  本条目把两者组合，并用混合采样（每轮 16 个配对批 + 全部 1,528 帧 shuffle，
  112 批/epoch）保住帧覆盖。配置 `dev3-ctx-pair.yaml`：
  `spatial_context=true` + `pair_beta_weight=0.5` + `pair_batches=16` + 跨度比 ≥1.3。
- Run: `20261103-212457_0e1a0599_s20261020_2cf5ef`（fold0，40 轮）
- 漂移（fold0，91 例）：

  | 指标 | 基线 | ctx | **ctx+pair** |
  |---|---:|---:|---:|
  | 融合斜率 | **0.196** | 0.314 | 0.303 |
  | β 斜率 | 0.898 | 0.472 | **0.450** |
  | βW 斜率 | +0.162 | −0.172 | **−0.224** |
  | 中位\|近远差\| | **0.849mm** | 1.655 | 1.817 |
  | 门控均值 | 0.21 | 0.71 | **0.73** |

- 精度（final_ema 离线口径）：

  | 集 | lt10 基线→ctx+pair | 2–3 | 7–8 | 8–9 |
  |---|---|---:|---:|---:|
  | B | 1.190→**1.581** | 2.55→**4.06** | 2.72→2.38 | 1.73→**0.89** |
  | D | 1.142→**1.274** | **3.10→2.49** | 0.89→1.39 | 1.97→2.19 |
  | R | 2.502→**2.845** | 3.41→3.45 | 0.92→3.03 | 0.68→0.97 |
  | dev0 | 1.300→**1.407** | **1.77→1.29** | 1.20→1.42 | 1.97→1.79 |

- Decision: **不采纳，并关闭该路线**。原因：
  ① 与 D-0052 的纯配对训练相比，配对批只占 14%（16/112），β 一致性信号弱得多，
     β 斜率只从 0.472 再降到 0.450，远没到 D-0052 的 0.196；
  ② 门控涨到 0.73（ctx 让它更信任 β），把 β 残余漂移放大，融合反而 0.303；
  ③ 精度三个测试集全退化（+0.39/+0.13/+0.34），2–3mm 只有 D 改善、B 反而恶化。
  **结论：β 补偿改善不等价于最终精度改善；在门控被学习的情况下，β 单独变好会把
  输出推向它、放大残余误差。** 这条路线（配对一致性 / 背景空间特征 / 二者组合）
  到此关闭；默认配置保持 `spatial_context=false`、`pair_sampling=false`。
- Evidence: `output/runs/beta-regression/v11-dev3-ctx-pair/20261103-212457_0e1a0599_s20261020_2cf5ef`、
  `output/reports/framing_drift/ctxpair_fold0/`
- Owner: twj

### D-0057 — 用户重划分的公司数据集（company-v1）与 ctx 从零训练：冻结 test 上 <10mm 病例等权 1.146

- Date: 2026-11-04
- Status: accepted（新数据版本；独立评估轨道，不与 B/D/R 可比）
- Context: 用户把公司数据集按病例重新划分为 train/val/test（≤10mm，70/15/15，
  1inch 分层），要求用它**从零**训练 ctx 架构并在其 test 划分上评估。
  原 v11 特征与旧清单绑定，必须重建。
- 数据构建（`scripts/build_company_features.py`、`scripts/build_company_manifest.py`）：
  - 框：用数据集自带的 `visioncheck_index.csv`（原图坐标），换算到内容裁剪坐标；
    **不重跑检测器**——实测在裁剪图上重检约 20% 帧丢框。
  - 深度：现役 `polyp_size_delivery_dav2.yaml`（DA-V2 metric-SfM 全网络微调），
    与老特征同源；每帧同时把伪彩深度图写到该病例 `visioncheck/<stem>_depth.jpg`。
  - 结果：**12,872 / 12,888 帧成功**，16 帧失败（病例 467_1421 的框落在内容区外）。
  - 清单 `curated/v11_frames/company-v1`：train 9,221 帧/829 例、val 2,093/182、
    test_company 1,558/179；图像经 `raw/company-v1` 软链引用。
  - **β_ref 重算 = 0.016986**（旧 0.0754 不可用：新图分辨率更高，S 以像素计）。
  - 冻结 benchmark `v11-company-test-v1`，benchmark_id `sha256:a1d74723…`。
- 训练：`v11-company-ctx`（ctx 架构、`init_checkpoint=""`、15 轮、seed 20261020、
  RankSim γ=2 + log 损失）。15 轮 ≈ 8,655 步（新集 577 批/轮）。
  Run `20261103-233352_78216d2e_s20261020_24fb66`（completed）。
- 结果（官方 eval.py，冻结 benchmark，1,558 帧）：

  | 指标 | 值 |
  |---|---:|
  | 帧级 <10mm MAE | **1.262** |
  | 帧级 MAE（全部） | 1.346 |
  | worst_bin（帧级） | 2.169（9–10 档） |
  | **病例等权 <10mm MAE** | **1.146** |
  | 病例等权 2–3mm | 1.86（9 例） |
  | 病例等权 3–4 / 4–5 / 5–6 | 1.14 / 0.97 / 0.78 |
  | 病例等权 6–7 / 7–8 | 1.20 / 0.81 |
  | 病例等权 8–9 / 9–10 | 2.02（12 例）/ 1.74（4 例） |

- Caveat: 训练集标签 ≤10mm，test 含 12 例 ≥10mm（未参与训练、也不在 <10 指标内）；
  该数据集与旧 single-v2 的划分/标签口径不同，**任何数字不得与 B/D/R 或旧 D 集混比**。
- 待办：要判断 ctx 在新数据上是否有效，需在同一 company-v1 上补**非 ctx 对照**
  （同种子同轮数）；目前只有绝对成绩，没有因果对照。
- Evidence: `output/runs/beta-regression/v11-company-ctx/20261103-233352_78216d2e_s20261020_24fb66`、
  `...-eval/20261104-083637_78216d2e_s20261020_60d09d`、`input/benchmarks/v11-company-test-v1.json`
- Owner: twj

### D-0058 — 裁剪去 OSD 修复 + 清单固化 crop_box：company-v1 特征重建、旧 ctx 成绩作废

- Date: 2026-11-04
- Status: accepted（数据侧缺陷修复；已重建 company-v1）
- Context: 用户发现 467_771_8.0mm 的深度图里带 OSD。排查确认不是深度模型问题：
  `detect_image_area` 用 **gray>12 最大连通域**，当 OSD 面板背景不是纯黑（灰度 20–40）
  时整幅图连成一个组件，裁剪退化为整图，**OSD 进入 RGB 与深度输入**。
  抽样：新 company-v1 随机 800 帧中 **6.0%（48 帧）** 受影响；旧 single-v2 公司帧
  300 抽样中 **10.0%（30 帧）** 受影响（历史数字同样带此污染，未回补）。
- 修复 ①（裁剪）：新增 `robust_content_box` —— **HSV 饱和度 >40 的最大连通域**，
  失败再退回原灰度法。600 抽样零失败；467_771 由 (0,0,1920,1080) 变为
  (655,0,1265,1080)，OSD 被切掉。
- 修复 ②（坐标系）：重建后训练直接崩在 `ZeroDivisionError: size/h`。原因：清单里的
  bbox 按新裁剪算，而数据集 `__getitem__` 仍用旧灰度法**重新裁剪**，两边原点不同，
  部分帧 ROI 裁成空图。修复：清单**逐帧固化 `crop_box`**（12,872/12,872 行），
  数据集优先使用清单裁剪，缺省才回退旧函数（旧 single-v2 行为不变）。
- 重建结果：特征 **12,872 帧**（train 9,221/829 例、val 2,094/182、test_company 1,557/179），
  15 帧失败（检测框落在内镜内容区外，多为 OSD 上的误检框，占比 0.12%）；
  **β_ref 重算 = 0.016903**；深度图后缀区分模型：`_depth_da.jpg`（DA-V2）与
  `_depth_endo.jpg`（EndoStreamDepth，metric mm）。
- 冻结 benchmark `v11-company-test-v1` 重建：membership_digest
  `0dae570009236a88`（1,557 帧），已与清单逐项核对一致；旧文件因重建失效且从未被
  claim 引用。
- 完整性核验（全部通过）：无病例跨 split、无重复 sample_id、bbox 全在裁剪内且无退化框、
  特征/图像零缺失、数据集抽样访问 305 帧零异常。
- **作废**：`20261103-233352_78216d2e_s20261020_24fb66`（v11-company-ctx）与其 eval
  （病例等权 1.146）基于污染特征与旧裁剪，**不得再引用**；base/ctx 均需在重建数据上重训。
- 数据源不一致（非本项目 bug，需用户确认）：`manifest.csv` 列 831 个 train 病例，
  磁盘只有 829 —— `467_803_7.5mm`、`68_586_2.0mm` 两个文件夹不存在，无法训练/评测。
- Evidence: `scripts/build_company_features.py`、`scripts/build_company_manifest.py`、
  `scripts/build_endostream_depth_images.py`、`core/data/datasets/v11_frames.py`、
  `input/benchmarks/v11-company-test-v1.json`
- Owner: twj

### D-0059 — 逐档图脚本按数据版本选队列：company-v1 不再默认评旧 B/D/R/clean

- Date: 2026-11-04
- Status: accepted
- Context: 训练结束自动生成的 `chart/per_bin_mae` 由 `scripts/plot_experiment.py` 产出，
  此前它**硬编码**读取 `single-v2` 清单并对 d/b/r/clean 四个旧 cohort 打分。
  对 company-v1 训练的模型来说：①这些不是它的测试集；②D/clean 与新训练集高度重叠
  （209/356、137/206 帧），数字严重偏乐观（D 0.649）；③旧特征裁剪口径也不同。
- Decision: ①`plot_per_bin` 增加 `company` 队列（标签「公司（新划分 company-v1）」）；
  ②`plot_experiment.measure` 按 cohort 选清单：`company` 用 run 自己的
  `data.version`，旧 d/b/r/clean 固定 single-v2；③对非 single-v2 的 run，默认队列
  改为**仅 company**（显式 `--cohorts` 仍可评旧集）。
- 已重生成 `v11-company-base` 的图：final_ema <10mm **1.107**（best.pt 1.401），
  与官方 eval 一致。
- Evidence: `scripts/plot_experiment.py`、`scripts/plot_per_bin.py`、
  `output/charts/v11-company-base.png`
- Owner: twj

### D-0060 — company-v1 上 base vs ctx 对照：ctx 只在 R 上更好，公司/B 都退化（不采纳 ctx）

- Date: 2026-11-04
- Status: accepted
- Context: 在干净重建的 company-v1（crop_box 固化、12,872 帧）上，同数据、同种子
  (20261020)、同 15 轮、同损失，只差 `spatial_context`：
  base `20261104-103104_d8f11691_s20261020_035a6f`、
  ctx  `20261104-123141_10b1b0ea_s20261020_c93f60`（均 completed）。
- 公司 test（冻结 benchmark，官方 eval.py，1,557 帧）：

  | 档 | base | ctx |
  |---|---:|---:|
  | 2–3 | 1.97 | 1.97 |
  | 3–4 | **1.22** | 1.39 |
  | 4–5 | **0.98** | 1.11 |
  | 5–6 | **0.81** | 0.85 |
  | 6–7 | **1.01** | 1.21 |
  | 7–8 | **0.76** | 1.08 |
  | 8–9 | **1.62** | 1.88 |
  | 9–10 | 2.48 | **2.01** |
  | **<10mm** | **1.107** | 1.259 |
  | 病例等权 <10 | **1.086** | 1.156 |
  | worst_bin | 2.482 | **2.014** |

- 跨数据集（旧 single-v2 特征，零图像重叠；离线推理）：
  - PolypSize B：base **1.062** vs ctx 1.295（ctx 差）
  - REAL-colon R：base 1.959 vs ctx **1.523**（ctx 好，2–5mm 全线更低）
- Decision: **ctx 不采纳**。三套数据里两套退化（公司 +0.15、B +0.23），只有 R 改善
  （−0.44，且 2–5mm 一致变好，不太像噪声）。公司数据上 7×7 背景分支没有带来净收益，
  与 D-0055（旧数据上 ctx 融合漂移变差）方向一致。
  记录：company-v1 上目前最好的是 **base**：公司 test <10mm 1.107（病例等权 1.086）、
  B 1.062、R 1.959。单 seed，未补 seed。
- Evidence: `output/runs/beta-regression/v11-company-{base,ctx}/...`、
  `v11-company-{base,ctx}-eval/...`、`output/charts/v11-company-{base,ctx}.png`
- Owner: twj

### D-0061 — benchmark2：公司队列换成 company-v1 新划分，B/R 不变；作为公司模型的默认出图口径

- Date: 2026-11-04
- Status: accepted
- Context: 旧 D/clean 与新 company-v1 训练集高度重叠（209/356、137/206 帧），不能再用；
  但 B/R 与公司数据零重叠，仍然有效。需要一个统一的"公司模型评测套件"。
- Decision: 新增 `input/benchmarks/benchmark2.json`：
  `company → v11-company-test-v1`、`b → v11-b-test-v2`、`r → v11-r-test-v2`，
  **旧 D/clean 由 company 取代**（clean 是 D 子集，随 D 一起退出）。
  `plot_experiment.py`：①支持 `--suite <json>`；②对 `data.version != single-v2`
  的 run **默认套用 benchmark2** 出图并写 TensorBoard。
- 已重生成两张图（TB 里 `chart/per_bin_mae`，final_ema / best.pt 两条）：

  | run | company | B | R |
  |---|---:|---:|---:|
  | company-base final_ema | **1.107** | **1.062** | 1.959 |
  | company-ctx final_ema | 1.259 | 1.295 | **1.523** |

  注意：B/R 仍用旧 single-v2 特征（零重叠、但裁剪口径是旧的），与之前的跨数据集数字一致。
- Evidence: `input/benchmarks/benchmark2.json`、`scripts/plot_experiment.py`、
  `output/charts/v11-company-{base,ctx}.png`
- Owner: twj

### D-0063 — ctx 代码审查：修复 `_box_features` 对 S 的隐式依赖；记录 base/ctx 的 β_ref 差异

- Date: 2026-11-04
- Status: accepted（实现修复 + 实验纯度说明）
- 审查结论（`core/models/v11.py`：`_box_features` / `_context_vector`）：
  1. **已修复**：`_box_features` 原先用 `W = S/a` 反推内容宽度，再把框映射到 224 画布。
     这在 `span_calibration` 开启时会被静默破坏（S 被重标定，W 失真，框位置/掩码全错）。
     改为只用宽高比：`wpx = 224 / max(H/W, 1)`，与 S 无关。实测 64 帧公司样本：
     新旧特征 **max |Δ| = 1.2e-7**、inside 掩码完全一致 → 不影响已训练 checkpoint 的结果。
  2. 其余检查通过：avgpool→Identity 后 `flatten(1)` 的还原顺序正确（通道主序）；
     view 0 取整图、query 取 ROI 视图；49 个 token 与 7×7 位置编码行主序一致；
     零初始化 `ctx_inject` 保证训练起点等价基线；`feats` 外部传入时正确报错。
- 实验纯度问题（记录，不改结果）：`v11-company-base` 与 `v11-company-ctx` 的 resolved
  config 有两处差异——`model.spatial_context` 和 **`model.beta_ref`（0.016986 vs 0.016903）**。
  原因：两次训练之间把配置更新成了重建清单后的 β_ref。0.5% 的 β 先验差异对结果可忽略
  （两者差距 0.15mm），但严格说不是单变量对比；若要完全干净，可用当前配置重跑 base。
- 设计观察（非 bug）：①只有 β 头接收上下文、门控不变，这与"β 变准 → 门控开大 →
  残余漂移被放大"的观测一致；②7×7 网格在 224 输入上每格约 1/7 内容宽，对 2mm 病灶偏粗，
  位置信息靠框特征补；③无 dropout，9k 帧规模下有轻微过拟合风险。
- Evidence: `core/models/v11.py`、`tests/test_spatial_context.py`（71 全过）
- Owner: twj

### D-0064 — 补充 10–15mm 病例微调（warm start 自 company-base）：大档小改善、中档退化，不采纳

- Date: 2026-11-04
- Status: accepted（负结果；补充数据本身保留在 company-v1-big）
- Context: 用户在 dataset 的 train 里新增 **80 例 / 1,893 帧、11–15mm** 病例。
  已生成 DA 特征并合并为 `company-v1-big`（train 11,114 = 9,221+1,893；val/test 不变）。
  训练从 `company-base` 的 `final_ema` **warm start**（校验：0 缺失键、0 张量差异、
  val lt10 复算 1.3556 与 base 一致），5 轮、lr 1e-4/1e-5、warmup 2、seed 20261020。
  Run `20261104-152948_e476958f_s20261020_ba6f53`（completed）。
- 顺带修复两处代码问题：
  ①`build_v11` 的 `init_checkpoint` 用 `ck["model"]` 读取，但 DLEF checkpoint 是对象，
     任何非空 init_checkpoint 都会 `TypeError`（此前从未启用过该字段）；已兼容两种格式。
  ②`plot_experiment` 里 suite 文件名复用了变量 `name`，导致图/JSON 被命名成
     `benchmark2-big.json.png`；已改为独立变量。
- 结果（benchmark2-big，final_ema）：

  | 指标 | base | **base-big** |
  |---|---:|---:|
  | 公司 test 帧级 <10 | **1.107** | 1.201 |
  | 公司 test 病例等权 <10 | **1.086** | 1.147 |
  | PolypSize B | **1.062** | 1.184 |
  | REAL-colon R | **1.959** | 2.336 |

  逐档（公司 test，帧级）：8–9 **1.62→1.50**、9–10 **2.48→2.38** 有改善；
  但 3–4 1.22→1.45、4–5 0.98→1.14、7–8 0.76→1.01 明显退化。
  病例等权下 7–8 反而 0.80→0.64、8–9 1.97→1.84。
- Decision: **不采纳**。补充大息肉把预测整体往上拉：大档略好，中档被牺牲，三套数据
  总量都变差。若还要走这条路，需要更保守的微调（1–2 轮 / lr 3e-5）或只对新病例
  加权，而不是把并集完整再训 5 轮。当前最佳仍是 `company-base`（1.107 / 1.062 / 1.959）。
- Evidence: `output/runs/beta-regression/v11-company-base-big/20261104-152948_e476958f_s20261020_ba6f53`、
  `output/charts/v11-company-base-big.png`、`output/reports/v11-company-base-big_evalmode.json`
- Owner: twj

### D-0065 — EndoStreamDepth 版非 ctx base：与 DA-V2 基本持平（略差），深度源不是瓶颈

- Date: 2026-11-04
- Status: accepted
- Context: 用 EndoStreamDepth（metric mm，`iter_15001 (1).pth`）重建 company-v1-endo 特征
  （14,765 帧，17 帧框越界丢弃），并给 B/R 生成同源 Endo 特征（395 帧，0 失败，
  清单 `b-r-endo`），保证训练/评测深度同源。训练配置与 `company-base` 完全一致：
  非 ctx、从零、15 轮、seed 20261020、RankSim γ=2；训练集 `train_max_truth_mm=10.5`
  （保留 10.0mm、排除新增 11–15mm，共 9,221 帧）；β_ref 重算 **0.000769**
  （Endo 的 q 是毫米，与 DA 的 0.0169 不同量级）。
- Run: `20261104-171224_f8044938_s20261020_d08721`（completed）
- 结果（final_ema）：

  | 指标 | DA base | **Endo base** |
  |---|---:|---:|
  | 公司 test 帧级 <10 | **1.107** | 1.137 |
  | 公司 test 病例等权 <10 | **1.086** | 1.100 |
  | PolypSize B（同源深度） | **1.062** | 1.111 |
  | REAL-colon R（同源深度） | **1.959** | 2.108 |

  逐档（公司 test，帧级）：5–6 Endo 更好（0.77 vs 0.81），其余档差 0.02–0.08。
- Decision: **两个深度模型基本等价**（差异在单 seed 噪声内，Endo 系统性略差 0.03–0.15）。
  说明当前瓶颈不是深度源：DA-V2 与 EndoStreamDepth 的架构/训练方式差异很大，
  但模型表现几乎一致 → 限制来自标签/几何分布与融合机制，而不是深度估计质量。
  默认继续用 DA（更快、特征已齐）；Endo 线可作为对照保留。
- Evidence: `output/runs/beta-regression/v11-company-base-endo/20261104-171224_f8044938_s20261020_d08721`、
  `output/charts/v11-company-base-endo.png`（benchmark2-endo 三面板）
- Owner: twj

### D-0066 — 过夜队列预登记：company-base 三 seed 噪声底 + 小档审计判定 + 单调标定诊断

- Date: 2026-11-04
- Status: pre-registered（结果落在 `output/reports/overnight_20261104.md`；判定后另开 D-0067 记录，本条不改写）
- Context: D-0060（ctx 否决）、D-0064（big 否决）、D-0065（endo 等价）三个结论全部建立在
  **单 seed**、0.03–0.15mm 的差值上，而项目从未测过 company-base 的 seed 噪声底。
  在补噪声底之前，任何 <0.2mm 的"改善/退化"都不能当结论。同时 D-0047 第③步（小档
  标签/框约定审计）一直没做，而 R 的 1–2mm、2–3mm 是当前唯一"全 seed 意义下真失败"的档。
- 队列（`scripts/run_overnight_20261104.sh` + `scripts/run_overnight_tail_20261104.sh`，
  串行、不与训练争 GPU）：
  1. seed 20261021、20261022 各训练一次 `input/configs/experiments/v11/company-base.yaml`
     （`--tag d0066`，除 seed 外与 base 完全一致），各跑一次 `--split test
     --benchmark input/benchmarks/v11-company-test-v1.json` 官方 eval；
  2. `scripts/plot_experiment.py --runs <21>,<22>,<base> --suite benchmark2.json
     --name v11-company-base-3seed`（三 seed 均值 + min/max 带）；
  3. `scripts/summarize_overnight.py` 汇总成 `output/reports/overnight_20261104.md`；
  4. `scripts/bias_recal_diagnosis.py`（三 run 各一次）+ `scripts/summarize_recal.py`
     追加为报告第 5 节。
- 判据（**先登记，后看数**）：
  - **A 噪声底**：以三 seed 的**全距**（max−min）为噪声尺度。任何变体（ctx/big/endo）与
    base 的差 |Δ| ≤ 该全距 → **判为不可区分**，不得写成"更好/更差"；只有 |Δ| > 全距才允许
    下结论。若全距本身 >0.2mm，则 D-0060/D-0064/D-0065 的措辞都要在 STATE 里降级为
    "单 seed 观察，未达噪声底"。
  - **B 小档（继承 D-0047 第③步）**：`k = truth/(D·S)`。小档 `k_ratio_vs_train ≤0.6` →
    标注/框约定不同，模型侧修不了；k 相近但 `bias_mean` 单向且 ≥0.5mm → 域偏移/模型问题。
  - **C 标定**：只在 `company-v1/val` 上拟合（isotonic 与 log-linear 两种），test 仅套用；
    val 侧数字用**病例级 2 折交叉拟合**。test 逐档下降且超标档减少 → 压缩偏差可用 val 修正；
    val 改善而 test 不改善 → 偏差曲线跨队列不稳定。任何情况下 Spearman 与档内 pred std
    **不变**，标定收益不得写成测量能力提升。
- 已完成部分（审计，`scripts/smallbin_audit.py`，base run `final_ema.pt`，与官方数字逐项吻合）：
  - 队列级 <10mm：company_test 1.107（病例 1.086，最差档 2.482）、B 1.062（最差档 3.388）、
    R 1.959（病例 2.022，最差档 3.262）、company_train 0.145。
  - **档内真值几乎是常数**（company/R 标签为整数 mm，档内 std=0.00），而预测均值只从
    3.95（2–3mm 档）走到 6.54（9–10mm 档），真值却从 2 走到 9 → **严重向均值压缩**；
    逐档 MAE ≈ |偏差均值|（company 2–3：MAE 1.97 / 偏差 +1.95；9–10：2.48 / −2.46；
    R 1–2：3.26 / +3.26）→ 误差几乎全是系统性偏置，不是随机噪声。
  - **B 判据未触发**：R 的 2–3mm `k_ratio_vs_train = 1.96×`、3–4mm 2.35×、B 各档 2.2–8.4×，
    全都**高于** company_train（不是 ≤0.6），即 β 分支的几何标度 k 在队列间本就不通用；
    这也解释了 B/R 上门控几乎完全走 direct 分支（wg≈0.10–0.19）。
  - `company_train` **没有 1–2mm 档**（最小档为 2–3mm，264 帧/43 例），R 的 1–2mm
    （21 帧/14 例）在训练分布之外；R 小档 51 例的偏差**全为正**（+0.5…+4.5mm），
    预测存在 ~3–5mm 的硬地板。
  - company 9–10（59 帧/**4 例**）与 B 7–8（5 帧/**1 例**）虽超 2mm，但病例数极少 →
    属噪声档，不作为优化目标；R 1–2（21/14）与 2–3（34/18）是真失败档。
- Evidence: `output/reports/smallbin_audit/{summary.md,summary.json,frames.csv,cases.csv}`、
  `output/review/lt3mm_r/`（R 小档 96 帧/51 例盲审联系表）、
  `output/logs/overnight_20261104{.log,.steps.log,.tail.log}`、
  `output/reports/overnight_20261104.md`、`output/reports/recal_diagnosis/`
- Owner: twj

### D-0067 — 输出侧解压缩两臂（门控下界 / direct 分支几何锚定）：预登记

- Date: 2026-11-05
- Status: pre-registered
- Context: D-0066 的小档审计把 company 的失败定位成**一个机制**——向均值压缩：
  逐档偏差从 2–3mm 的 **+1.95** 单调走到 9–10mm 的 **−2.46**（交叉点 5–6mm），
  预测均值只从 3.95 走到 6.54，而真值从 2 走到 9；档内真值几乎是常数（标签整数 mm，
  std=0.00），逐档 MAE ≈ |偏差均值| → 误差几乎全是系统性偏置。8–9mm 档的几何标度
  自洽（`k_ratio_vs_train` 1.09×，而 9–10 是 3.55×）却仍偏 −1.56 → 问题在**输出侧**，
  不在输入/特征侧。机制上：门控收敛到 wg=0.13–0.26（初值 0.88），即 **74–87% 的输出
  来自没有几何锚定的 `softplus(mm)` direct 分支**，而 β 分支（`β·D·S`，有量纲、
  天然可外推）几乎被关掉。已否决的输入侧路线不再重提（ctx D-0060、big D-0064、
  endo D-0065、size_input D-0027、分支监督 D-0045、增广 D-0040、span_calibration D-0019）。
- Decision: 加**两个互斥的单变量臂**，都只改前向、**不新增任何参数**，关闭时与旧模型
  逐位一致（`tests/test_decompression_arms.py` 断言 `state_dict()` 键相同 + 位级相等，
  旧 checkpoint 与在跑的队列不受影响）：
  - **A `model.gate_beta_floor: 0.5`**（`v11-company-gatefloor`）：
    `wg = floor + (1−floor)·sigmoid(logit)` → wg∈[0.5,1)，强制至少一半权重走有量纲锚定的
    β 分支，把"大尺寸靠几何外推"的能力放出来。
  - **B `model.direct_scale_anchor: true`**（`v11-company-diranchor`）：
    direct 分支从"直接回归绝对 mm"改成**对数空间无量纲乘性修正**
    `pred_dir = exp(head_dir(·))·D·S`，bias 初值 `log(beta_ref)` → 起点与 β 分支完全相同；
    去掉绝对 mm 回归的均值回归倾向，同时保留它作为"β 的修正项"的角色。
  - 两臂都以 `company-base.yaml` 为模板、seed 20261020、15 轮、从零、RankSim γ=2，
    除该单一开关外**逐字段一致**（`core/tools/new_experiment.py --from` 生成）。
- 达标口径（用户 2026-11-05 明确）：**benchmark2 的 company + B，所有 <10mm 的 1mm 档
  MAE < 2mm**；R 不要求；旧的 single-v2 `test_d` **不是**目标。
  当前 base：company <10mm 1.107，只有 9–10=2.48 未达标（59 帧/**4 例**）、
  2–3=1.97 贴线；B <10mm 1.062，只有 7–8=3.39 未达标（5 帧/**1 例** = `polyp-size-04`，
  标签 7.5mm，其 S/W=0.184 而 B 自己的 8–9=0.414、9–10=0.448 → **队列内部就不自洽**；
  Polyp-Size 的 42 段视频**就是** B，无法补数据）。
- 判据（**先登记，后看数**；全部先在 val 上选，test 只确认一次）：
  - **PASS**：company 9–10 < 2.0 且其余各档不差于 base 三 seed 的**最大值**；
    B <10mm 不差于 base 三 seed 最大值、7–8 不进一步恶化。R 只作回归观察。
  - **FAIL**：任一判定队列出现新的 ≥2mm 档，或 company/B 的 lt10 超出噪声底上限。
  - PASS 也**不等于采纳**：还需第二个 seed 复核（AGENTS.md：不报单个幸运 seed）。
  - 自动判定由 `scripts/summarize_arms.py` 生成（✅/⚠️/❌ 逐档 + PASS/FAIL 表），
    追加为 `output/reports/overnight_20261104.md` 第 6 节，数字全部读自落盘 JSON。
- 队列：`scripts/run_overnight_arms_20261105.sh`（等 D-0066 主队列 + 标定尾巴都退出、
  GPU 空闲后启动；两臂串行；每臂 = 训练 → 官方 `--split test --benchmark
  v11-company-test-v1.json` → `plot_experiment.py --suite benchmark2.json`）。
- 已知的数据侧上限（本臂不解决，登记以免重复尝试）：**company-v1 train 的真值上限是
  10.00mm**（9–10 档 98 帧/8 例，其 S/W=0.476 而 test 的 9–10 是 0.284；10–11 有
  940 帧/56 例但被 `train_max_truth_mm=10.5` 排除），所以 9–10 档永远坐在标签范围边缘、
  且训练/测试的同档几何不一致。`company-v1-big`（11,114 帧/909 例，上限 15.00mm）
  + `benchmark2-big.json` 已存在；D-0064 用 5 轮 warm start 试过（9–10 2.48→2.38，
  中档退化，未采纳），**从零重训**才能去掉 warm start 混淆——留作第三臂，需另行确认。
- Caveat: 两臂训练时工作树仍是 dirty（含本条改的 `core/config.py`、`core/models/v11.py`），
  run 的 `code_version` 会带 dirty；提交时机由 owner 决定，不在队列里自动 commit。
- Evidence: `input/configs/experiments/v11/company-{gatefloor,diranchor}.yaml`、
  `tests/test_decompression_arms.py`、`scripts/run_overnight_arms_20261105.sh`、
  `scripts/summarize_arms.py`、`output/logs/arms_20261105{.log,.steps.log,.chain.log}`、
  `output/reports/arms_20261105.md`、`output/charts/v11-company-{gatefloor,diranchor}.png`
- Owner: twj

### D-0068 — D-0066 与 D-0067 的结果：噪声底确立、标定被判否、门控不是压缩成因、diranchor 拿到首个 company 全档 <2mm

- Date: 2026-11-05
- Status: accepted（记录结果；D-0066、D-0067 由 pre-registered 转为已判定）
- 队列执行：全部步骤 OK，无失败。主队列 20:45→00:49（两 seed），标定尾巴 00:50→00:53，
  两臂链 00:53→04:58（`grep -E "FAIL" output/logs/*.log` 为空）。

**1）噪声底（D-0066 判据 A）—— 已确立**

| 队列 | 三 seed `<10mm` | 全距 |
|---|---|---|
| company | 1.107 / 1.119 / 1.151 → **1.125** | **0.044** |
| B | 1.062 / 1.079 / 1.090 → **1.077** | **0.028** |
| R | 1.959 / 2.021 / 1.809 → 1.930 | 0.211 |

- ⇒ **D-0060（ctx）否决成立**：Δ company +0.152、Δ B +0.232，均远超全距，不是噪声。
- ⇒ **D-0064（big）否决成立**：Δ +0.094 / +0.122 同样超距。
- ⇒ **D-0065（endo）措辞需修正**：Δ company +0.031（**在噪声内，不可区分**）、
  Δ R +0.149（噪声内）、但 **Δ B +0.048 > 0.028 → B 上 Endo 确实更差**。
  原「基本持平」只在 company 成立；「深度源不是瓶颈」仍成立（差异 ≤0.15 且方向不一致）。
- ⇒ **真失败档（三 seed 全超 2mm）**：company **9–10**（2.482–2.898）、B **7–8**（2.894–3.388）、
  R 1–2 / 2–3 / 7–8 / 9–10。company 2–3（1.930–2.075）、B 2–3（1.824–2.405）、B 9–10
  （1.372–2.147）只有 1/3 seed 超标 → **边缘档，不是稳定失败**。

**2）单调标定（D-0066 判据 C）—— 判否，不进配方**

只在 `company-v1/val`（2094 帧/182 例）拟合，test 仅套用，val 用病例级 2 折交叉拟合：

| 口径 | val lt10 | company lt10 | company 9–10 | company 超标档 |
|---|---|---|---|---|
| 标定前 | 1.356 | 1.107 | 2.48 | 1 |
| isotonic | 1.891 ❌ | 1.648 ❌ | 2.70 | 4 ❌ |
| log-linear | 1.429 ❌ | 1.097（噪声级） | 2.41 | 2 ❌ |

- **连域内（val 交叉拟合）都不改善** → 压缩**不是可用单调映射消除的全局刻度错误**；
  log-linear 斜率 b=0.866–0.895 也说明 val 上 pred 并不"欠响应"，压缩集中在**两端尾部**。
- isotonic 制造大量并列值 → Spearman 由 0.425 崩到 0.017/−0.15（**不是**「分辨力不变」；
  严格单调的 log-linear 才保持排序）。
- 报告 §5 的「分辨力没变」表述只适用于 log-linear，isotonic 一行的排序已破坏。

**3）解压缩两臂（D-0067）—— 按预登记判据都 FAIL，但把机制问题定死了**

- **臂 A `gate_beta_floor=0.5`（`20261105-005349_dc4296d5_s20261020_4dfb35`）**：
  门控确实被钉住（wg=0.52–0.53），但 **β 分支占输出的比例仍只有 0.10–0.17**——因为
  `pred_beta` 的幅值本来就远小于 `pred_direct`，改权重不改数值。逐档偏差与 base 重合
  （2–3 **+1.99** vs base +1.95；9–10 **−2.57** vs −2.46，反而略差）；company lt10 1.124
  （=base 均值）、worst_bin 2.598（噪声区间内）、B lt10 1.050、R 1.855。
  ⇒ **门控权重不是压缩的成因**，这条路线关闭。
- **臂 B `direct_scale_anchor=true`（`20261105-025342_36433ff6_s20261020_631bbc`）**：
  意外结果——**门控饱和到 wg=1.00、β 占比 1.00**：锚定后 direct 与 β 是同一参数化
  （都等于 `exp(·)·D·S`），gate 直接放弃了 direct，模型退化为**纯 β·D·S**。
  于是压缩真的减轻：company 逐档偏差 **+1.95/−2.46 → +1.80/−1.68**，
  **worst_bin 2.482 → 1.825，8 个 <10mm 档首次全部 <2mm**；
  **val 同步**改善（worst_bin 3.973 → **3.333**）⇒ 不像单 seed 巧合。
  代价：中档变差（3–4 +0.21、4–5 +0.20、7–8 +0.22 均超噪声底上限），
  company lt10 1.107→**1.199**（超上限 +0.048），B lt10 1.062→**1.205**（超 +0.115），
  B 的 2–3=2.373、7–8=3.381 仍超标（这两档 base 也超，且分别只有 3 例 / 1 例）。
  ⇒ 按预登记判据「其余档不差于噪声底最大值」→ **FAIL，不采纳**。
- **合起来的关键结论**：压缩**存在于 β 分支内部**（纯 β 仍然 +1.80/−1.68），
  不是融合/门控造成的；而融合的门控之所以"偏向 direct"，是因为 β 分支输出幅值偏小——
  也就是说**两个分支都在向均值回归，只是程度不同**。跨队列上纯 β 最差
  （B 的 k 是 company 的 2–8 倍，`D·S` 标度不通用）→ 放弃 direct 分支等于放弃逃生口。

**4）下一步（尚未跑，需 owner 选）**

- diranchor **补第二个 seed**（约 2 小时）以确认「company 全档 <2mm」与中档退化的幅度；
  只有重复成立才允许作为 company 主力。
- 真正对因的臂：让模型**只预测无量纲标度** `k = truth/(D·S)`（把 D·S 作为硬分母而不是加性特征），
  或用**序数/分档头**替代 log-space 平滑 L1——两者都直接攻击"档内均值回归"，而不是再调融合。
- 不再重提：门控下界（本条已否）、全局单调标定（本条已否）、ctx/big/endo/size_input/分支监督/增广。
- Evidence: `output/reports/overnight_20261104.md`（§1–§6）、
  `output/reports/{v11-company-base-3seed,v11-company-gatefloor,v11-company-diranchor}_evalmode.json`、
  `output/reports/recal_diagnosis/`、`output/reports/smallbin_audit_{diranchor,gatefloor}/`、
  `output/charts/v11-company-{base-3seed,gatefloor,diranchor}.png`、
  eval `20261105-025150_dc4296d5_s20261020_9a6872`、`20261105-045653_36433ff6_s20261020_fe4693`
- Owner: twj

### D-0069 — 分档头（ordinal head）：把逐档 MAE 判据变成分类问题（预登记）

- Date: 2026-11-05
- Status: pre-registered
- Context: D-0068 关掉了两条路——全局单调标定（连域内 val 交叉拟合都变差：lt10
  1.356→log-linear 1.429、isotonic 1.891）与门控权重（floor=0.5 把 wg 钉在 0.52，
  逐档偏差却与 base 重合）。diranchor 进一步证明**压缩在 β 分支内部**（wg 饱和到 1.00、
  纯 β·D·S 后偏差仍有 +1.80/−1.68）。标量回归 + log 空间 SmoothL1 的二次核
  （|log err|<0.1）叠加病例/档位重加权，学到的是「重加权后的条件均值」→ 必然向均值压缩。
- 新增诊断（`scripts/bin_accuracy.py`，读 `smallbin_audit` 的 frames.csv，`<10mm` 帧）：

  | 模型 | 队列 | 分档准确率 | ±1 档 | 命中档 MAE | 分错档 MAE |
  |---|---|---:|---:|---:|---:|
  | base（s20261020） | 公司 test | **28.9%** | 69.8% | **0.409** | 1.390 |
  | base（s20261020） | B | **36.0%** | 72.5% | 0.327 | 1.476 |
  | diranchor | 公司 test | 23.4% | 65.5% | 0.407 | 1.441 |
  | diranchor | B | 25.5% | 69.0% | 0.368 | 1.491 |
  | gatefloor | 公司 test | 28.9% | 68.1% | 0.415 | 1.412 |

  **档宽 1mm 且标签是整数 mm → 分档正确时档内误差必然 <1mm**（实测命中档 MAE 只有
  0.41/0.33）。也就是说「每个 <10mm 档 MAE <2mm」的判据，本质上是**分档准确率**问题，
  而现在的模型只有 29%/36%。全部 headroom 都在分类上，不在档内回归上。
- Decision: 加**分档头**臂 `v11-company-ordinal`（`input/configs/experiments/v11/company-ordinal.yaml`）：
  - `model.ordinal_head=true`：11 类（0–9 各一档 + ≥10mm 一档，与 `mm_case_bin.bin_of`
    同口径）交叉熵替代标量回归，输出 = **档左端 + sigmoid(档内偏移)**；≥10 档用
    `10 + softplus(·)`。初值＝`truth_median` 所在档的档中心（与旧 head_dir bias 初值同一思路）。
  - **纯 mm 参数化（owner 2026-11-05 选定）**：D·S 只经 geom token 作为特征，不做硬乘性
    锚定——B 的 k 是 company 的 2–8 倍，硬锚定跨队列不通用（diranchor 在 B 上退化最多即此因）。
  - `loss.name=mm_ordinal`：`L = CE(bin) + ordinal_offset_weight·SmoothL1(档内偏移)
    + rank_gamma·RankSim`，CE 与偏移项都按 `mm_case_bin` 的**病例/档位等权**口径加权
    （`case_bin_mean` 从 `mm_case_bin` 抽出，200 组随机批次 × 2 种口径 `torch.equal`
    逐位等价 → 既有 run 的损失语义未变）。RankSim γ=2.0 保留，与 base 同配方。
  - 单变量：相对 company-base 只改 head+loss（`log_space` 被 CE 取代，是同一改动的必然结果）；
    seed 20261020、15 轮、从零。`ordinal_head` 与 `gate_beta_floor`/`direct_scale_anchor`/
    `size_input` 在 config 层**互斥**，头与损失必须成对出现（否则 ConfigError）。
  - 开启时不构造 β/direct/门控三个头；关闭时结构与旧模型完全一致（`tests/test_ordinal_head.py`）。
- 判据（**先登记，后看数**；test 只确认一次）：
  - **PASS（达标口径）**：company + B 的**所有 <10mm 1mm 档 MAE <2mm**（R 不要求）。
  - **护栏**：company/B 的 `<10mm` 总 MAE 相对三 seed 上限（1.151 / 1.090）恶化 ≤ **+0.05**
    ——档内量化本身最多引入 0.5mm，lt10 略差是这条路线的结构性代价，不能超出这个量级。
  - **机制指标**：分档准确率必须**显著高于 base 的 28.9%（company）/ 36.0%（B）**。
    若 CE 训练没能把 bin_acc 抬上去 → **特征分辨力不足，头不是瓶颈**，下一步应回到
    特征/监督信号（而不是继续换头）。
  - **选模一致性**：val（EMA）`worst_bin` 应低于 base 的 3.973；否则即使 test 好看也不采纳。
  - PASS 仍需第二个 seed 复核后才算采纳（AGENTS.md：不报单个幸运 seed）。
- 队列：`scripts/run_ordinal_20261105.sh`（训练 → 官方 eval → benchmark2 图 →
  小档审计 → 分档准确率 → 与三 seed 噪声底/两臂同表对比）。
- Caveat: 工作树仍 dirty（本条改了 `core/config.py`、`core/models/v11.py`、
  `core/losses/{mm_case_bin,mm_ordinal,build}.py`），run 的 `code_version` 带 dirty；
  提交时机由 owner 决定，不在队列里自动 commit。
- Evidence: run `20261105-103728_e4c55d52_s20261020_4510de`、
  `output/reports/ordinal_20261105.md`、`output/reports/ordinal_bin_accuracy.md`、
  `output/reports/smallbin_audit_ordinal/`、`output/charts/v11-company-ordinal.png`、
  `tests/test_ordinal_head.py`（15 项，全套 94 项通过）
- Owner: twj

### D-0070 — 分档头（ordinal head）结果：FAIL，bin_acc 反降 → 头不是瓶颈

- Date: 2026-11-05
- Status: accepted（负结果；D-0069 的预注册判据已判定）
- Context: 按 D-0069 在 company-v1 上训练 `v11-company-ordinal`（11 类 CE + 档内偏移 +
  RankSim γ=2、纯 mm 参数化、seed 20261020、15 轮、从零）。D-0069 的 evidence 指向
  11:37 被外部 SIGKILL 的中断 run `20261105-103728_e4c55d52_s20261020_4510de`
  （epoch 7/15，manifest 仍为 running，只有 epoch-1 best.pt，不是结果）；正式 run 为
  `20261105-130723_e4c55d52_s20261020_bfeda1`（15 轮 completed），官方 eval
  `20261105-152316_d02d0ccc_s20261020_63019b`（benchmark `v11-company-test-v1`）。
- 结果（final_ema）：
  - company test：lt10 **1.874**（base 三 seed 1.125 [1.107–1.151]）、worst_bin **3.908**；
    未达标档 2–3（2.01）、7–8（2.81）、8–9（3.37）、9–10（3.91）。
  - PolypSize B：lt10 **1.493**（base 1.077），2–3/7–8/8–9/9–10 未达标。
  - REAL-colon R：2.502（仅观察）。
  - val：ema worst_bin **4.455**（base 3.973），选模护栏失败。
  - **bin_acc 反降**：company **18.1%**（base 28.9%）、B **32.0%**（base 36.0%）；
    命中档 MAE 只有 **0.135**，分错档 MAE **2.259**——分类头把档内误差压到 <1mm，
    但分错档的比例更高；逐档命中率 2–3 0%、7–8 1.8%、8–9 0%、9–10 0%。
- Decision: **FAIL，不采纳**。按 D-0069 的机制判据，CE 训练没有把 bin_acc 抬上去
  （反而下降）→ **特征分辨力不足，头不是瓶颈**。不再在 head/loss/输出参数化上继续换；
  下一步回到**特征/输入侧**（提高输入或 ROI 紧视图分辨率、高分辨率紧视图分支、换
  backbone 表示），而不是继续换分类头。备选「直接回归 k = truth/(D·S)」同属输出侧，
  优先级下调。
- Alternatives: 加训/加长 schedule（train CE 已降到 ~1.3，val 不改善，先验判断收益低）；
  调 class weight（不能解释尾档命中率 0%）；继续调 ordinal 头（已证伪）。
- Consequences: 判据中「分档准确率是全部 headroom」这一步被否定；base 的 28.9%/36.0%
  仍是当前上限。任何后续特征侧改动都必须先看 bin_acc 是否抬升，再谈逐档 MAE。
- Evidence: run `20261105-130723_e4c55d52_s20261020_bfeda1`、
  eval `20261105-152316_d02d0ccc_s20261020_63019b`、
  `output/reports/ordinal_20261105.md`、`output/reports/ordinal_bin_accuracy.md`、
  `output/reports/smallbin_audit_ordinal/`、`output/charts/v11-company-ordinal.png`
- Owner: twj

### D-0071 — DataLoader 并行化（data.num_workers / pin_memory）；不改样本顺序

- Date: 2026-11-05
- Status: accepted
- Context: 训练每 step ~0.75–0.9 s，GPU 利用率只有 17–30%、显存 4.6/11 GB。
  实测单样本 `cv2.imread`（1920×1080，部分 4K）+ BGR→RGB 27 ms、`np.load` 11.5 ms，
  batch 16 仅数据就要 ~0.54–0.61 s；所有 `DataLoader` 构造都用默认 `num_workers=0`，
  数据解码与 GPU 计算完全串行。数据盘是机械盘（TOSHIBA MG03ACA100），页缓存之外
  还有寻道延迟。
- Decision: `DataConfig` 新增 typed 字段 `num_workers`（默认 0，校验 [0,32]）与
  `pin_memory`（默认 false）；`build_loader` 把二者传给全部 5 条 DataLoader 分支，
  `num_workers>0` 时启用 `persistent_workers=True`、`prefetch_factor=2`；
  `pin_memory` 在无 CUDA 时自动关闭。`company-base.yaml` 与 `company-ordinal.yaml`
  设为 `num_workers: 4` / `pin_memory: true`；其他配置可用
  `--set data.num_workers=4 --set data.pin_memory=true` 临时启用。
- Alternatives: 半分辨率解码（`cv2.IMREAD_REDUCED_COLOR_2`）——会改变像素重采样路径，
  与历史 run 不可比，押后；把数据迁到 SSD——系统盘只剩 8.3 GB，且页缓存已挡住大部分
  读，收益低。
- 实测（company-v1 train，batch 16，页缓存已热）：
  - loader-only：**541 → 167 ms/batch**（29.6 → 95.6 samples/s，**3.2×**）；
  - 120-step 训练：**726 → 190 ms/step**（**3.8×**）；
  - 预计 15 轮从 ~2h15 降到 ~35min。
- Consequences: shuffle 仍由主进程 generator 决定，样本内容与 batch 顺序不变；
  `scale_aug`/`light` 等使用 `__getitem__` 内 `random` 的配置在 worker>0 时本来
  就依赖 worker RNG，后续若要严格复现需固定 worker seed。4K 帧仍是最慢样本，
  若还要更快再做半分辨率解码。
- Evidence: `core/config.py`、`core/data/loaders.py`、`core/data/build.py`、
  `tests/test_dataloader_workers.py`、`input/configs/experiments/v11/{company-base,company-ordinal}.yaml`、
  run `v11-loader-perf/20261105-152813_42d38272_s20261020_3ba383`（debug，仅测速）
- Owner: twj

### D-0072 — diranchor 是欠训练（val 曲线仍在改善）；smallbin_audit 在分档头 run 上的 NaN 修复

- Date: 2026-11-05
- Status: accepted（补充 D-0068/D-0070；本条原误编号为 D-0070，与并行会话的条目冲突，已重编号）
- Context: D-0070 否决了分档头并把方向定在特征/输入侧。复核三个 run 的 val（EMA）曲线时
  发现一条**尚未被记录**的事实：diranchor（唯一让 company 全档 <2mm 的臂）在 15 轮时
  **还没收敛**，而 base 早已开始过拟合。
- 证据（`metrics.jsonl`，EMA 口径）：

  | epoch | base worst_bin | diranchor worst_bin | base lt10 | diranchor lt10 |
  |---|---:|---:|---:|---:|
  | 7 | 3.595 | 3.408 | 1.373 | 1.539 |
  | 10 | 3.841 | 3.340 | 1.359 | 1.435 |
  | 13 | 3.948 | 3.338 | 1.358 | 1.419 |
  | 15 | **3.973**（持续变差） | **3.333**（仍在降） | 1.356 | **1.407**（仍在降） |

- Decision: 把「**diranchor 延长 schedule（30–40 轮）+ 第二个 seed**」列为下一步的第一优先，
  理由是它的中档退化（company 3–4/4–5/7–8 各 +0.2）可能只是没收敛，而不是这条路线的固有代价；
  D-0071 之后 15 轮只要 ~35 分钟，这个实验的成本已经很低。判定仍全程用 val，test 只确认一次。
  这与 D-0070 的「转特征/输入侧」不冲突：特征侧是更大的改动，先吃掉已有的欠训练收益。
- 附带修复：`scripts/smallbin_audit.py` 在分档头 run 上崩溃——缺 β/direct/门控分支时
  逐帧字典写 `None`，`med()` 的 `np.isfinite(None)` 抛 TypeError（链条里那一步 FAIL）。
  改为统一记 `NaN`（`round(nan)` 仍是 nan，下游 `med()`/表格格式化都能处理），
  已重跑覆盖 `output/reports/smallbin_audit_ordinal/`。
- Evidence: `output/runs/beta-regression/v11-company-{base,diranchor}/.../metrics.jsonl`、
  `scripts/smallbin_audit.py`、`output/reports/smallbin_audit_ordinal/summary.md`
- Owner: twj

### D-0073 — diranchor 跟进：第二 seed 复核 + 40 轮长 schedule（预登记）

- Date: 2026-11-05
- Status: pre-registered
- Context: diranchor 是目前**唯一让 company 全部 <10mm 档 <2mm** 的臂（worst_bin 1.825、
  val EMA worst_bin 3.333），但只有单 seed，按规则不能采纳；且 D-0072 发现它在 15 轮时
  **val 仍在单调改善**（worst_bin 3.408→3.340→3.338→3.333、lt10 1.539→1.435→1.419→1.407），
  而 base 自 ep7 起持续变差（3.595→3.973）→ 中档退化可能只是没收敛。D-0071 之后
  15 轮约 30–60 分钟，这两个实验的成本已经很低。
- 两个 run（除 seed / epochs 外与 `company-diranchor.yaml` 逐字段一致）：
  1. **复核**：`company-diranchor.yaml --set seed=20261021`（15 轮，`--tag d0073`）；
  2. **长 schedule**：`company-diranchor40.yaml`（`epochs: 40`、`early_stopping.min_epochs: 40`
     以保持 base 那条「早停等于关闭」的性质；seed 20261020）。
  两者都带 `--set data.num_workers=4 --set data.pin_memory=true`（D-0071：只改加载并行度，
  样本内容与 batch 顺序不变；diranchor 无 `scale_aug`/`light`，`__getitem__` 内没有 RNG）。
- 判据（**先登记，后看数**；全程 val 选模，test 每个 run 只评一次）：
  - **① 复核**：company 全档 <2mm 必须在第二个 seed 上重现（关键档 9–10 <2.0）。
    两 seed 的 company worst_bin 全距＝这条路线自己的噪声尺度；若 B 的 lt10 两 seed 都
    >1.090（base 噪声底上限）→ B 的退化是真实的，不是单 seed 现象。
  - **② 长 schedule（采纳条件）**：val EMA worst_bin 继续低于 15 轮的 **3.333**；
    company 全档仍 <2mm；中档退化收敛回来（3–4/4–5/7–8 相对 base 三 seed 上限的 Δ ≤ 0）；
    B 的 lt10 不进一步恶化（≤ **1.205**）。若 40 轮内 val worst_bin 不再改善 →
    说明 15 轮已接近该配方的极限，中档退化是这条路线的固有代价。
  - **首要读点（D-0070 指定）**：`scripts/bin_accuracy.py` 的分档准确率必须抬过 base 的
    **28.9%（company）/ 36.0%（B）**；否则即使逐档达标，也只说明读出映射变了、
    分辨力问题没解决，特征侧仍必须做。
  - 两个 seed 一致才允许把 diranchor 记为 company 主力；单个 seed 好看不采纳。
- 队列：`scripts/run_diranchor_followup_20261105.sh`（① → ② 串行；每个 run：官方
  `--split test --benchmark v11-company-test-v1.json` → benchmark2 逐档图 → 小档审计 →
  分档准确率同表 → 与三 seed 噪声底对比汇总）。
- Caveat: 工作树仍 dirty（含 D-0069/D-0071/D-0072 的改动），`code_version` 带 dirty；
  提交时机由 owner 决定。
- Evidence: `input/configs/experiments/v11/company-diranchor40.yaml`、
  `output/reports/diranchor_followup_20261105.md`、`output/reports/diranchor_bin_accuracy.md`、
  `output/charts/v11-company-diranchor{,-2seed,40}.png`、
  `output/reports/smallbin_audit_diranchor{_s21,40}/`、
  `output/logs/diranchor_followup{.log,.steps.log,.chain.log}`
- Owner: twj

### D-0074 — diranchor 跟进结果：company test 判据 2 seed + 40 轮成立；val 与 bin_acc 判据未过，B 7–8 仍是唯一阻塞

- Date: 2026-11-05
- Status: accepted（D-0073 预注册判据已判定；混合/负结果）
- Context: D-0073 队列 15:50–18:38 完成：① `company-diranchor` seed 20261021
  15 轮；② `company-diranchor40` seed 20261020 40 轮。两者都带 `data.num_workers=4/
  pin_memory=true`（D-0071），其余与 diranchor 单变量一致。
- 结果（final_ema；company 为官方 eval，B/R 为 benchmark2 离线 evalmode）：
  - **s21（15 轮）**：company lt10 **1.152**、worst_bin **1.95**（8 个 <10mm 档全部 <2）；
    B lt10 **1.176**、7–8 **3.19**（2–3 1.97、8–9 1.45）；R 1.964；
    val ema worst_bin **3.621**（best 3.405 @ep5）、lt10 1.425。
  - **d40（40 轮）**：company lt10 **1.119**、worst_bin **1.744**（全档 <2；2–3 **1.744**、
    9–10 **1.547**）；B lt10 **1.125**、7–8 **3.51**（2–3 1.908 已达标）；R 1.851；
    val ema worst_bin **3.607**（best 3.414 @ep10）、lt10 **1.324**。
  - **2 seed 均值（s20+s21）**：company lt10 1.176、全档 <2（2–3 1.878、9–10 1.779）；
    B lt10 1.190、2–3 2.173、7–8 3.285。
  - **bin_acc**：s21 company 26.8%/B 25.5%；d40 company 26.2%/B 24.5%，
    均**仍低于 base 的 28.9%/36.0%**（D-0070 的首要读点未过）。
- 按 D-0073 判据：① company 全档 <2mm 在第二 seed 重现 ✅；② val worst_bin 需低于
  s20 的 3.333 ❌（s21 3.621、d40 3.607），中档 Δ≤0 ❌（d40 3–4 +0.262、4–5 +0.094），
  B lt10 ≤1.205 ✅（1.176/1.125）；首要读点 bin_acc ❌。
- Decision: **company test 判据在 diranchor 路线上可复现，但不采纳为新的主基线**：
  (a) val 没有继续改善（3.62/3.61 > 3.333），说明中档退化不是单纯「15 轮没训够」；
  (b) bin_acc 没有抬过 base → 改善来自 test 上的偏差重分配，不是模型分辨力提升；
  (c) d40 只有单 seed，且其 company lt10 1.119 与 base 均值 1.125 的差异在 seed 噪声内。
  B 仍只剩 7–8（3.19–3.51，`polyp-size-04` 一例）与 lt10 高于 base 上限；B 的几何标度
  问题留给随后排队的 span-normalization（`span_reference_width=1258`）检验。
- Consequences: 不要把「company 全档 <2mm」当成已解决的测量能力——它在 test 上成立，
  但 val 与 bin_acc 不支持泛化结论；任何后续报告必须同时给 val worst_bin 与 bin_acc。
  下一次若要巩固该结论，需要 d40 的第二 seed，或先解决 B 7–8 与特征分辨力。
- Evidence: s21 `20261105-155004_187149e3_s20261021_8a5d93`；d40
  `20261105-164359_28105278_s20261020_8e76ad`；eval
  `20261105-183629_28105278_s20261020_5baa06`；
  `output/reports/diranchor_followup_20261105.md`、`diranchor_bin_accuracy.md`、
  `output/charts/v11-company-diranchor-2seed.png`、`v11-company-diranchor40.png`
- Owner: twj

### D-0075 — 跨度归一化（S/W）结果：预登记判据未达到，B 7–8 改善但 company 9–10 回退，不采纳

- Date: 2026-11-05
- Status: accepted（负结果）
- Context: 按 span-normalization 计划（D-0073 队列之后）在隔离 worktree 运行
  `company-diranchor-span-norm`：diranchor + `model.span_reference_width=1258`
  （跨度 S → 1258·S/W），seed 20261021、15 轮、workers 4；对照为同种子 diranchor
  15 轮 `20261105-155004_187149e3_s20261021_8a5d93`。原 launcher 被 swap 门槛
  （SwapFree 2 MiB < 256 MiB）卡住 1h43m，改为复用未修改的 preflight/smoke/train
  流程、仅绕过 swap 检查后启动；pinned 文件哈希校验通过。
- 结果（final_ema，run `20261105-202956_40f4313f_s20261021_9cdd40`）：
  - **val（预登记判据）**：9–10 **3.621→3.847（+0.227，需 ≤−0.3）**、2–3
    1.782→2.124（+0.342，需 ≤+0.2）、4–5 1.031→**1.725（+0.694）**、8–9
    2.077→1.798（−0.279）；→ **未达到**。
  - **company 官方 eval**（`20261105-211422_...-test-84bf65`）：lt10 **1.108**
    （对照 1.152）、worst_bin **2.347**（9–10；对照 1.95；2–3 1.951、3–4 1.211、
    4–5 0.989、5–6 0.785、6–7 1.005、7–8 0.803、8–9 1.676）→ **9–10 回到 2 以上，
    company「全档 <2」被打破**；病例等权 lt10 1.066。
  - **B 官方 eval**（`20261105-211449_...-test-f893ad`）：lt10 **1.127**
    （对照 1.176）、worst_bin **2.983**（7–8；对照 3.19；2–3 1.755、8–9 1.252、
    9–10 1.675）→ **7–8 仍 >2**，但 lt10 与 7–8 均改善；病例等权 1.127。
- Decision: **不采纳**。S/W 把 company 9–10 的误差换到 B 的 7–8/5–6，属于
  **trade 而非净改善**；预登记的 val 判据（9–10 ≥0.3 改善、其他 ≤0.2 退化）未过。
  B 7–8 仍是唯一 blocker（`polyp-size-04` 单例），且 company 全档 <2 只在
  diranchor40 单 seed 上出现过。跨度归一化路线到此关闭，不进入主基线。
- Consequences: 不再用全局 S/W 归一化推进 B；若继续 B 7–8，应做该档的标签/框
  专项复核（单例）或转特征侧（D-0070）；任何报告需同时给 val 逐档与 test。
- Evidence: run `20261105-202956_40f4313f_s20261021_9cdd40`、
  eval `20261105-211422_40f4313f_s20261021_8a8b2c-test-84bf65`、
  `20261105-211449_5b38edef_s20261021_491325-test-f893ad`、
  `output/reports/span_normalization_20261105/{result.md,preflight.json,case_audit/}`、
  `output/charts/v11-company-diranchor-span-norm.png`
- Owner: twj

### D-0076 — benchmark2 的 R 队列排除 1–2mm 档（reporting suite 过滤；冻结 benchmark 不变）

- Date: 2026-11-05
- Status: accepted
- Context: R（REAL-colon）1–2mm 在 company-v1 训练集中没有任何样本（D-0047），
  该档只有 21 帧/14 例，却经常决定 R 的 worst_bin（如 diranchor s21 的 3.10），
  把 R 的观察指标变成「不可外推档」的噪声。用户要求 benchmark2 的 R 计算不再
  纳入 1–2mm，且以后默认如此。
- Decision: 在 **reporting suite** 层实现过滤，不改冻结 benchmark：
  - `input/benchmarks/benchmark2{,-big,-endo}.json` 的 `r` 成员新增
    `"min_truth_mm": 2.0`；
  - `scripts/plot_experiment.py` 的 `measure()` 按 `min_truth_mm` 过滤 manifest 行，
    无 suite 时的旧口径对 `r` 默认也取 `2.0`（`MIN_TRUTH_DEFAULT`）；
  - 只影响离线出图 / `_evalmode.json` / 基于它的汇总；`v11-r-test-v2.json`
    保持冻结，官方 `eval.py --benchmark v11-r-test-v2.json` 仍评全部帧。
  - 新增 `tests/test_benchmark_suite_filter.py` 覆盖 suite 字段与过滤函数。
- Alternatives: 新建 `v11-r-test-v3`（需要新的 split/成员过滤与 data 侧支持）；
  改 metric 跳过某一档（会改变 metric protocol，且 lt10 仍含该档样本）。
- Consequences: 以后 benchmark2 的 R 图表/汇总从 2–3mm 开始，lt10/MAE 也不含 1–2；
  历史图表与 JSON 不重写（仍含 1–2）。R 的 worst_bin 因此改为由 7–8 等档决定，
  与 company/B 的判据口径更接近；R 仍不是验收队列。若需要官方 R eval 也排除
  1–2，必须新建 benchmark 版本。
- Evidence: `input/benchmarks/benchmark2*.json`、`scripts/plot_experiment.py`、
  `tests/test_benchmark_suite_filter.py`、`input/benchmarks/README.md`
- Owner: twj

### D-0077 — 选模口径：robust_worst_bin（≥8 例的档）+ 病例等权 lt10；验收 benchmark 不变

- Date: 2026-11-05
- Status: accepted
- Context: 旧 monitor `ema/v11/worst_bin` 由 val 的 9–10 档决定，而该档只有约 41 帧/4 例；
  D-0074 用这个指标否掉 40 轮，但 test 与 val lt10 都支持 40 轮；D-0075 的 val 9–10
  也只有个位数病例。选模指标本身噪声过大，是当前实验排序失真的来源之一。
- Decision: `core/metrics/v11.py` 改为累积 pred/target/case 的 `V11Metric`，原有
  `mae/mae_lt10/mae_ge10/acc10/mae_k_k+1/worst_bin` 定义不变，新增**诊断字段**：
  `case_mae`（病例内先平均、病例间再平均）、`case_mae_lt10`、
  `robust_worst_bin`（只在病例数 ≥8 的 1mm 档里取最差档）、`robust_bin_count`。
  新 run 的 `early_stopping.monitor` 用 `ema/v11/robust_worst_bin`，
  `case_mae_lt10` 作护栏；`ema/v11/worst_bin` 继续记录但不再选模。
  `benchmark.metric_versions` 不变（v11: v2），不新建测试 benchmark；
  新增字段是训练/选模诊断，验收指标（worst_bin、逐档 MAE）定义未改。
- Alternatives: 新建 metric version v3 + 新 benchmark（会破坏与历史 company/B 的可比性）；
  post-hoc 只用 aggregate 指标（拿不到病例级）。
- Consequences: 新旧 run 的 `best.pt` 选择规则不同；`final_ema` 仍是末轮，队列用它，
  因此训练轨迹可比。报告必须同时给 `robust_worst_bin` 与 `case_mae_lt10`，并注明
  它们不是官方验收指标。没有 case_id 的合成 batch 把每个样本当独立病例（兼容旧测试）。
- Evidence: `core/metrics/v11.py`、`tests/test_v11_case_metric.py`、
  `output/reports/stability_protocol_20261105.md`
- Owner: twj

### D-0078 — β-only 融合开关 + span 输入零初始化旁路（保留 S，不替换）

- Date: 2026-11-05
- Status: accepted（预注册，队列运行中）
- Context: 推理侧实测 diranchor 的 `wg ≥0.9999`、direct 贡献 ≤0.01%，
  门控/direct 在推理时是死重；但 `head_dir` 挂在共享注意力主干上，训练时仍收梯度，
  删掉会改变优化轨迹，只能用实验回答。span-norm 的「替换 S→1258·S/W」已被
  D-0075 判否；正确形式应是保留 S、把 `log(S/W)` 与 `log W` 作为零初始化条件输入。
- Decision:
  - `ModelConfig.fusion: gate|none`（默认 gate，旧 checkpoint/历史 run 结构不变）。
    `fusion=none` 不构造 `head_dir`/`gate`，只留 `head_beta`（零初始化 →
    起点 `βref·D·S`，与 diranchor 初值逐位一致）；与 `gate_beta_floor/
    direct_scale_anchor/ordinal_head` 互斥。
  - `ModelConfig.span_input=true`：`V11FramesDataset` 增加 `S_norm=S/W` 与 `W`；
    β 头加零初始化旁路 `[log(S/W)−log(0.24629), log W−log(1258)]`（两项均为
    company-v1 train 中位数；零初始化保证起点与无旁路模型逐位一致）。与
    `size_input` 互斥；不替换 `S`。
  - 新配置：`company-diranchor40-sel.yaml`（d40 + 新选模指标）、
    `company-diranchor40-spaninput.yaml`（+ span_input）、
    `company-diranchor40-betaonly.yaml`（fusion=none）；40 轮、2 seed。
- Alternatives: 直接删除 gate/direct 不实验（无法排除辅助头深监督）；继续用
  S→S/W 替换（D-0075 已否）。
- Consequences: β-only 的 `state_dict` 没有 `head_dir/gate`，不能与旧 checkpoint
  互载；span_input 新增 2·d 参数，旧 config 不受影响；`plot_experiment` 与
  `smallbin_audit` 的自定义 collate 已补 `S_norm/W`。数据 schema/benchmark_id 不变。
- Evidence: `core/models/v11.py`、`core/config.py`、
  `core/data/datasets/v11_frames.py`、`tests/test_fusion_span.py`、
  `input/configs/experiments/v11/company-diranchor40-{sel,spaninput,betaonly}.yaml`
- Owner: twj

### D-0079 — 过夜队列预注册：d40-sel s21 → spaninput(s20,s21) → β-only(s20,s21)

- Date: 2026-11-05
- Status: pre-registered（队列 22:43 启动，预计 6–7 小时）
- Context: 用户要求在 diranchor40 主干上一次性验证三件事：s21 噪声底、
  span 旁路（D-0078）、β-only（D-0078）；S1–S6 与采纳规则见
  `output/reports/stability_protocol_20261105.md`（先写规则，后看数）。
- Decision: 串行队列
  `scripts/run_overnight_20261105.py`：① d40-sel s21；② spaninput s20；
  ③ betaonly s20；④ spaninput s21；⑤ betaonly s21。每个 run 40 轮、
  `checkpoint_every=5`（只改容错）、官方 company eval、benchmark2 出图
  （R 排除 1–2mm，D-0076）、小档审计；失败只记录不中断。
  采纳规则：spaninput 需 S1 下降且 company 9–10<2、B 7–8 不低于 2.83，
  S2/S3 不恶化、S4 不低于 d40；β-only 若落在 d40 的 seed 噪声内则采纳为简化主体，
  否则保留辅助头并记录其隐式深监督作用。步骤 5/6（big 从零、448 分辨率）
  等本队列结果后再定，不在今晚执行。
- Consequences: 队列期间的 code/配置以当前 working tree 为准（dirty）；run 的
  `code_version` 带 dirty。正式采纳仍需第二 seed/复核（β-only 和 spaninput
  本队列已各有 s20/s21，但 d40-sel 只有 s21，s20 用旧 monitor 但 final_ema 可比）。
- Evidence: `scripts/run_overnight_20261105.py`、`output/logs/overnight_20261105*`、
  `output/reports/stability_protocol_20261105.md`
- Owner: twj

### D-0080 — 追加步骤 5/6：company-v1-big 从零 + 448 分辨率 screen（独立 448 benchmark）

- Date: 2026-11-05
- Status: pre-registered（append 队列 23:17 起等待前序 D-0079 队列退出）
- Context: 用户要求把第一段计划里暂缓的步骤 5/6 也追加进今晚链路。
  步骤 5 是数据侧：company-v1-big（11,114 帧/909 例，真值上限 15mm）从零训练，
  去掉 D-0064 的 5 轮 warm start 混淆；步骤 6 是特征分辨率：224→448，
  必须同时重训 base@448 当锚。
- Decision:
  - **big-scratch**：`company-big-scratch.yaml`＝diranchor40 recipe（gate/direct anchor、
    RankSim γ=2、40 轮、从零、seed 20261020、`beta_ref=0.016903`），
    data.version `company-v1-big`；官方 company eval 用
    `--set data.version=company-v1` 复用冻结 `v11-company-test-v1`（测试帧相同），
    benchmark2 图用 `benchmark2-big.json`。
  - **448**：`DataConfig.input_size`（默认 224，校验 32 的倍数）；`rgb_views/_letterbox`
    按该尺寸；448 用 batch 4 × grad_accum 4（有效 batch 16、优化器 step 数不变）。
    `base448`（company-base 15 轮）与 `diranchor15-448`（diranchor 15 轮）各 1 个
    screen run。`data.eval_transform='identity448'`（no-op）把 448 的评估协议
    固定成独立 benchmark：`v11-company-test-v1-448.json`，
    benchmark_id `sha256:e3cb0ad9…`，与 224 的 `sha256:d3ddb232…` 不同；
    测试帧/标签/指标不变。
  - `scripts/run_overnight_append_20261105.py` 等前序队列 PID 3697898 退出后自动接力：
    big-scratch → base448 smoke → base448 → diranchor15-448 smoke → diranchor15-448；
    失败只记录、不中断。日志 `output/logs/overnight_append_20261105.*`。
- Alternatives: 448 直接复用 224 benchmark_id（不满足用户“benchmark_id 变”的要求）；
  448 用 batch 8（显存风险）；big 用 warm start（D-0064 的混淆）。
- Consequences: 448 是 **15 轮 screen**，不是 40 轮确认；若 screen 有信号再补 40 轮。
  448 结果不能与 224 benchmark_id 混比，必须标注 `v11-company-test-v1-448`。
  big 的官方 eval 使用冻结 company benchmark + data.version override；报告时注明
  训练数据版本与测试帧。
- Evidence: `input/configs/experiments/v11/company-big-scratch.yaml`、
  `company-base448.yaml`、`company-diranchor15-448.yaml`、
  `input/benchmarks/v11-company-test-v1-448.json`、`core/config.py`、
  `core/data/datasets/v11_frames.py`、`core/data/transforms/identity.py`、
  `scripts/run_overnight_append_20261105.py`
- Owner: twj

### D-0081 — 采纳 beta-only 作为算法主体（owner 2026-11-06）

- Date: 2026-11-06
- Status: accepted（owner 决定）
- Context: β-only 与 diranchor 的配对结果（final_ema，官方 company eval + benchmark2 新口径）：

  | 指标 | β-only s20 | β-only s21 | d40 s20 | d40-sel s21 |
  |---|---:|---:|---:|---:|
  | company lt10 | 1.146 | **1.090** | 1.119 | 1.122 |
  | company worst | 1.764 | 1.932 | 1.744 | 1.868 |
  | B lt10 | 1.114 | **1.093** | 1.125 | 1.121 |
  | B worst（7–8） | 3.46 | 3.58 | 3.51 | 3.68 |
  | R lt10（新） | 1.767 | 1.778 | 1.711 | 1.715 |

  β-only 两个 seed 的 S1（B 7–8）为 3.46/3.58，落在 d40 的 3.51/3.68 内；
  company worst 1.764/1.932 vs d40 1.744/1.868，差 0.02–0.06（d40 两 seed 全距 0.124）；
  company/B lt10 在两个 seed 上均不差于（s21 甚至更好）同 seed diranchor。
  按 D-0078 预注册「β-only ≈ diranchor（S1/lt10 在 seed 噪声内）→ 可简化」，
  owner 决定采纳。
- Decision: **算法主体改为 beta-only**（`model.fusion: none`，只保留 `head_beta`）：
  - 新实验配置默认 `fusion: none`；`gate`/`head_dir` 不再进入主配方；
  - diranchor / gatefloor 的历史 run 保留但不再作为主体；
  - β-only 的 s20/s21（`company-diranchor40-betaonly` 两个 run）作为证据；
  - 报告需按 seed 并列，不取单个幸运 seed；B 7–8（3.46–3.58）仍是唯一 blocker；
  - S3/S4/S6 的正式表在 append 完成后补齐；本决定不改变验收 benchmark。
- Alternatives: 保留 diranchor（多两个头、推理时 wg=1、没有收益）；继续等更多 seed（owner 已决定）。
- Consequences: 后续 448 臂改为 beta-only@448（D-0082）；模型参数更少、无门控需要解释；
  旧 checkpoint 不能直接载入 fusion=none（state_dict 无 head_dir/gate）。
- Evidence: `v11-company-diranchor40-betaonly/20261106-032421_9659caf2_s20261020_e4fafe`、
  `.../20261106-075001_f83b8b9b_s20261021_48542a`、对应 eval 记录与
  `output/reports/v11-company-diranchor40-betaonly-s2026102{0,1}_evalmode.json`
- Owner: twj

### D-0082 — 后续 448 臂改为 beta-only@448（保留实验名兼容 append 队列）

- Date: 2026-11-06
- Status: accepted
- Context: D-0081 采纳 beta-only 后，原计划中的 `company-diranchor15-448`（diranchor@448）
  不再是主配方；append 队列已经加载了固定的步骤名，正在跑 base448。
- Decision: 把 `input/configs/experiments/v11/company-diranchor15-448.yaml` 的模型改为
  `fusion: none`（删除 `direct_scale_anchor`），其余不变（448、15 轮、batch4×accum4、
  `v11-company-test-v1-448`）。**experiment 名保留 `v11-company-diranchor15-448`**
  以兼容 append 队列的 `find_run`；run 的 resolved config 会明确显示 `fusion: none`。
  `base448` 保持 base recipe（gate），作为 448 的 base 锚。
- Alternatives: 改 experiment 名（append 的 find_run 会找不到，需手动补 eval/chart/audit）；
  重排队列（会打断正在跑的 base448）。
- Consequences: 448 的比较变为 **β-only@448 vs β-only@224**（s20/s21）和
  base448 vs base224；旧的 diranchor@448 臂不再存在。报告里必须注明
  `diranchor15-448` 是历史名字、实际为 beta-only@448。
- Evidence: `input/configs/experiments/v11/company-diranchor15-448.yaml`、
  `benchmark.py verify`（`v11-company-test-v1-448`，id `sha256:e3cb0ad9…`）、
  `scripts/run_overnight_append_20261105.py`
- Owner: twj

### D-0083 — benchmark2 把 PolypSize B 7–8mm 单例当离群，不再报该档 MAE（owner 2026-11-06）

- Date: 2026-11-06
- Status: accepted（owner 决定）
- Context: B 7–8mm 档只有 `polyp-size-04` 一个病例（5 帧），且是唯一持续超标的档
  （base 2.89–3.39、diranchor 3.19–3.51、β-only s20/s21 3.46/3.58）。owner 要求用官方检测器
  复核该例的框：`polyp_v2.1_yolov8_230731_ssl_lst.onnx` 输出的框与 manifest 存储框逐像素一致
  （置信度 0.81–0.88，span 182–197 px），所以问题不在检测。该例 S/W=0.156–0.169，而 B 自己的
  8–9/9–10 档为 0.414/0.448，队列内部几何不自洽；42 段 Polyp-Size 视频本身就是 B，无法补数据。
  owner 判断这属于单例离群，决定 benchmark2 不再报告 7–8 档。
- Decision: `benchmark2.json`、`benchmark2-big.json`、`benchmark2-endo.json` 的 B 成员加
  `"exclude_bins": [7]`；`scripts/plot_experiment.py` 的 `filter_rows()` 按 `int(truth_mm)` 剔除该档
  （7.0–7.99 剔除，8.0 归入 8–9），suite 路径与默认路径（B 无 suite）都生效。company 与 R 口径不变。
  **这是报告口径，不是新 benchmark**：冻结的 `v11-b-test-v2.json` 不动，官方
  `eval.py --benchmark v11-b-test-v2.json` 仍逐帧评估包括 7–8；任何引用 benchmark2 的表/图必须
  注明 B 已排除 7–8 单例。
- Alternatives: 保持报告 7–8 并继续优化（owner 已判定不可达）；删除/改写冻结 benchmark
  （禁止）；把该 case 从数据中删掉（会改变数据版本与训练成员，代价更大）。
- Consequences: benchmark2 的 B `worst_bin`/`lt10`/逐档图从本日起不再含 7–8；历史图表与
  `_evalmode.json` 不重写（仍含 7–8），新旧数字不可直接比较；B 7–8 仍可在官方冻结 benchmark
  和审计里查到。`tests/test_benchmark_suite_filter.py` 覆盖三个 suite 的 `exclude_bins` 与
  `filter_rows` 行为。
- Evidence: `input/benchmarks/benchmark2{,-big,-endo}.json`、`scripts/plot_experiment.py`、
  `tests/test_benchmark_suite_filter.py`、`output/review/polyp_size_04_detector/`、
  `input/benchmarks/v11-b-test-v2.json`（未改）
- Owner: twj

### D-0084 — 纯图像单视图回归（image-only）不可行（2026-11-06）

- Date: 2026-11-06
- Status: accepted（负结果；owner 2026-11-06 要求先做这个快速对照）
- Context: owner 要求验证「不用两个视图、直接从图像出毫米数」是否够用。配置
  `company-imageonly.yaml`：`model.image_only: true`、`data.views=1`、`model.views=1`，
  去掉深度 token、10 维几何标量、β·D·S 分支与 gate，只保留 ResNet50 单视图 + 注意力池化 +
  `softplus(head_img)` 直接回归；`fusion: none`，15 轮，seed 20261020，company-v1。
- Decision: **不采纳 image-only**。final_ema（ep15，benchmark2 新口径：B 排除 7–8、R 排除 1–2）：
  company lt10 **1.547**、B lt10 **1.671**、R lt10 2.095；对比同数据 β-only@224 双视图
  s20/s21 的 company 1.146/1.090、B 1.054/1.029。差距远超 seed 噪声，且 company/B 所有档都
  明显退化；单视图 + 纯图像没有足够尺度信息，深度/几何/β 参数化不可省。
- Alternatives: 单视图 + 深度/几何（未测，但双视图 β-only 已支持）；image-only 40 轮
  （15 轮 val 已无改善迹象，不优先）。
- Consequences: 主配方保持双视图 β-only（D-0081）；本 run 作为负结果保留，不进入报告主表。
  图表 `output/charts/v11-company-imageonly-s20261020-no78.png`。
- Evidence: `input/configs/experiments/v11/company-imageonly.yaml`、
  run `v11-company-imageonly/20261106-163734_d4b7e8e6_s20261020_abd1c4`、
  `output/reports/v11-company-imageonly-s20261020-no78_evalmode.json`
- Owner: twj

### D-0085 — Endo β-only 否决，主配方保持 DA β-only（2026-11-06）

- Date: 2026-11-06
- Status: accepted（owner 2026-11-06 预登记的终选判据）
- Context: 按 D-0081 的 β-only 配方、把 DA-V2 深度换成 EndoStreamDepth（`company-betaonly-endo.yaml`，
  `company-v1-endo` / `b-r-endo`，40 轮，seed 20261020，final_ema ep40）。官方 eval
  `20261106-192119_a24a260b_s20261020_9156c1`，benchmark2-endo（D-0083 口径：B 排除 7–8、R 排除 1–2）。
  预登记判据：Endo≈DA 要求 company/B lt10 差 ≤0.10、S1 ≤0.15。
- Decision: **不采纳 Endo，主配方保持 DA β-only（D-0081）**。结果：
  - company lt10 **1.109** vs DA 1.146/1.090（Δ −0.037/+0.019，在噪声内）→ 通过；
  - S1（B 7–8）**1.706** vs 3.459/3.580 → 通过（Endo 单独修好了这个单例）；
  - B lt10 **1.642** vs 1.054/1.029（Δ **+0.588/+0.613**）→ FAIL；B 2–3 3.003、9–10 2.156 均 >2；
  - R lt10 2.196 vs 1.767/1.778（Δ +0.42，次要队列）。
  B 退化远超历史 B 噪声全距 0.028，且 company 没有任何超出噪声的收益 → 不跑第二 seed。
- Alternatives: 再跑 Endo 第二 seed（判据已以 0.59 的差距失败，无必要）；仅用 Endo 修 B 7–8
  （该档已按 D-0083 不报告，且 Endo 同时破坏 2–3/9–10）。
- Consequences: 终选为 **DA β-only**；Endo run/eval/图/审计保留为负结果，可作为「深度特征换源
  会改变跨队列几何标度」的证据。报告 `output/reports/endo_betaonly_20261106.md`。
- Evidence: `input/configs/experiments/v11/company-betaonly-endo.yaml`、
  run `v11-company-betaonly-endo/20261106-172028_a24a260b_s20261020_663041`、
  eval `v11-company-betaonly-endo-eval/20261106-192119_a24a260b_s20261020_9156c1`、
  `output/charts/v11-company-betaonly-endo-s20261020.png`
- Owner: twj

### D-0086 — Endo 的 B 误差是队列级 β 头校准偏移，不是米制刻度或 token OOD（2026-11-06）

- Date: 2026-11-06
- Status: accepted（只读诊断结论）
- Context: 为解释 D-0085 中 Endo 在 B 上 lt10 1.642（DA 1.054/1.029）而做的两项只读诊断：
  1. **米制刻度**（`scripts/endo_scale_audit.py`）：用 `f=q_mm·span/truth` 作隐含焦距。
     B/company_test = **1.084 [0.944, 1.224]**（CI 含 1），q_mm 21.4 vs 21.8 → B 不是常数
     因子偏移；R/company = 2.492 [2.134, 2.887] 才是真正的 2.5× 刻度偏移。DA 的
     B/company 反而只有 0.306 [0.256, 0.369]。
  2. **token/β 输入 OOD**（`scripts/endo_token_ood.py`）：Endo B 的病灶深度结构
     `MAD/median` 0.056 vs company 0.066（DA B 0.042），B 并不结构性 OOD；β 输入线性
     探针 AUC Endo 0.929 / DA 0.999，但 B 上逐帧 OOD–|误差| Spearman 两者均 **−0.08**，
     按 OOD 四分位的 MAE 持平（Endo 1.81/1.57/1.61/1.59）→ 误差不是 OOD 帧驱动。
  3. 机制：`k_pred/k_true` 在 B 上 2–3 为 2.01、9–10 为 0.80（Endo），是向训练分布
     压缩；Endo 的 head 在 B 2–5mm 额外加正偏差，DA 的 head 在 3–5mm 恰好抵消。
     用 `pred=(1−α)·β_ref·D·S+α·model_pred` 扫描，Endo B 在 α=0.25 时 lt10 1.153，
     但 company 从 1.109 退化到 1.814 → 队列级 α 需 B 标签，属 test tuning，不可用。
- Decision: 记录结论：**B 的问题不是米制刻度、不是 token 结构 OOD，而是 Endo β 头在 B 上的
  队列级校准偏移**；任何用 B 标签的队列级标定都禁止。主配方保持 DA β-only（D-0085）。
- Alternatives: 继续调 α/队列标定（不可报告）；换深度源（D-0085 已否决）。
- Consequences: 下一臂只能从模型结构/训练目标入手，且训练数据只能来自 company；
  B 只允许在训练完成后评一次。
- Evidence: `output/reports/endo_scale_20261106/findings.md`、
  `output/reports/endo_token_ood_20261106/findings.md`、`scripts/endo_scale_audit.py`、
  `scripts/endo_token_ood.py`
- Owner: twj

### D-0087 — Metric-Beta：β-only + 保留米制刻度的深度 token（预注册 2026-11-06）

- Date: 2026-11-06
- Status: accepted（待训练验证）
- Context: D-0085/D-0086 表明 Endo 的米制刻度本身在 B 上一致（f 比 1.084），但当前
  `prepare_lesion` 把深度通道按帧中位归一化成 `depth/global_q`（[0,4]），模型看不到
  绝对毫米尺度；β 头的 k 校准在 B 上向 company 先验压缩。owner 要求在 beta-only
  上做一个「不丢失米制信息」的改进并训练验证。
- Method: **Metric-Beta**（只改模型输入编码，不改数据/benchmark）：
  - 新增 `model.metric_depth: true`、`model.metric_ref_mm: 20.0`；
  - `V11Net.forward` 用 `D=q` 与 `scalars[7]=q/global_q` 反推 `global_q`，把深度通道 0
    换成 `clip(log(depth_mm / 20), -2, 2)`（无效像素归零），通道 1 仍为有效掩码；
  - 仍是 `fusion: none` 的 β-only，`beta_ref=0.000769`（物理初值），无新增参数；
  - 新配置 `input/configs/experiments/v11/company-betaonly-endo-metric.yaml`，
    40 轮、seed 20261020、workers 4、train_max 10.5；数据仍 `company-v1-endo`。
  - 只允许 `fusion='none'`、非 image_only，且 `data.version` 必须含 `endo`（配置校验）。
- Pre-registered evaluation: 官方 `v11-company-test-v1-endo`（company）+ `benchmark2-endo`
  （B 排除 7–8、R 排除 1–2）。对照：Endo 相对 token s20（company 1.109 / B 1.642）、
  DA β-only s20/s21（company 1.146/1.090、B 1.054/1.029）。
  采纳条件：**company lt10 ≤1.20 且 B lt10 ≤1.20，所有报告档 <2**；若 B >1.30 或
  company >1.25 → 否决。单 seed 先跑；通过才补第二 seed。B 只评一次，不用 B 标签。
- Alternatives: 先加物理锚定/约束 k（会牺牲 company 大档，见 D-0086 α 扫描）；不做。
- Consequences: 这是 D-0085 之后 Endo 的唯一干净再试；若不达标，Endo 线关闭。
- Evidence: `core/config.py`、`core/models/v11.py`、`tests/test_metric_depth.py`、
  `input/configs/experiments/v11/company-betaonly-endo-metric.yaml`
- Owner: twj

### D-0088 — Metric-Beta 否决：保留米制 token 没修好 B，Endo 线关闭（2026-11-06）

- Date: 2026-11-06
- Status: accepted（负结果；D-0087 预注册判据触发）
- Context: D-0087 的 Metric-Beta（β-only + `metric_depth: true`，模型内把深度通道换成
  `log(depth_mm/20)`）40 轮、seed 20261020 完成：
  run `v11-company-betaonly-endo-metric/20261106-205351_b1412548_s20261020_ef4fff`，
  官方 eval `20261106-224053_b1412548_s20261020_4f51ff`，benchmark2-endo（B 排除 7–8、
  R 排除 1–2）。final_ema（ep40）：

  | 指标 | Metric-Beta | Endo 相对 token | DA s20 | DA s21 |
  |---|---:|---:|---:|---:|
  | company lt10 | **1.089** | 1.109 | 1.146 | 1.090 |
  | company case_mae_lt10 | 1.063 | 1.140 | 1.070 | — |
  | B lt10 | **1.722** | 1.642 | **1.054** | **1.029** |
  | B 2–3 | 3.13 | 3.00 | 1.77 | 1.74 |
  | B 3–4 | 1.92 | 1.94 | 0.86 | 0.82 |
  | B 8–9 | 0.88 | 1.56 | 1.87 | 1.64 |
  | B 9–10 | 1.55 | 2.16 | 1.51 | 1.58 |
  | R lt10 | 2.229 | 2.196 | 1.767 | 1.778 |

- Decision: **否决 Metric-Beta**。预注册条件是 company lt10 ≤1.20 且 B lt10 ≤1.20；
  company 1.089 通过，但 **B 1.722 > 1.30 的否决线**，且比 Endo 相对 token 基线
  （1.642）还差。**Endo 线关闭**，主配方保持 DA β-only（D-0081/D-0085）。
- Interpretation: 保留米制深度 token 只把 company 拉到与 DA s21 同级（1.089），
  没有改变 B 的 2–5mm 高估（B 2–3 偏差仍 +3.13）；这与 D-0086 的判断一致——B 的
  瓶颈是 β 头在队列间的 k 校准，不是深度 token 的刻度。继续做 Endo 只剩无标签域
  适应/head 层约束，但 B 7–8 已不报告、2–3 仅 3 例，期望收益低，停止投入。
- Alternatives: 物理锚定/约束 k（预计牺牲 company 大档）；无标签域适应（转导，
  需 owner 授权且超参只能在 company val 选）。
- Consequences: Endo 的负结果保留；后续实验回到 DA β-only 主线（特征侧/448 40 轮
  或报告收尾），不再以 Endo 为候选。
- Evidence: `input/configs/experiments/v11/company-betaonly-endo-metric.yaml`、
  `core/config.py`、`core/models/v11.py`、`tests/test_metric_depth.py`、
  `output/reports/v11-company-betaonly-endo-metric-s20261020_evalmode.json`、
  `output/reports/smallbin_audit_v11-company-betaonly-endo-metric-s20261020/summary.md`
- Owner: twj

### D-0089 — 物理锚定 + 有界残差头（Metric-Anchor，预注册 2026-11-06）

- Date: 2026-11-06
- Status: accepted（待训练验证）
- Context: D-0086/D-0088 定位到 Endo 的 B 误差是 β 头的队列级 k 压缩，而不是米制刻度或
  token OOD；物理项 `β_ref·D·S` 在 B 上更稳（混合 α=0.25 时 B 1.153），但 company 会退化。
  owner 选择先做方案一：把 k 拆成「几何物理项 + 有界特征残差」，阻止 head 在 B 上漂移。
- Method: **Metric-Anchor**（β-only，无新增数据/benchmark）：
  - 保留 `model.metric_depth: true`（log(depth_mm/20) token）；
  - `k_phys = beta_ref · exp(1.0·tanh(z_phys/1.0))`，`z_phys` 只吃
    `[log(D/20), log(S/300)]`（米制深度、像素跨度），即 k_phys/beta_ref ∈ [0.37, 2.72]；
  - 特征残差 `k_res = exp(0.3·tanh(z_res/0.3))`，z_res 来自原 head_beta，
    即残差 ∈ [0.74, 1.35]；
  - `k = k_phys · k_res`，`pred = k·D·S`；phys_net/head 末层零初始化，起点
    = beta_ref·D·S（与普通 β-only 一致）；新增 phys_net 只 3×32+32×1 参数。
  - 配置 `input/configs/experiments/v11/company-betaonly-endo-anchor.yaml`，
    40 轮、seed 20261020、workers 4、train_max 10.5；数据仍 `company-v1-endo`。
- Pre-registered evaluation（官方 company + benchmark2-endo，B 排除 7–8、R 排除 1–2）：
  - **替代 DA**：B lt10 ≤1.10 **且** company lt10 ≤1.15，所有报告档 <2；
  - **否决**：B >1.20 或 company >1.25；
  - 中间区间（B 1.10–1.20、company ≤1.15）：只记作 Endo 内部改进，不换主配方。
  - 单 seed 先跑；只有达到「替代 DA」才补第二 seed。B 只评一次，不用 B 标签。
- Alternatives: 软锚定辅助损失（方案 3，作为 anchor 太硬时的回退）；尺度增广（方案 2）。
- Consequences: 这是 D-0088 后 Endo 的第二次结构性尝试；若仍不达标，Endo 线彻底关闭。
- Evidence: `core/config.py`、`core/models/v11.py`、`tests/test_metric_depth.py`、
  `input/configs/experiments/v11/company-betaonly-endo-anchor.yaml`
- Owner: twj

### D-0090 — Metric-Anchor 否决：B 小档改善、company 大档崩，Endo 线彻底关闭（2026-11-07）

- Date: 2026-11-07
- Status: accepted（负结果；D-0089 预注册判据触发）
- Context: D-0089 的 Metric-Anchor（metric token + 物理 k_phys + ±0.3 有界残差）40 轮、
  seed 20261020 完成（train 20261106-232535_8bb60ae1_s20261020_d5e6d1，finish 2026-11-07 01:13；
  官方 eval 20261107-011359_8bb60ae1_s20261020_119b5e），final_ema（ep40）：

  | 指标 | Metric-Anchor | Endo 相对 token | Metric-Beta | DA s20 / s21 |
  |---|---:|---:|---:|---:|
  | company lt10 | 1.110 | 1.109 | **1.089** | 1.146 / 1.090 |
  | company 9–10 | **2.47** | 1.92 | 1.60 | 1.70 / 1.67 |
  | company worst_bin | **2.47** | 1.90 | 1.84 | 1.76 / 1.93 |
  | B lt10 | **1.491** | 1.642 | 1.722 | **1.054 / 1.029** |
  | B 2–3 | 2.76 | 3.00 | 3.13 | 1.77 / 1.74 |
  | B 3–4 | 1.51 | 1.94 | 1.92 | 0.86 / 0.82 |
  | B 4–5 | 1.14 | 1.52 | 1.52 | 0.50 / 0.49 |
  | B 9–10 | **2.28** | 2.16 | 1.55 | 1.51 / 1.58 |
  | R lt10 | 2.111 | 2.196 | 2.229 | 1.767 / 1.778 |

- Decision: **否决 Metric-Anchor**。预注册「替代 DA」要求 B ≤1.10、company ≤1.15、所有报告档 <2；
  实际 **B 1.491 > 1.20** 且 **company 9–10 2.47 > 2**，两条否决条件都触发。best.pt（ep9）
  company 1.243 / B 1.323 更差。**Endo 线彻底关闭，主配方保持 DA β-only**。
- Interpretation: 锚定确实把 B 2–5mm 拉回来了（B lt10 1.642→1.491，B 2–3 3.00→2.76、
  3–4 1.94→1.51、4–5 1.52→1.14），但 ±0.3 残差也压死了 company 5–10 的修正
  （company 9–10 1.92→2.47、B 9–10 2.16→2.28）。这是 D-0086 所示「company 校准 vs B 稳健」
  的硬 tradeoff：同一个全局模型无法同时满足两端；DA 的特征表示恰好两者都过，Endo 不行。
  软锚定只会落在 Endo 与 Anchor 之间，不可能同时达到 B ≤1.10 和 company 全档 <2。
- Alternatives: 软锚定（预期不足）；尺度增广（不确定，收益低）；无标签域适应（转导，需授权）。
- Consequences: Endo 三次尝试（相对 token / 米制 token / 物理锚定）全部否决；后续回到
  DA β-only 主线（特征侧/448 40 轮或报告收尾），不再以 Endo 为候选。
- Evidence: `input/configs/experiments/v11/company-betaonly-endo-anchor.yaml`、
  `core/models/v11.py`、`tests/test_metric_depth.py`、
  `output/reports/v11-company-betaonly-endo-anchor-s20261020_evalmode.json`、
  `output/reports/smallbin_audit_v11-company-betaonly-endo-anchor-s20261020/summary.md`
- Owner: twj

### D-0091 — Endo 米制焦距头：把尺度未知量收敛到 f（预注册，2026-11-08）

- Date: 2026-11-08
- Status: pre-registered（第四次 Endo 结构性尝试）
- Context: D-0090 后复盘三条 Endo 失败路径：相对 token（B 1.642）、米制 token（B 1.722）、
  物理锚定（B 1.491 但 company 9–10 2.47）。共同点是让外观头自由决定尺度：旧 β 头
  允许 β 变化 e^±2.06≈0.13–7.8×，在 B 上学成尺寸相关压缩（B 2–3 k_pred/k_true 2.01、
  9–10 0.80）。米制刻度诊断给出关键事实：B/company 隐含 f 比值 **1.084 [0.944, 1.224]**，
  即米制深度下 B 与 company 的物理焦距只差 ~8%，真正需要跨队列变化的量只有 f。
- Hypothesis: 令 size = D_mm·S_px/f，`log f = log(1/beta_ref) + c·tanh(z_feat/c)`，
  紧界 c=0.15 只允许 ±16% 的焦距差。这样外观头无法再做尺寸相关压缩，而 company↔B
  的 1.08× 仍在界内，两端可同时保住。D·S 仍用米制（metric_depth token）；不再保留
  Metric-Anchor 的 k_phys(log D,log S) 项（D-0090 中它压死 company 大档）。
- Architecture: 复用现有 `head_beta`（2d→64→1，末层零初始化），因此**不新增参数**、
  关闭时与旧 β-only 逐位一致；`f_ref=1/beta_ref`，零初始化起点 = `beta_ref·D·S`。
  `metric_focal_head` 与 `metric_anchor` 互斥，仅支持 `fusion='none'`、`metric_depth=true`、
  Endo 数据版本。配置 `input/configs/experiments/v11/company-betaonly-endo-focal.yaml`，
  seed 20261020、40 轮、其余同 Metric-Beta。
- Gate（先登记，单 seed 先跑）:
  - 替代 DA：company 全档 <2mm **且** B（剔 7–8）全档 <2mm **且** B lt10 ≤1.10；
  - 否决：任一 company 档 >2mm **或** B lt10 >1.20；
  - 中间区间（B 1.10–1.20、company 全档 <2）只记作 Endo 内部改进，不换主配方；
  - 只有达到「替代 DA」才补第二 seed；B 只评一次，不用 B 标签调参。
- Alternatives: 焦距残差加 ±0.1 逐帧乘性项（先不加，保持单变量）；按队列无监督标定 f
  （转导，接近 test-tuning，仅作最后退路）。
- Consequences: 若失败，Endo 线第四次否决并永久关闭，回到 DA β-only 主线。若通过，
  需复核 seed 20261021 并在 benchmark2 上重出三队列逐档图。
- Evidence: `core/config.py`、`core/models/v11.py`（`focal_beta`）、
  `tests/test_metric_depth.py`、`input/configs/experiments/v11/company-betaonly-endo-focal.yaml`、
  `output/reports/endo_scale_20261106/findings.md`、`output/reports/endo_token_ood_20261106/findings.md`
- Owner: twj

### D-0092 — 米制焦距头 v1 否决：紧界与逐帧 k 跨度不匹配（2026-11-08）

- Date: 2026-11-08
- Status: accepted（负结果；D-0091 的预注册门槛在训练中即已不可能达到，run 于 epoch 20 停止）
- Context: D-0091 的 raw-f_px 紧界头（`c=0.15`）训练到 epoch 18 时
  val `mae_lt10=2.859`、`robust_worst_bin=4.925`；同数据前三个 Endo run（相对 token /
  metric token / anchor）epoch 40 的 val 为 **1.30 / 1.90** 左右。原因已量化：
  - 逐帧最优 `k=truth/(D·S)` 的 p10–p90 在 val 为 **0.39–2.41×** 中位（0.00029–0.00179
    vs 0.00074），test_company 为 **0.47–2.12×**；±0.15 紧界只能覆盖 0.84–1.16×。
  - 固定 `beta_ref` 的常数预测 MAE 为 val 3.511 / test_company 2.232；即使取最优常数
    修正仍为 val 2.656 / test 2.216 —— 逐帧修正量不是“小残差”。
  - 旧 β 头 `k_bound=2.06`（≈0.13–7.8×）的大范围正是用来吸收这个逐帧方差的；
    队列中位 f 只差 8–10% 不代表逐帧 f 只差 8–10%。另：content 宽度 company
    p10–p90 849–1266（全距 786–2529），R 中位 1350；raw `f_px` 紧界还混入了分辨率。
- Decision: **否决 v1 并停止训练**；该 run 不当作 completed 结果（manifest 仍为
  `running`，SIGTERM 停止，见 STATE §7）。Endo 主配方仍为 DA β-only。
- Interpretation: “固定 f_ref + 紧界”的前提被证伪。下一版必须允许宽范围的逐帧修正，
  同时用结构约束阻止外观头做队列级压缩（B 的失败模式）。
- Consequences: v2 设计（待 owner 确认后预注册）：`f/W` 归一化 + 几何宽界
  （D,S,W,H,S/W,S/H,深度统计）+ 外观窄残差（有界 bbox/分割修正）。若 v2 仍失败，
  Endo 线关闭。
- Evidence: run `v11-company-betaonly-endo-focal/20261108-152112_3f6cc575_s20261020_4deeab`
  （epoch 18 val）、`output/logs/endo_focal_20261108.*`、`core/models/v11.py`、
  `core/config.py`、`input/configs/experiments/v11/company-betaonly-endo-focal.yaml`。
- Owner: twj

### D-0093 — Endo 米制焦距头 v2：几何宽界 + 外观窄残差（预注册，2026-11-08）

- Date: 2026-11-08
- Status: pre-registered（第五次 Endo 结构性尝试；v1 见 D-0092）
- Context: v1 的紧界（±0.15 raw f_px）在 epoch 18 就 val lt10 2.86（对照 1.30），
  因为逐帧 `k=truth/(D·S)` 的 p10–p90 是 0.39–2.41×（val）/0.47–2.12×（test），
  不是 ±16%。同时 company 的 content 宽度 p10–p90 849–1266，raw f_px 紧界还混入
  分辨率；旧宽头能拟合 company 但外观头会在 B 上做队列级压缩。v2 的目标是：
  **几何负责宽范围相机/深度尺度，外观只做窄范围框/分割修正。**
- Design:
  - 预测：`pred = beta_ref · D · S · exp(r_app − c_phys·tanh(z_phys/c_phys))`。
    等价于 `pred = D·(S/W)/(f/W)`，其中 `log(f/W) = −log(beta_ref) − log(W) + c_phys·tanh(...)`；
    W 在基座中与 D·S 约掉，零初始化严格回到 `beta_ref·D·S`。
  - 几何头输入（10 维）：`log(D/20)`、`log(S/300)`、`log(W/1258)`、`log(H/1080)`、
    `log((S/W)/0.24629)`、`log((S/H)/0.30)`、`log(框宽/框高)`、
    深度图均值/std/有效占比。H 由 `scalars[:,5]=H/W` 与 batch W 还原。
  - `c_phys=2.0`（乘数 0.135–7.39×），几何头 MLP 10→64→1、末层零初始化。
  - 外观残差复用 `head_beta`（池化特征）：`r_app = c_app·tanh(z_app/c_app)`，
    `c_app=0.3`（0.741–1.350×），末层零初始化。
  - 无新数据、无增广；新参数仅几何 MLP（约 0.7k）。
  - 配置：`input/configs/experiments/v11/company-betaonly-endo-focal2.yaml`，
    seed 20261020、40 轮、其余同 v1。
- Gate（单 seed 先跑）:
  - 替代 DA：company 全档 <2mm **且** B（剔 7–8）全档 <2mm **且** B lt10 ≤1.10；
  - 否决：任一 company 档 >2mm **或** B lt10 >1.20；
  - 中间区间只记作 Endo 内部改进，不换主配方；只有替代 DA 才补第二 seed。
- Risks: 若逐帧 k 的跨度主要来自框/标签噪声而非物理量，几何头会欠拟合，company 退化；
  若 B 的尺度差异不能被 W/H/深度统计解释，B 仍失败 → 说明米制深度+几何不足，Endo 关闭。
- Alternatives: 放宽 `c_app` 到 0.5；bbox jitter；若仍失败则关闭 Endo。
- Evidence: `core/config.py`、`core/models/v11.py`（`focal_v2_scale`、
  `_focal_phys_inputs`）、`tests/test_metric_depth.py`、
  `input/configs/experiments/v11/company-betaonly-endo-focal2.yaml`。
- Owner: twj

### D-0094 — 米制焦距头 v2 结果：B 大幅改善、company 两尾档仍超 2mm，按预注册否决（2026-11-08）

- Date: 2026-11-08
- Status: accepted（负结果；D-0093 门槛未过）
- Context: v2 40 轮完成（train 20261108-164046_0a821430_s20261020_12a397，19:08 完成；
  company 官方 eval 20261108-190943_0a821430_s20261020_9d1848）。final_ema（ep40）：

  | 指标 | v2 | 旧 Metric Endo | DA β-only s21 |
  |---|---:|---:|---:|
  | company lt10 | **1.062** | 1.089 | 1.090 |
  | company 2–3 | 2.126 | 1.84 | 1.93 |
  | company 9–10 | **2.593** | 1.60 | 1.56 |
  | company worst_bin | 2.593 | 1.84 | 1.93 |
  | B lt10 | **1.172** | 1.722 | 1.029 |
  | B 2–3 | 2.013 | 3.13 | 1.74 |
  | B 3–4 | 1.214 | 1.92 | 0.82 |
  | B 4–5 | 0.869 | 1.52 | 0.49 |
  | B 5–6 | 1.283 | 1.56 | 1.56 |
  | B 8–9 | 1.150 | 0.88 | 1.63 |
  | B 9–10 | 1.271 | 1.55 | 1.57 |
  | R lt10 | 2.001 | 2.229 | 1.778 |

- Decision: **按 D-0093 否决 v2**：company 2–3（2.126）与 9–10（2.593）>2mm，
  B 2–3（2.013）>2mm，且 B lt10 1.172 >1.10 → 不替换 DA。
- Interpretation: v2 是目前最好的 Endo（company lt10 1.062 为全项目最好，B 1.172 为 Endo 最好；
  B 3–5、8–10 全面改善），说明“几何宽界 + 外观窄残差”方向正确，B 的队列压缩基本被修掉。
  但 company 两尾档出现压缩：2–3 偏差 **+2.13**、9–10 偏差 **-2.59**；旧宽头 company 9–10
  是 1.60。最可能的约束是 `c_app=0.3` 太窄，外观头无法修尾档的框/分割误差；几何头虽然能
  看 S/D/W，但不像外观那样能判断框的松紧。
- Consequence: D-0093 原写“v2 失败则关闭 Endo”。若 owner 授权一次例外，v3 只改一个变量：
  `metric_focal_app_bound: 0.3 → 0.6`，其余完全同 v2，门槛同 D-0093；若 v3 仍不过则关闭。
  未经授权不再启动。
- Evidence: run `v11-company-betaonly-endo-focal2/20261108-164046_0a821430_s20261020_12a397`、
  `output/reports/v11-company-betaonly-endo-focal2-s20261020_evalmode.json`、
  `output/reports/smallbin_audit_v11-company-betaonly-endo-focal2-s20261020/summary.md`、
  `output/charts/v11-company-betaonly-endo-focal2-s20261020.png`。
- Owner: twj

### D-0095 — Endo 焦距头 v3：唯一放宽外观残差 c_app 0.3→0.6（预注册，2026-11-08）

- Date: 2026-11-08
- Status: pre-registered（owner 授权 D-0094 后的一次例外；v3 仍不过则关闭 Endo）
- Context: D-0094 的 v2 是 Endo 目前最好：company lt10 **1.062**（全项目最好）、
  B lt10 **1.172**（Endo 最好），B 3–5/8–10 全面改善；但 company 2–3 **2.126**、
  company 9–10 **2.593**、B 2–3 **2.013** 超 2mm。审计显示 company 2–3 偏差 **+2.13**、
  9–10 偏差 **-2.59**，是尾档压缩；geometry 头看得到 S/D/W，但判断不了框的松紧，
  最可能的约束是 `c_app=0.3`（外观乘性修正仅 0.74–1.35×）。
- Hypothesis: 只放宽外观残差到 `c_app=0.6`（0.55–1.82×），让外观头修尾档框/分割误差；
  几何头（`c_phys=2.0`）仍锚定队列尺度，B 不应退回旧宽头的队列压缩。
- Design: 与 D-0093/v2 **完全一致，唯一变量** `metric_focal_app_bound: 0.3 → 0.6`；
  配置 `input/configs/experiments/v11/company-betaonly-endo-focal3.yaml`，
  seed 20261020、40 轮、其余同 v2。
- Gate（同 D-0093，单 seed）:
  - 替代 DA：company 全档 <2mm **且** B（剔 7–8）全档 <2mm **且** B lt10 ≤1.10；
  - 否决：任一 company 档 >2mm **或** B lt10 >1.20；
  - 中间区间只记作 Endo 内部改进；只有替代 DA 才补第二 seed。
- Consequences: 若 v3 仍失败，Endo 线关闭，主配方保持 DA β-only；不再追加 v4。
- Evidence: `DECISIONS.md` D-0093/D-0094、`core/models/v11.py`、
  `input/configs/experiments/v11/company-betaonly-endo-focal3.yaml`、
  `output/reports/v11-company-betaonly-endo-focal2-s20261020_evalmode.json`。
- Owner: twj

### D-0096 — 米制焦距头 v3 结果：B 全档 <2 但 company 双尾仍超，按预注册关闭 Endo（2026-11-08）

- Date: 2026-11-08
- Status: accepted（负结果；D-0095 门槛未过；Endo 线关闭）
- Context: v3（仅 `c_app` 0.3→0.6）40 轮完成（train 20261108-194051_84c11f03_s20261020_a446cd，
  22:11 完成；company 官方 eval 20261108-221118_84c11f03_s20261020_a33152）。final_ema（ep40）：

  | 指标 | v3（c_app=.6） | v2（c_app=.3） | DA β-only s21 |
  |---|---:|---:|---:|
  | company lt10 | 1.069 | **1.062** | 1.090 |
  | company 2–3 | **2.184** ❌ | 2.126 ❌ | 1.93 |
  | company 9–10 | **2.180** ❌ | 2.593 ❌ | 1.56 |
  | company worst_bin | 2.184 | 2.593 | 1.93 |
  | B lt10 | 1.200 | **1.172** | 1.029 |
  | B 2–3 | **1.916** | 2.013 | 1.74 |
  | B 3–4 / 4–5 / 5–6 | 1.163 / 0.887 / 1.463 | 1.214 / 0.869 / 1.283 | 0.82 / 0.49 / 1.56 |
  | B 8–9 / 9–10 | 1.153 / 1.513 | 1.150 / 1.271 | 1.63 / 1.57 |
  | R lt10 | 1.940 | 2.001 | 1.778 |

- Decision: **按 D-0095 否决 v3**：company 2–3（2.184）与 9–10（2.180）>2mm；
  B 虽然报告档全部 <2（最差 2–3 1.916），但 B lt10 **1.200 >1.10**（且比 D-0093 的硬否决线
  1.20 高 0.0001）→ 不替换 DA。**Endo 线关闭，不再追加 v4。**
- Interpretation: 放宽 `c_app` 确实修好了 company 9–10（2.593→2.180）和 B 2–3
  （2.013→1.916），并让 B 报告档全部 <2；但没有修好 company 2–3（2.126→2.184），
  还把 B lt10 从 1.172 推高到 1.200（B 5–6 1.283→1.463、9–10 1.271→1.513）。
  两个尾档现在都卡在 ~2.18，方向相反（2–3 高估、9–10 低估），说明剩余误差不是
  一个全局乘性边界能修的：继续调 `c_app` 只会在 company 与 B 之间来回交换，属于
  test-tuning。v1→v2→v3 的结论是：米制深度 + 几何宽界能修队列尺度（B lt10
  1.49→1.17–1.20），但修不了 company 双尾的标签/框/病灶形态歧义。
- Consequence: Endo 不再作为 DA 的候选；主配方保持 DA β-only（company 1.090 / B 1.029，
  报告档全 <2）。若未来重启 Endo，必须是新的预注册假设（例如尾档标签审计或
  成对尺度监督），不能继续在 benchmark 上调 `c_app`。
- Evidence: run `v11-company-betaonly-endo-focal3/20261108-194051_84c11f03_s20261020_a446cd`、
  eval `20261108-221118_84c11f03_s20261020_a33152`、
  `output/reports/v11-company-betaonly-endo-focal3-s20261020_evalmode.json`、
  `output/reports/smallbin_audit_v11-company-betaonly-endo-focal3-s20261020/summary.md`、
  `output/charts/v11-company-betaonly-endo-focal3-s20261020.png`。
- Owner: twj
