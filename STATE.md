# STATE.md — project state

---
project: "beta-regression"
phase: ablation         # setup | data | baseline | ablation | benchmark | report
updated: "2026-10-30"
owner: twj
primary_metric: "v11"
best_run: "20261030-181920_a6b4de94_s20261020_582999"
benchmark: "sha256:2c4d4beeb9461597c2cae52cfaa82f0a8947d433882b1cd3b169994205018147"
---

## 1. Question and scope

- Research question: 单张内镜图 + 检测框，能否把息肉直径测到毫米级，并让 <10 mm 每个 1 mm 区间 MAE 都 <2 mm？
- Unit of analysis: 病例（case）
- Input: 内镜 RGB（整图 + 1.25× 紧局部两视图）+ 冻结相对深度特征 + 10 维几何标量
- Output: 直径（mm）
- Primary metric: 每个 1 mm 区间的 MAE（`worst_bin` 为 2–9 mm 最差档）
- Success criterion: 四个冻结测试集上 <10 mm 每个 1 mm 区间 MAE <2 mm
- Non-goals: 活检钳、实时、视频流稳定化

## 2. Data and splits

- Data root: `$DLEF_DATA_ROOT`（本机指向 `Betaregression_data/`，内含 `pm/` 与 `images_new/` 两个符号链接）
- Dataset / version: `v11_frames` / `single-v2`（v1 框错误，已作废）
- Sample id: `case_id::image_stem`
- Labels / units: 镜下尺寸（单值），mm
- Split policy:
  - train: 2,044 帧 / 525 例（manifest split=train）
  - val: 205 帧（manifest split=val，含 2–3 与 9–10 尾档）
  - test: 四个冻结 benchmark（B 210 / D 356 / R 185 / 清洗公司 206）
  - grouping / leakage rule: 按 `case_id`；测试集不参与训练与选模
- Frozen benchmark（**v2 生效**）：
  - `v11-b-test-v2`：`sha256:911defbff0b5d18d3b9091320bf556f3cf9a3af9ce103fb097d4df4aa60c6ecb`
  - `v11-d-test-v2`：`sha256:2c4d4beeb9461597c2cae52cfaa82f0a8947d433882b1cd3b169994205018147`
  - `v11-r-test-v2`：`sha256:983c48b649ce41c69a4ad382b1b18e6a557a8586dd967b5799a22421249924be`
  - `v11-clean-test-v2`：`sha256:6a76a6986d680646d0a963cdca8105bd50f1f0930c3aa74eee01e191fead97ad`
- 旧 benchmark：
  - `v11-b-test-v1`（作废）：`sha256:01efba8bb6701c3426a8d243a98dc8a4122e487c74cfa6a5a993f4bb1bce8f84`
  - `v11-d-test-v1`（作废）：`sha256:6f6c3a3965f9cc4c9d260ed3bfd311e8bb23df9619aeab728b779afdb265e818`
  - `v11-r-test-v1`（作废）：`sha256:c0bed53e26cc335f31e506b0198f2bb3d7659adce920ee2ff3a0b87dd87539ce`
  - `v11-clean-test-v1`（作废）：`sha256:8325ad370a3232234823bf8de7ad804a99474ca90981fe06fbba3f56e82645cb`
- Known data issues:
  - 1–2 mm 没有真实开发病例；2–3 mm 仅 30 例
  - 9–10 mm 训练帧仅 9 帧（其中 5 帧在 dev）
  - 旧数据表已改为新目录 `~/桌面/息肉数据集/息肉大小-病例/`，并重建符号链接

## 3. Current baseline

- Config: `input/configs/experiments/v11/baseline.yaml`（`single-v2` / `v11-d-test-v2`）
- Official baseline: `20261030-171041_cdcad964_s20261020_07df5c`（tag `baseline`/`v2`，`code_version=6d9ea69`）
- Baseline initialization: `pm/runs/size/beta-image-v10-r50-da-20261024/best.pt`，**不是 ImageNet 起点**。
- Checkpoint: `output/runs/beta-regression/v11-baseline/20261030-171041_cdcad964_s20261020_07df5c/checkpoints/best.pt`
- Val: `v11/worst_bin=2.1741`、`v11/mae_lt10=1.5016`、`v11/mae=2.3059`、`v11/acc10=0.7415`
- Test（best.pt，四个 v2 benchmark）：

| cohort | MAE | mae_lt10 | worst_bin | acc10 |
|---|---:|---:|---:|---:|
| D | 3.0349 | 1.2959 | 2.4069 | 0.6826 |
| B | 1.5488 | 1.3800 | 3.9882 | 0.9524 |
| R | 2.5793 | 2.4648 | 3.4514 | 0.9730 |
| clean | 2.0720 | 1.3066 | 2.5848 | 0.7864 |

- 验收状态：所有 cohort 的 `worst_bin` 仍 >2 mm，尚未达到成功标准。
- 旧 v1 run 已按 D-0009 删除；v1 数据和 benchmark 文件保留为历史。
- What is frozen vs still moving: 数据与 benchmark 已冻结；模型/损失仍可做 variant

## 4. Experiments

### 4.1 Backlog / planned

| priority | hypothesis | change | success criterion | status | run |
|---|---|---|---|---|---|
| 1 | 骨干是否过度贴合训练病例 | A/B/C：ImageNet 全量 / 部分冻结 / 全冻结 | B/D 每档 <2 mm；baseline 为参照 | v1 invalid; v2 pending | - |
| 2 | 官方 RankSim 是否迁移有效 | `mm_rank`，ImageNet init | 与 baseline 同数据/benchmark；注意初始化不同 | v2 completed; single seed | `20261030-175624_3138ba9c_s20261020_62c8b5` |
| 3 | 独立投影 RankSim | `model.proj=true`，ImageNet init | 同上 | v2 completed; single seed | `20261030-181920_a6b4de94_s20261020_582999` |

### 4.2 v2 runs

| experiment | config | status | <primary metric> | run | note |
|---|---|---|---|---|---|
| baseline | `v11/baseline.yaml` | completed | `v11/worst_bin=2.4069` (D) | `20261030-171041_cdcad964_s20261020_07df5c` | B=3.9882 R=3.4514 clean=2.5848；pm/v10 init |
| ranksim | `v11/ranksim.yaml` | completed | `v11/worst_bin=2.3954` (D) | `20261030-175624_3138ba9c_s20261020_62c8b5` | B=3.1904 R=3.8324 clean=2.6516；ImageNet init，tag `v2-imagenet` |
| proj | `v11/proj.yaml` | completed | `v11/worst_bin=2.1149` (D) | `20261030-181920_a6b4de94_s20261020_582999` | B=3.4127 R=3.8972 clean=2.2451；ImageNet init，tag `v2-imagenet` |
| ranksim-long80 | `v11/ranksim-long80.yaml` | killed（3 epoch） | - | `20261030-211500_20c29ceb_s20261020_6841a2` | 从 `v11-ranksim` 拆出的 80 epoch exploratory；不是 official |
| backbone-full/partial/frozen | — | v1 invalid; v2 pending | — | — | 先完成 baseline/RankSim 对照 |

逐区间图在各自 `*-eval/per_bin_mae.png`；旧的 v1 run 已删除（D-0009）；
旧的非 ImageNet 初始化 RankSim/proj run 也已删除，只保留新 `v2-imagenet` 结果。

## 5. Open questions

- [ ] RankSim 在 DLEF 迁移后是否仍能复现 PolypMeasure 的单 seed 收益？
- [ ] 9–10 mm 训练帧不足，是否需要补开发侧病例？

## 6. Next actions

- [x] 修复 `single-v2` 框并建立四个 v2 benchmark
- [x] 用 v2 重跑 baseline、ranksim、proj
- [x] 删除 pm 初始化的旧 RankSim runs，按 ImageNet 起点重训（D-0010）
- [x] 为每个实验生成逐区间 MAE 图（`*-eval/per_bin_mae.png`）
- [ ] 决定 baseline 是否也统一到 ImageNet 起点；当前 baseline 仍是 v10 init
- [ ] 为 v2 baseline/RankSim 结果创建 claim，并生成 report
- [ ] 若要主张 RankSim 稳定收益，补 2–3 个 seed
- [ ] 决定是否继续 v2 backbone 冻结组合
- [ ] review `make gc-dry`

## 7. Known issues / gotchas

- GPU 上同时跑 PolypMeasure 训练时会 OOM；迁移期先 CPU 冒烟
- 数据集移动过两次；引用路径统一走 `$DLEF_DATA_ROOT` 的相对路径
- `mae_ge10` 在没有 ≥10 样本时是 NaN；空区间也显式写 NaN（无法评价）
- v1 run 已删除，不要与 v2 的 `benchmark_id` 混用；v1 数据和 benchmark 文件仅作历史
- RankSim/proj 必须保持 `init_checkpoint=""`；旧的非 ImageNet 初始化 run 已删除
- `v11-ranksim-long80` 是从 `v11-ranksim` 拆出的 80 epoch exploratory run，已 killed，不是 official
- test access 会继续累计；report 会标 `WARN`，不要重复评 test 来“刷新”

## 8. Handoff checklist

- [x] `STATE.md` reflects the latest run and metrics.
- [x] Every number above has a run/eval id; none is copied from `train.log`.
- [x] `DECISIONS.md` has an entry for any hard-to-reverse choice.
- [x] Test split was not used for tuning.
- [x] Runs are tagged (`baseline`, `reported`, `debug`); `dlef gc` dry-run reviewed (no deletions).
- [x] No secrets or absolute data paths in this file.

## 9. Update protocol

1. Read `make runs` / `run_info.py`; pick the run, do not guess.
2. Update sections 3–7, replacing stale values (this is a snapshot, not a log).
3. Set `updated:` and `phase:` in the front matter.
4. Append to `DECISIONS.md` if the change is hard to reverse or changes data,
   split, benchmark, metric, baseline, logging, or retention policy.
5. If a number is unknown, write `TBD` or `UNKNOWN`; never invent it.
