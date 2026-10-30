# STATE.md — project state

---
project: "beta-regression"
phase: baseline         # setup | data | baseline | ablation | benchmark | report
updated: "2026-10-30"
owner: twj
primary_metric: "v11"
best_run: "-"           # official baseline pending; old run is exploratory
benchmark: "sha256:6f6c3a3965f9cc4c9d260ed3bfd311e8bb23df9619aeab728b779afdb265e818"
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
- Dataset / version: `v11_frames` / `single-v1`
- Sample id: `case_id::image_stem`
- Labels / units: 镜下尺寸（单值），mm
- Split policy:
  - train: 2,044 帧 / 525 例（manifest split=train）
  - val: 205 帧（manifest split=val，含 2–3 与 9–10 尾档）
  - test: 四个冻结 benchmark（B 210 / D 356 / R 185 / 清洗公司 206）
  - grouping / leakage rule: 按 `case_id`；测试集不参与训练与选模
- Frozen benchmark:
  - `v11-b-test-v1`：`sha256:01efba8bb6701c3426a8d243a98dc8a4122e487c74cfa6a5a993f4bb1bce8f84`
  - `v11-d-test-v1`：`sha256:6f6c3a3965f9cc4c9d260ed3bfd311e8bb23df9619aeab728b779afdb265e818`
  - `v11-r-test-v1`：`sha256:c0bed53e26cc335f31e506b0198f2bb3d7659adce920ee2ff3a0b87dd87539ce`
  - `v11-clean-test-v1`：`sha256:8325ad370a3232234823bf8de7ad804a99474ca90981fe06fbba3f56e82645cb`
- Known data issues:
  - 1–2 mm 没有真实开发病例；2–3 mm 仅 30 例
  - 9–10 mm 训练帧仅 9 帧（其中 5 帧在 dev）
  - 旧数据表已改为新目录 `~/桌面/息肉数据集/息肉大小-病例/`，并重建符号链接

## 3. Current baseline

- Config: `input/configs/experiments/v11/baseline.yaml`
- Official baseline: 尚未重跑（先提交项目代码，再重跑）
- Exploratory run: `20261030-140626_f0c990c1_s20261020_fd2ec8`
- Checkpoint: `output/runs/beta-regression/v11-baseline/20261030-140626_f0c990c1_s20261020_fd2ec8/checkpoints/last.pt`
- Tags: `baseline`（exploratory；`code_version=no-git`）
- Val metrics: `v11/mae=2.4022`、`v11/worst_bin=2.287`、`v11/mae_lt10=1.38453`、`v11/acc10=0.721951`
- Test metrics: 未评（冻结 benchmark 已建，但旧 run 不作为正式 baseline）
- Why exploratory: run 在 seed/RNG 修复和项目 git commit 之前启动；checkpoint 只含旧版
  Python RNG，无法按当前代码完整恢复。
- What is frozen vs still moving: 数据与 benchmark 已冻结；模型/损失仍可做 variant

## 4. Experiments

### 4.1 Backlog / planned

| priority | hypothesis | change | success criterion | status | run |
|---|---|---|---|---|---|
| 1 | 骨干是否过度贴合训练病例 | A/B/C：ImageNet 全量 / 部分冻结 / 全冻结 | B/D 每档 <2 mm，A 为参照 | in progress | `20261030-140626_f0c990c1_s20261020_fd2ec8`（exploratory） |
| 2 | 官方 RankSim 是否迁移有效 | `mm_rank` | 与 A 同口径对比 | planned | - |
| 3 | 独立投影 RankSim | `model.proj=true` | 同上 | planned | - |

### 4.2 Completed

| experiment | config | status | <primary metric> | run | note |
|---|---|---|---|---|---|
| baseline（exploratory） | `v11/baseline.yaml` | completed | `v11/worst_bin=2.287` | `20261030-140626_f0c990c1_s20261020_fd2ec8` | 旧代码、`code_version=no-git`；需提交后重跑 |
| backbone-full | `v11/backbone-full.yaml` | completed | `v11/worst_bin=3.187` (D, last.pt) | `20261030-143509_cf99d0ab_s20261020_cf85b9` | 旧代码；只有 `last.pt`，需重跑 |
| backbone-partial | `v11/backbone-partial.yaml` | completed | `v11/worst_bin=2.866` (D, last.pt) | `20261030-145437_a6164b31_s20261020_87ae17` | 旧代码；只有 `last.pt`，需重跑 |
| backbone-frozen | `v11/backbone-frozen.yaml` | running | - | `20261030-151846_b122017e_s20261020_2cc3ad` | DLEF `937f11e` 启动，已有 `best.pt` |

## 5. Open questions

- [ ] RankSim 在 DLEF 迁移后是否仍能复现 PolypMeasure 的单 seed 收益？
- [ ] 9–10 mm 训练帧不足，是否需要补开发侧病例？

## 6. Next actions

- [ ] 提交项目代码，消除 `code_version=no-git`
- [ ] 用 DLEF `937f11e` 重跑 `backbone-full` / `backbone-partial`，确保产出 `best.pt`
- [ ] 重跑 `v11/baseline` 正式训练（40 轮，GPU，`--tag baseline`）
- [ ] 用四个 benchmark 评正式 baseline，记录 run/eval id
- [ ] 依次跑 ranksim / proj 两个 variant
- [ ] 更新 `STATE.md` 的 official baseline 数字与 phase

## 7. Known issues / gotchas

- GPU 上同时跑 PolypMeasure 训练时会 OOM；迁移期先 CPU 冒烟
- 数据集移动过两次；引用路径统一走 `$DLEF_DATA_ROOT` 的相对路径
- `mae_ge10` 在没有 ≥10 样本时是 NaN
- best checkpoint 修复在 DLEF `937f11e`；此前的 run 只有 `last.pt`，
  需用新代码重跑才有 `best.pt`；`backbone-frozen` 启动于修复后

## 8. Handoff checklist

- [x] `STATE.md` reflects the latest run and metrics.
- [x] Every number above has a run/eval id; none is copied from `train.log`.
- [x] `DECISIONS.md` has an entry for any hard-to-reverse choice.
- [x] Test split was not used for tuning.
- [ ] Runs are tagged (`baseline`, `reported`, `debug`); `dlef gc` dry-run reviewed.
- [x] No secrets or absolute data paths in this file.

## 9. Update protocol

1. Read `make runs` / `run_info.py`; pick the run, do not guess.
2. Update sections 3–7, replacing stale values (this is a snapshot, not a log).
3. Set `updated:` and `phase:` in the front matter.
4. Append to `DECISIONS.md` if the change is hard to reverse or changes data,
   split, benchmark, metric, baseline, logging, or retention policy.
5. If a number is unknown, write `TBD` or `UNKNOWN`; never invent it.
