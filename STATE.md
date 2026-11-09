# STATE.md — project state

---
project: "beta-regression"
phase: ablation         # setup | data | baseline | ablation | benchmark | report
updated: "2026-11-08"
owner: twj
primary_metric: "v11"
best_run: "20261102-182028_2351695d_s20261020_c66b85"
benchmark: "sha256:2c4d4beeb9461597c2cae52cfaa82f0a8947d433882b1cd3b169994205018147"
---

## 1. Question and scope

- Research question: 单张内镜图 + 检测框，能否把息肉直径测到毫米级，并让 <10 mm 每个 1 mm 区间 MAE 都 <2 mm？
- Unit of analysis: 病例（case）
- Input: 内镜 RGB（整图 + 1.25× 紧局部两视图）+ 冻结相对深度特征 + 10 维几何标量
- Output: 直径（mm）
- Primary metric: 每个 1 mm 区间的 MAE（`worst_bin` 为 2–9 mm 最差档）
- Success criterion（**owner 2026-11-05 更正，见 D-0067**）：`benchmark2` 的 **company（company-v1
  `test_company`）+ B（single-v2 `test_b`）** 上，<10 mm 每个 1 mm 区间 MAE <2 mm；R **不要求**；
  B 7–8mm 只有 `polyp-size-04` 一例（5 帧），**自 2026-11-06 起按 D-0083 从 benchmark2 报告剔除**，
  冻结 `v11-b-test-v2` 与官方 eval 不变；旧的 single-v2 `test_d` **不是**目标。
  `PROJECT.yaml:9` 仍写着旧的四集口径，待 owner 确认后同步
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
- **PM 侧 ranksim 训练数据血统（2026-11-02 核对）**：两代划分不同——
  `v11-rank2-{official,gaussian,proj}` 用 2044/525 + 205 val，**与当前 single-v2 逐帧同一批**；
  `v11-mm-bin-ranksim` 用 2154/551 + 95 val = 当前 train **+ 110 帧/26 例**（来自当前 val，
  多为 newdata，含 2mm×6、3mm×37、9–10mm×6 等稀缺档），其 val 为当前 val 的 95 帧子集。
  两代四个测试集一致（B 210 / D 356 / R 185 / clean 206）。→ 老版 PM run 已含「部分
  val-merge」，与当前 baseline 直接对比本质上是跨数据划分比较。
- **数据结构事实（2026-11-01 审计）**：clean 测试集 206 帧是 **D 测试集 356 帧的子集**
  （帧级 100% 重叠、同标签、clean 无独有病例）——「四个独立测试集」实为
  **三个独立集（B/D/R）+ 一个 D 子集**；解读 clean 成绩时不得当作独立证据。
  其余 split 间无任何病例重叠（训练无泄漏）；同病例跨帧标签一致；bbox 抽查合法。
- Known data issues:
  - 1–2 mm 没有真实开发病例；2–3 mm 仅 30 例
  - 9–10 mm 训练帧仅 9 帧（其中 5 帧在 dev）
  - 旧数据表已改为新目录 `~/桌面/息肉数据集/息肉大小-病例/`，并重建符号链接

## 3. Current baseline

- **最优配方（定稿）**：`对数空间损失 + RankSim γ=2.0 + valmerge(50 例 dev) + shuffle +
  40 轮 cosine + full backbone + EMA 优先选模`
  —— `v11-ranksim-log-g20`，3 seed（20261020/21/22）
- **3 seed 官方 eval（修复后口径，eval 模式）**：
  - B `<10mm` MAE = **1.113**（1.066 / 1.150 / 1.123），worst_bin = 2.708
  - D `<10mm` MAE = **1.111**（1.097 / 1.135 / 1.102），worst_bin = 2.710
  - R `<10mm` MAE = 2.356（**次要评测集**），clean = 1.150（D 子集，非独立）
- **达标状态**：B/D 其余 7 档全部 <2mm；未达标 = **B 2–3（2.32）/ B 7–8（2.71）/
  D 2–3（2.71）**。B 7–8 是单一扁平病灶（5 帧）；2–3mm 是训练样本稀缺（90 帧）。
- 报告与 claim：`output/reports/report.md`、`output/claims/best-*.json`
  （每个 claim 引用 3 个 seed 的 eval id 并 pin benchmark_id；test_access 为 WARN，
  属预期，见 §2 的访问历史）
- γ=1.0 与 γ=2.0 统计上不可区分（见 D-0042）；γ=1.0 单 seed 为 B 1.130 / D 1.073。
- 所有历史 baseline / 旧策略 run 已删除；实验账本见 §4，决策见 DECISIONS。
- **当前目标口径（company + B，见 §1）下的达标状态**（2026-11-05，item 46–48）：
  - 主力仍是 `v11-company-base`（三 seed：company **1.125 [1.107–1.151]**、B **1.077 [1.062–1.090]**）；
    未达标档 = company **9–10**（三 seed 全超）、B **7–8**（三 seed 全超）；边缘档 company 2–3、B 2–3/9–10（1/3 seed 超）。
  - `v11-company-diranchor` 单 seed 把 company **8 个档全部压到 <2mm**（worst_bin 1.825、val 3.333 同步），
    代价是均值变差（company 1.199 / B 1.205）且 B 仍不达标 → **尚未采纳**，等第二个 seed。
  - 上面的「3 seed 官方 eval / B/D/clean」段落属于**旧的 single-v2 口径**，保留作历史，不再作为达标依据。
- **算法主体（2026-11-06 owner 采纳，D-0081）**：**beta-only**（`model.fusion: none`），s20/s21；company lt10 **1.146/1.090**（8 档全 <2）、B **1.114/1.093**（按 D-0083 剔除 7–8 单例后其余 7 档全 <2；剔除前该档 3.46/3.58）、R 1.767/1.778；d40/diranchor/gatefloor 保留为历史对照，不再作为主配方。
- **深度换源终选（D-0085/D-0088）**：Endo β-only company 1.109（同 seed 与 DA 差 −0.037/+0.019，噪声内），
  但 B lt10 1.642（比 DA 差 +0.59）、R 2.196（+0.42）→ **Endo 否决，保持 DA β-only**。
  Endo 的 B 7–8 单例反而降到 1.706，但 B 2–3（3.00）/9–10（2.16）崩，不能交换。
  Metric-Beta（保留米制 token）company **1.089**、B **1.722**、R 2.229 → B 未改善、反而更差，
  按预注册否决（D-0088）。Metric-Anchor（物理 k + 有界残差）company 1.110（9–10 **2.47**）、
  B **1.491**（2–3 2.76、9–10 2.28）→ B 小档改善但 company 大档崩，按预注册否决（D-0090）。
  **Endo 三次尝试全部否决**（D-0090）；2026-11-08 第四次尝试（D-0091 米制焦距头 v1）
  在 epoch 18 val lt10 2.86（对照 1.30），紧界与逐帧 k 0.39–2.41× 的跨度不匹配 →
  **v1 否决并停止**（D-0092）。v2 完成（D-0094）：company lt10 **1.062**（全项目最好）、
  B lt10 **1.172**（Endo 最好），但 company 2–3 **2.126** / 9–10 **2.593**、B 2–3 **2.013**
  超 2mm → **按 D-0093 否决**。v3（D-0095，仅 `c_app` 0.3→0.6）完成（D-0096）：
  company lt10 1.069，但 2–3 **2.184** / 9–10 **2.180** 仍 >2；B 报告档全 <2
  （最差 2–3 1.916）但 lt10 **1.200** >1.10 → **按 D-0095 否决，Endo 线关闭**。
  主配方保持 DA β-only。
- 关键修复：D-0018（内容裁剪现算）、D-0023/D-0041（eval 模式，**修复前所有
  eval.py 记录不可用**）、D-0026（EMA 优先选模）、D-0028（三件套 checkpoint）。

## 4. Experiments

### 4.1 Backlog / planned

| priority | hypothesis | change | success criterion | status | run |
|---|---|---|---|---|---|
| 1 | 40 轮 schedule 是否就是根因 | `ranksim-repro`：40ep cosine + patience10，其余同 80 轮 run | 对齐 rank2-official（B≈1.2/D≈1.13/R≈2.37） | queued | - |
| 2 | case_bin vs shuffle（同配方） | `baseline-shuffle`：仅 `data.sampler=shuffle` | 四 benchmark 对比 baseline-best | queued | - |
| 3 | proj 用 best recipe | `model.proj=true` + `mm_rank`（40ep） | 同上 | queued | - |
| 4 | backbone 训练范围 | `backbone=partial/frozen`（40ep；full 与 baseline 重复，已移除） | 同上 | queued | - |
| 5 | align vs rank2-official 差距是否 seed 噪声 | 同配方补 seed 20261021/20261022（+梯度范数日志） | **已验证**：B 散布 1.57–2.14，seed21 距 PM 仅 0.39/0.10/0.41；梯度范数两边重合——实现无差异，单 seed 结论不可靠 | completed | `20261031-101410`、`20261031-103839` |
| 6 | B/D 达标（新标准：两集全档 <2mm） | ①scale 增广 ②isotonic 标定 ③SWA——全部**证伪**（各自未压下目标档或仅挪位置） | 仅 B 8-9 过线；2-3 与 7-8 不动 | completed | `20261031-142117` 等 |
| 7 | 8-9 档超标（0.1-0.4）是否 seed 噪声 | baseline-shuffle seed 21/22 | 任一 seed 全档 <2mm → 达标可行 | running | 队列 20261031-1623 |
| 8 | 小档相对梯度缺失 | `loss.relative: true`（相对误差，小档按比例放大压力） | **B 受益**：2-3 2.93→2.67、7-8 3.12→2.44、8-9 过线、lt10 1.18（历史最好）；D 无效（2-3 n=8 信息瓶颈） | completed | `20261031-200602` |
| 9 | relloss 的 seed 复核 | relloss seed 21/22 | **已验证**：三 seed B worst 2.67–3.03、D worst 2.53–2.76——无 seed 过线，2–3 档系统性 | completed | `20261031-210713`、`20261031-220123` |
| 10 | 数据侧最后一张牌：旧 val（含 2-3/9-10 尾档）并入训练 | `val_from_train_cases: 50`（D-0020） | **D 2-3 突破**：2.28–2.64 → **1.85** ✅；D 仅剩 8-9（2.78）；B 剩 2-3（2.55）/7-8（3.45）/9-10（2.19）——单 seed | completed | `20261031-225639` |
| 11 | valmerge 三 seed 终判 | seed 20/21/22 全档对照 | D 2-3 改善（1.85/2.08/1.88，2/3 seed <2 但**不能写已解决**）；D 8-9 恶化（2.61–2.85）；B 2-3/7-8/9-10 仍不达标。**注意：valmerge 同时改了训练成员与选模集，且新 val 只覆盖 4-6mm——收益归因与终局判断均需 D-0022 的新协议** | completed | `20261031-225639`（s20）、`20261101-000109`（s21）、`20261101-004857`（s22） |
| 12 | REAL-colon 是否可入训补公开域小息肉 | 45 病灶/97,779 标注帧审计 + 拼图人工复核（1-2mm×8、3-5mm×8） | **结论：帧级质量不足**——最高分帧也多为「框在、病变难辨/贴边/器械遮挡」，1-2mm 档不可用，3-5mm 仅零星可辨 | completed（调查） | `output/reports/realcolon_pool_audit.csv` |
| 13 | **D-0022 预注册对照**：小档改善是否以大档为代价 | dev 池（train+val，2,249 帧/605 例）病例级三折；A=纯相对损失 vs B=混合损失（0.75·\|err\|+0.25·m·\|err\|/d，m=5.0 预固定）；6 次训练，seed 固定，不碰 test | **推进条件（预注册）**：B 相对 A，dev 7–10mm 最差档（帧平均）改善 ≥0.5mm 且 2–3mm 退化 ≤0.2mm；报告全部档位。满足才补 seed，不满足再看分支输出。**进行中** | running | `scripts/run_fold_compare.py`（6 run，D-0023 修复后重启） |
| 14 | **D-0023 框架 bug**：验证/评估从未进 eval 模式（BN train 模式 + running stats 被验证数据污染） | `Runner._validate` 加 eval()/恢复 + `policy.eval_context`；回归测试 | **已确认根因并量化**：align checkpoint BN 重估后 B 1.57→1.245/R 2.78→2.313（达 PM 水平）；eval.py 的 test 记录不受影响（新旧逐位一致）；pre-fix 各实验的 BN 重估修正数字见 `output/reports/bn_recal_per_bin.json`；**实验排序不变，valmerge 仍最优且无污染** | fixed | 影响所有 pre-fix DLEF run 的 checkpoint BN stats；PM 侧不受影响 |
| 15 | RankSim + 相对损失 + valmerge | `v11-ranksim-relloss`（40 轮跑满，修复口径） | **B 1.054 / D 1.261 / R 2.131**（B、R 历史最好）；小中档全面改善，7–9mm 退化 | completed | `20261102-093428` |
| 16 | 对数空间损失 + RankSim + valmerge（C2） | `v11-ranksim-log`，log(pred/truth) SmoothL1 β=0.1 | **大档救回**：D 8–9 2.90→1.32、B 8–9 2.97→0.87、B 9–10 3.38→1.06；但小档退化（B 2–3 1.95→2.37、D 2–3 2.51→2.89）。`last.pt`(ep40) 优于选中的 `best.pt`(ep7)：D 1.150 vs 1.221 —— 选模集只含 4–6mm 的实证危害 | completed | `20261102-104927` |
| 17 | 两模型预测平均（诊断，非方法） | relloss(best_ema) + log(last) 简单平均 | **D 仅剩 2–3（2.70）超标**，8–9=1.98 过线；B 剩 2–3（2.16）与 7–8（2.75）。两端失败已收敛到「2–3mm 档 + B 单一扁平病灶」 | completed | 见本条 run |
| 18 | **EMA 优先选模（D-0026）** | Runner 训练结束保存 `final_ema.pt`；`best_checkpoint_path` = final_ema → best_ema → best；chart 默认画 final_ema+best | `v11-ranksim-log` 的 ep40 EMA 实测 B 1.130 / D 1.073 / R 2.379（此前 best/best_ema 因 4–6mm 窄 val 停在 ep7，B 1.306/D 1.219）。剩余未达标：B 2–3（2.07）/ B 7–8（2.55）/ D 2–3（2.75） | completed | `20261102-104927` last.pt:ema |
| 19 | 尺寸显式输入（D-0027）是否解决 2–3mm | `v11-ranksim-log-size`：log+rank+valmerge + `size_input`（β 头加 log(S)/log(D) 零初始化旁路） | **证伪**：总 <10mm 基本持平（B 1.130→1.127、D 1.073→1.080），但 **B 2–3 反而 2.07→2.36**、D 2–3 2.75→2.82；`size_proj` 权重已学到非零（机制通，不是接线错）。结论：2–3mm 偏差不是「头看不见 S」造成的 | completed | `20261102-133448` final_ema ep40 |
| 20 | 相对+对数双项损失（rellog） | `v11-ranksim-rellog`：`relative` + `log_weight=1.0` + RankSim + valmerge | **无协同**：B 1.135（log 1.130 / relloss 1.054）、D 1.121（log 1.073）；逐档不如 log（D 8–9 1.76 vs 1.57、B 2–3 2.21 vs 2.07）。结论：两项相加只是稀释，取不到两端各自优势 | completed | `20261102-153854` final_ema ep40 |
| 21 | RankSim 消融（log 配方下 γ=0 vs 1） | `v11-log-norank`（纯 log，无排序项） | **rank 有用**：B 1.405→1.130（−0.275）、D 1.166→1.073（−0.093）、R 2.535→2.379（−0.156）。γ 值得调 | completed | `20261102-164838` final_ema ep40 |
| 22 | RankSim 权重调优 | γ=0.5 / 1.0 / 2.0 三档，其余同 ranksim-log | **γ 单调改善 B/R**：B 1.184→1.130→**1.066**、R 2.380→2.379→**2.293**；D 1.070→1.073→1.097（+0.024，噪声级）。**trade-off**：B 2–3 随 γ 变好（2.42→2.07→2.01），但 B 7–8 变差（2.56→2.55→2.83）。单 seed，需复核 | completed | `20261102-173514`(g05)、`20261102-181928`(g20) |
| 23 | R 测试集并入训练（γ=2.0 基座） | `v11-ranksim-log-g20-r`：train 2,076→2,261 帧（+R 185 帧） | **无净收益**：B 1.066→1.089（+0.023）、D 1.097→1.095（−0.002）；逐档只在 D 2–3（2.74→2.37，n=8）与 B 2–3（2.01→2.47）出现反向变化，量级均在小区间噪声内。R 自身降到 0.143 = 训练集内表现（已失效，不可报告） | completed | `20261102-192828` final_ema ep40 |
| 24 | 轻量增广（翻转/亮度对比度/噪声，D-0040） | `v11-ranksim-log-g20-aug`：γ=2.0 + light 增广 | **轻微负收益**：B 1.066→1.115、D 1.097→1.130、R 2.293→2.412；逐档只有 B 8–9 改善（1.50→1.02），B 2–3 变差（2.01→2.25）。结论：增广不采纳，保留 identity | completed | `20261102-204124` final_ema ep40 |
| 25 | γ=2.0 三 seed 复核 | seeds 20261020/21/22，同配置 | **稳定**：B 1.066/1.150/1.123（均值 **1.113**，散布 0.084）、D 1.097/1.135/1.102（**1.111**，0.038）、R 2.293/2.356/2.419（2.356）。逐档均值：B 2–3 **2.32**、B 7–8 **2.71**；D 2–3 **2.71**、D 8–9 1.72。**γ=2.0 与 γ=1.0（单 seed 1.130/1.073）差异在 seed 噪声内**；单 seed 的 1.066 属好种子 | completed | `20261102-182028`、`20261102-215308`、`20261102-223918` |
| 26 | **D-0041 修复：TorchModel 缺 eval()** | 补 train/eval/training passthrough + 回归测试；用修复后 eval.py 重测 3 seed | 官方 eval 与离线口径逐位一致（B 1.066/1.150/1.123）；此前所有 eval.py 记录为 train 模式，不可用于报告 | fixed | `DLEF/adapters/torch/components.py` |
| 27 | 分支独立监督（D-0045 预注册） | dev_folds=3/fold0：`v11-dev3-g20` vs `+0.25·L_β+0.25·L_direct` | **闸门未过**：dev 2–3mm 1.77→1.74（需 ≥0.5 改善）；1–2mm −0.59 但 n=1；其余档 ±0.13；测试 B 1.190→1.230 / D 1.142→1.111 / R 2.502→2.416（噪声内）。**分支路线关闭** | completed（负结果） | `20261103-111009`、`20261103-120259` |
| 28 | 小息肉病例级审计（D-0047） | dev fold0 + B/D 逐病例、按来源；k=truth/(D·S) 与纯几何先验检验 | **机制锁定**：dev 2–3mm 1.77 = company 2.20 + newdata 1.13；全部病例单向高估、帧内 std 0.12；测试小档 k≈0.031–0.034（全局 0.073 的 0.43–0.46×），训练小档 0.051；纯几何 k(D·S) 预测仍为 B 2.76 / D 3.13 → **该档在训练分布里不可学**，需标签/框审计 | completed | `case_audit_20261103-111009.csv` |
| 29 | 取景远近漂移审计（D-0049，审查第 1 步） | dev fold0（721 帧/202 例）逐帧 p/W、q、β/直接/门控；纳入 91 例（S 变化中位 2.18×）做 log-log 斜率 | **漂移成立且定位到 β 分支**：融合斜率中位 **0.20**（32% 病例 >0.3，近远差中位 **0.87mm**，44% >1mm）；**β 分支 0.91**（几乎纯跟随表观大小）、直接分支 0.02、门控 0.21；分解 dlog q/dlogS=**−0.25**（应 −1）、dlog β_eff=**+0.16**（应 0）→ 深度补偿只到 1/4。下一步：配对一致性训练 + 整图 7×7 空间特征（两个独立对照，预注册门槛见 D-0049） | completed（诊断） | `output/reports/framing_drift/` |
| 30 | 配对训练 A/B 定稿（D-0050，审查第 2 步） | 统一 a=p/W 诊断；新增 PairSampler（真实远近成对采样）与 `loss.pair_beta_weight`（β 分支配对一致性）；两臂 A=配对采样 / B=+0.5·L_pair,β，其余（帧、步数 96 批/epoch、seed、fold0）完全一致 | 实现完成、66 测试通过；**待 fold1/fold2 基线队列结束后启动**。预注册：漂移 ≤0.14（现 0.196）、2–3mm −0.5mm、其余档 +≤0.2mm；β 变稳但最终 mm 不改善则不接入 | ready（未启动） | `dev3-pair.yaml`、`dev3-pair-beta.yaml` |
| 31 | dev 三折 out-of-fold 汇总（D-0051，审查第 2 步） | fold0/1/2 同配方各训一次；每折用未训练该折的模型预测，合并 605 病例/2,249 帧，病例等权 | **2–3mm 跨折重复**：病例等权 **1.81mm**（32 例/90 帧；fold0 单折 1.77）；3–4 1.54、4–5 1.03、5–6 0.62、6–7 0.99、7–8 1.21、8–9 1.65；<10mm 合计 **1.185**；来源 company 1.137/newdata 1.248/phantom 1.548/virtual 1.851 | completed | `output/reports/dev3_folds/` |
| 32 | 配对一致性训练 A/B（D-0052，审查第 2 步） | A=`dev3-pair`（配对采样）；B=`+0.5·L_pair,β`；同帧/步数/seed/fold0，40 轮 | **漂移达标、精度不达标 → 不接入**：融合漂移 0.196→**0.121**（−38%）、β 0.898→**0.196**、βW +0.16→**−0.53**、无塌缩；但 B 集 lt10 1.190→**1.630**、dev0 1.300→1.630、B 2–3 2.55→3.48。主因：配对采样每轮只见 512 个不同帧（基线 2,044）→ 下一步 C 臂＝全帧 shuffle + 额外配对批 | completed（负结果） | `20261103-151549`、`20261103-160727` |
| 33 | 剔除 ≥10mm 训练帧（D-0053） | `dev3-lt10`：`data.train_max_truth_mm=10`，训练 1,297 帧（基线 1,528），val/test 不变 | **小档改善、大档回退 → 不直接采纳**：dev0 2–3 1.77→**1.43**、3–4 1.71→1.21；B 2–3 2.55→**1.97**、R lt10 2.50→**2.18**；但 7–8/8–9 三集一致变差，**worst_bin B 2.72→3.29、D 1.97→2.55**。下一步：阈值改 12/15 或对 ≥10mm 降权 | completed（混合） | `20261103-172712_5751b9c5_s20261020_30ef38` |
| 34 | 只剔除 >15mm 极端样本（D-0054） | `dev3-lt15`：训练 1,473 帧（只少 55 帧 15–26mm） | **无实质变化**：dev0 lt10 1.291（基线 1.300）、2–3 1.72（1.77）、7–8 1.09（1.20）；B 1.223 / D **1.126** / R **2.459**（基线 1.190/1.142/2.502）；漂移 0.187（0.196）。结论：**极端样本无害**，lt10 的代价来自去掉 10–15mm 锚点 → 关闭过滤路线 | completed（负结果） | `20261103-184249_9bbadcd5_s20261020_4f2fde` |
| 35 | 7×7 背景空间分支（D-0055，审查第 3 步） | `dev3-ctx`：ResNet layer4 保留 7×7 网格 + 位置编码 + 框关系 + 交叉注意力，零初始化注入 β 头 | **β 补偿改善、融合变差 → 不采纳**：β 斜率 0.898→**0.472**、βW +0.16→**−0.17**、小档 β 0.94→0.18；但门控 0.21→**0.71** 放大残余漂移，融合 0.196→**0.314**；测试 B/D/R lt10 全部退化（1.45/1.32/2.60）。唯一未试组合：ctx + β 一致性 + 全帧 shuffle/额外配对批 | completed（负结果） | `20261103-201244_62583ba8_s20261020_214580` |
| 36 | ctx + β 一致性 + 混合采样（D-0056，组合臂） | `dev3-ctx-pair`：7×7 背景分支 + 配对损失 0.5 + 16 配对批 + 全帧 shuffle（112 批/轮） | **不达标 → 路线关闭**：β 0.898→**0.450**、βW −0.224（目标 −0.75），但门控 0.21→**0.73** 放大残余漂移，融合 0.196→**0.303**；B/D/R lt10 = 1.581/1.274/2.845（全部退化）。结论：β 变好 ≠ 最终变好，门控是约束点 | completed（负结果） | `20261103-212457_0e1a0599_s20261020_2cf5ef` |
| 37 | 用户重划分 company-v1 + ctx 从零训练（D-0057） | 新数据：train 9,221/829、val 2,093/182、test 1,558/179；深度特征重建（DA-V2 metric-SfM）、每帧深度图写入 visioncheck；β_ref=0.016986；ctx 架构从零、15 轮 | **冻结 test（1,558 帧）**：帧级 <10mm **1.262**、worst_bin 2.17；**病例等权 <10mm 1.146**；2–3 1.86（9 例）、3–4 1.14、4–5 0.97、5–6 0.78、6–7 1.20、7–8 0.81、8–9 2.02（12 例）。独立数据版本，不与 B/D/R 可比；尚无同数据非 ctx 对照 | completed | `20261103-233352_78216d2e_s20261020_24fb66` |
| 38 | OSD 裁剪修复 + crop_box 固化（D-0058） | 饱和度裁剪替代 gray>12；清单逐帧存 crop_box；company-v1 特征重建 12,872 帧；DA/Endo 深度图后缀区分 | **修复两处数据缺陷**：新数据 6%、旧 single-v2 10% 帧曾把 OSD 带进输入；重建后 train 9,221/829、val 2,094/182、test 1,557/179，β_ref=0.016903，benchmark digest `0dae5700`；完整性核验全过；旧 company-ctx 成绩作废 | completed | `v11-company-v1` 重建 + `v11-company-base/20261104-103104` |
| 39 | 逐档图按数据版本选队列（D-0059） | `plot_experiment`/`plot_per_bin` 增加 company 队列；非 single-v2 默认只评自身 test | 修复 TB 图与官方数字不一致：旧脚本硬编码 single-v2，把 company-v1 模型评在**重叠的** D/clean 上（D 209/356 帧）。重生成后 final_ema <10mm **1.107**、best.pt 1.401，与官方 eval 一致 | completed | `output/charts/v11-company-base.png` |
| 40 | company-v1 base vs ctx（D-0060） | 同数据/种子/15 轮，唯一变量 `spatial_context` | **ctx 不采纳**：公司 test <10mm **1.107 vs 1.259**（病例等权 1.086 vs 1.156）、B 1.062 vs 1.295；只有 R 改善（1.959→1.523）。base 为当前最好：公司 1.107 / B 1.062 / R 1.959 | completed | `20261104-103104`（base）、`20261104-123141`（ctx） |
| 41 | benchmark2 + 两张补图（D-0061） | `input/benchmarks/benchmark2.json`：company(新)+B+R，D/clean 退出；非 single-v2 run 默认按 benchmark2 出图 | base final_ema：company **1.107** / B **1.062** / R 1.959；ctx：1.259 / 1.295 / **1.523**。两张图已写入 TB `chart/per_bin_mae` | completed | `output/charts/v11-company-{base,ctx}.png` |
| 42 | ctx 代码审查（D-0063） | 审 `_box_features`/`_context_vector`；修复对 S 的隐式依赖 | **修复**：框投影改用宽高比（与 span_calibration 解耦），新旧特征 max|Δ|=1.2e-7、掩码一致；**记录**：base/ctx 两 run 的 β_ref 差 0.016986 vs 0.016903（0.5%，不影响结论但非严格单变量） | completed | `core/models/v11.py` |
| 43 | 补充 10–15mm 微调（D-0064） | company-v1-big（train 11,114）；warm start 自 company-base，5 轮 lr 1e-4 | **大档小改善、中档退化 → 不采纳**：公司 test 帧级 1.107→**1.201**、病例等权 1.086→1.147；B 1.062→1.184、R 1.959→2.336；8–9 1.62→**1.50**、9–10 2.48→**2.38**，但 3–4/4–5/7–8 退化。顺带修 init_checkpoint 加载 bug 与图表命名 bug | completed（负结果） | `20261104-152948_e476958f_s20261020_ba6f53` |
| 44 | EndoStreamDepth 版 base（D-0065） | company-v1-endo（14,765 帧）+ b-r-endo（395）；非 ctx、≤10mm、15 轮、从零 | **与 DA 基本持平（略差）**：公司 test 帧级 1.137（DA 1.107）、病例等权 1.100（1.086）；B 1.111（1.062）、R 2.108（1.959）。结论：深度源不是瓶颈 | completed | `20261104-171224_f8044938_s20261020_d08721` |
| 45 | 小档标签/框约定审计（D-0047 第③步，D-0066） | `scripts/smallbin_audit.py`：base run `final_ema` 在 company_train/company_test/B/R 上逐帧导出 k=truth/(D·S)、门控、分支、框几何 | **误差主因是系统性压缩，不是标签噪声**：档内真值几乎是常数（整数 mm，std=0.00），预测均值只从 3.95（2–3 档）走到 6.54（9–10 档），逐档 MAE≈\|偏差\|（公司 2–3：1.97/+1.95；9–10：2.48/−2.46；R 1–2：3.26/+3.26）。**B 判据未触发**：R 2–3 的 k_ratio=1.96×、B 各档 2.2–8.4×，均**高于** company_train（不是 ≤0.6），β 分支几何标度跨队列本就不通用，B/R 门控几乎全走 direct（wg 0.10–0.19）。company_train **无 1–2mm 档**；R 小档 51 例偏差全为正（+0.5…+4.5），预测有 3–5mm 硬地板。噪声档（不作优化目标）：公司 9–10（59 帧/**4 例**）、B 7–8（5 帧/**1 例**）；真失败档：R 1–2（21/14）、R 2–3（34/18） | completed | `output/reports/smallbin_audit/`、`output/review/lt3mm_r/` |
| 46 | company-base 三 seed 噪声底 + 单调标定诊断（D-0066） | 过夜串行队列：seed 20261021/20261022 各训一次（`--tag d0066`）+ 官方 test eval → 三 seed 均值/全距图（benchmark2）→ 汇总报告 → `bias_recal_diagnosis.py`（只在 company-v1/val 拟合 isotonic 与 log-linear，val 侧病例级 2 折交叉拟合，test 仅套用） | **噪声底**（final_ema，`<10mm`）：company **1.125 [1.107–1.151] 全距 0.044**、B **1.077 [1.062–1.090] 全距 0.028**、R 1.930 [1.809–2.021] 全距 0.211。→ **ctx（Δ +0.152/+0.232）与 big（Δ +0.094/+0.122）的否决成立**（超出全距）；**endo 需修正**：company Δ+0.031 在噪声内、R Δ+0.149 在噪声内，但 **B Δ+0.048 > 0.028 → B 上 Endo 确实更差**。逐档「全 seed 超标」（真失败）：company **9–10**（2.482–2.898）、B **7–8**（2.894–3.388）、R 1–2/2–3/7–8/9–10；company 2–3（1.930–2.075）与 B 2–3/9–10 只有 1/3 seed 超标 → 边缘档。**标定被判否**：val 交叉拟合 lt10 1.356→log-linear 1.429（更差）、isotonic 1.891（灾难）；company test lt10 1.107→1.097（噪声级）、9–10 2.48→2.41、超标档 1→2；val 的 9–10 本来就是 3.97（比 test 更差）→ **压缩不是全局刻度错误，单调映射修不了** | completed | `20261104-103104`（s20261020）、`20261104-204533`（s20261021）、`20261104-224346`（s20261022）；eval `20261104-224301…a7839a`、`20261105-004556…97ecf9` |
| 47 | 解压缩臂 A：门控下界（D-0067） | `company-gatefloor.yaml`＝company-base + `model.gate_beta_floor: 0.5`（唯一变量，不新增参数）；seed 20261020、15 轮、从零 | **FAIL 且机制被否**：wg 确实被钉在 0.52–0.53，但 **β 分支的输出占比仍只有 0.10–0.17**（`pred_beta` 幅值远小于 direct）→ 加权改变不了数值；逐档偏差与 base 几乎重合（2–3 +1.99 vs base +1.95；9–10 **−2.57** vs −2.46，还略差）。company lt10 **1.124**（=base 均值）、worst_bin 2.598（噪声内）、B lt10 1.050、R 1.855。**结论：门控不是压缩的成因** | completed（负结果） | `20261105-005349_dc4296d5_s20261020_4dfb35`；eval `20261105-025150…fd2321` |
| 48 | 解压缩臂 B：direct 分支几何锚定（D-0067） | `company-diranchor.yaml`＝company-base + `model.direct_scale_anchor: true`（`pred_dir = exp(head_dir(·))·D·S`，bias 初值 `log(beta_ref)`）；其余同 item 47 | **判据 FAIL，但拿到了项目第一个 company 全档 <2mm**：门控**饱和到 wg=1.00 / β 占比 1.00** → 锚定后 direct 与 β 参数化重复，被彻底放弃，模型退化成 **纯 β·D·S**。压缩确实减轻：company 逐档偏差 +1.95/−2.46 → **+1.80/−1.68**，**worst_bin 2.482→1.825（8 个档全部 <2）**，且 val 侧同步改善（worst_bin 3.973→**3.333**，非 test 巧合）。代价：中档变差（3–4 +0.21、4–5 +0.20、7–8 +0.22 超出噪声底上限），company lt10 1.107→**1.199**（超噪声底 +0.048）、B lt10 1.062→**1.205**（超 +0.115，最大退化）。**解释**：B 的 k 是 company 的 2–8 倍，纯 β（依赖 D·S 尺度）失去跨队列逃生口 → B 的 2–3=2.373、7–8=3.381 仍超标（这两档 base 也超，且只有 3 例/1 例） | completed | `20261105-025342_36433ff6_s20261020_631bbc`；eval `20261105-045653…adb7a9` |


| 49 | 分档头（ordinal head，D-0069，预登记） | `company-ordinal.yaml`＝company-base + `model.ordinal_head: true` + `loss.name: mm_ordinal`（11 类 CE + 档内偏移 + RankSim γ=2，病例/档位等权）；**纯 mm 参数化**，D·S 只作特征不做硬锚定；seed 20261020、15 轮、从零 | 依据：`scripts/bin_accuracy.py` 显示 base 的**分档准确率只有 28.9%（company）/ 36.0%（B）**，而命中档时 MAE 只有 0.409/0.327 → 档宽 1mm、标签整数 mm，**分档正确即档内误差 <1mm**，判据本质是分类问题，全部 headroom 在分类上。判据先登记：company+B 全档 <2mm；lt10 相对噪声底上限（1.151/1.090）恶化 ≤+0.05；**bin_acc 必须显著高于 base**，否则=特征分辨力不足、头不是瓶颈；val worst_bin 需低于 base 的 3.973 | completed（负结果，D-0070）：company lt10 **1.874**（base 1.125）、worst_bin **3.908**；B lt10 **1.493**（base 1.077）；R 2.502；val ema worst_bin 4.455（base 3.973）。**bin_acc 反降**：company **18.1%** vs base 28.9%、B **32.0%** vs 36.0%；命中档 MAE 仅 0.135、分错档 2.259 → 头不是瓶颈，特征分辨力不足 | run `20261105-130723_e4c55d52_s20261020_bfeda1`；eval `20261105-152316_d02d0ccc_s20261020_63019b`；`output/reports/ordinal_20261105.md`、`output/reports/ordinal_bin_accuracy.md` |

| 50 | diranchor 跟进：第二 seed + 40 轮（D-0073，预登记） | ① `company-diranchor.yaml --set seed=20261021`（15 轮复核）；② `company-diranchor40.yaml`（`epochs=40`、`min_epochs=40` 保持「早停等于关闭」）；两者都带 `num_workers=4/pin_memory=true`（D-0071，不改样本顺序） | 依据 D-0072：diranchor 在 15 轮时 val 仍单调改善（worst_bin 3.408→3.333、lt10 1.539→1.407），base 自 ep7 起变差（3.595→3.973）→ 中档退化可能只是没收敛。判据先登记：① company 全档 <2mm 必须在第二 seed 重现，两 seed 全距＝该路线自己的噪声尺度；② val worst_bin 需继续低于 3.333、中档 Δ≤0、B lt10 ≤1.205；**首要读点仍是 bin_acc 能否抬过 base 的 28.9%/36.0%**（D-0070） | completed（D-0073/D-0074）：s21 company **1.152/1.95**（全档 <2）、B 1.176（7–8 3.19）；d40 company **1.119/1.744**（全档 <2）、B 1.125（7–8 3.51）；**val worst 3.621/3.607 仍 >3.333、bin_acc 26.8/26.2%（B 25.5/24.5%）仍低于 base** → company 判据成立但不采纳；B 7–8 与 bin_acc 是剩余阻塞 | s21 `20261105-155004_187149e3_s20261021_8a5d93`；d40 `20261105-164359_28105278_s20261020_8e76ad`；eval `20261105-183629_28105278_s20261020_5baa06`；`output/reports/diranchor_followup_20261105.md`、`diranchor_bin_accuracy.md` |

| 51 | 跨度归一化（S/W，D-0075） | `company-diranchor-span-norm.yaml` = diranchor + `model.span_reference_width=1258`；seed 20261021、15 轮、workers 4 | 预登记：val 9–10 改善 ≥0.3、其他档退化 ≤0.2；company/B 冻结基准各评一次 | **未达到**：val 9–10 3.621→3.847（+0.227）、2–3 +0.342、4–5 +0.694；company lt10 **1.108** 但 9–10 **2.347**（全档 <2 被打破）；B lt10 **1.127**、7–8 **2.983**（仍 >2）→ 不采纳 | run `20261105-202956_40f4313f_s20261021_9cdd40`；eval `20261105-211422_40f4313f_s20261021_8a8b2c-test-84bf65`、`20261105-211449_5b38edef_s20261021_491325-test-f893ad`；`output/reports/span_normalization_20261105/result.md` |

| 52 | 选模口径 + span 旁路 + β-only（D-0077/0078/0079） | ① d40-sel s21（新 monitor `ema/v11/robust_worst_bin`）；② spaninput s20/s21（保留 S + 零初始化 `[log(S/W), log W]`）；③ betaonly s20/s21（`fusion=none`）；40 轮、workers 4、checkpoint_every 5 | spaninput：S1 下降且 company 9–10<2、B 7–8 不低于 2.83，S2/S3 不恶化、S4 不低于 d40；β-only：S1/三队列 lt10 落在 d40 seed 噪声内 → 可简化，否则保留辅助头并记录隐式深监督；S1–S6 见 `output/reports/stability_protocol_20261105.md` | running（22:43 启动，预计 6–7h） | 队列 `scripts/run_overnight_20261105.py`；状态/日志 `output/logs/overnight_20261105*` |

| 53 | 步骤 5/6 追加（D-0080） | ① `v11-company-big-scratch`（company-v1-big 从零，diranchor40 recipe，40 轮）；② `v11-company-base448`（15 轮，batch4×accum4）；③ beta-only@448（`company-diranchor15-448.yaml` 已改 `fusion: none`；15 轮，batch4×accum4；experiment 名保留以兼容 append 队列）；448 用独立 `v11-company-test-v1-448` benchmark（id `sha256:e3cb0ad9…`） | big：去掉 D-0064 warm start 混淆；448：base@448 锚 + diranchor screen；screen 有信号才补 40 轮 | waiting（append 队列等前序 PID 3697898） | `scripts/run_overnight_append_20261105.py`；日志 `output/logs/overnight_append_20261105*` |

| 54 | Endo 米制焦距头 v1（D-0091，预注册，2026-11-08） | `company-betaonly-endo-focal.yaml`：metric token + `log f = log(1/beta_ref) + c·tanh(z/c)`，c=0.15；复用 `head_beta`、末层零初始化、无新参数；`metric_anchor` 互斥；40 轮、seed 20261020 | 替代 DA：company 全档 <2 **且** B（剔 7–8）全档 <2 **且** B lt10 ≤1.10；否决：任一 company 档 >2 或 B lt10 >1.20；单 seed 先跑 | **stopped（D-0092）**：epoch 18 val lt10 2.859 / robust_worst 4.925（对照 1.30/1.90）；逐帧 k p10–p90 0.39–2.41×，±16% 紧界不可能拟合 | run `20261108-152112_3f6cc575_s20261020_4deeab`（manifest running，中断）；日志 `output/logs/endo_focal_20261108.*` |

| 55 | Endo 米制焦距头 v2（D-0093，预注册，2026-11-08） | ① `f/W` 归一化等价式，`pred = beta_ref·D·S·exp(r_app − c_phys·tanh(z_phys/c_phys))`；② 几何宽界 `c_phys=2.0`，输入 10 维 `[logD, logS, logW, logH, log(S/W), log(S/H), 框宽高比, 深度均值/std/有效占比]`；③ 外观窄残差 `c_app=0.3`（复用 `head_beta`）；④ 零初始化严格等于 `beta_ref·D·S` | 同 D-0091：company 全档 <2 且 B（剔 7–8）全档 <2 且 B lt10 ≤1.10；否决：任一 company 档 >2 或 B lt10 >1.20 | **completed（D-0094 否决）**：company lt10 **1.062**（全项目最好）但 2–3 **2.126** / 9–10 **2.593**；B lt10 **1.172**（Endo 最好）但 2–3 **2.013**；R 2.001 | run `v11-company-betaonly-endo-focal2/20261108-164046_0a821430_s20261020_12a397`；eval `20261108-190943_0a821430_s20261020_9d1848` |

| 56 | Endo 焦距头 v3（D-0095，预注册，2026-11-08） | 唯一变量：`metric_focal_app_bound: 0.3 → 0.6`，其余完全同 v2/D-0093；思路：v2 的 company 尾档偏差 ±2.1–2.6mm，需给外观头更大的框/分割修正范围，几何头仍锚定队列尺度 | 同 D-0093：company 全档 <2 且 B（剔 7–8）全档 <2 且 B lt10 ≤1.10；仍不过则关闭 Endo | **completed（D-0096 否决；Endo 关闭）**：company lt10 1.069 但 2–3 **2.184** / 9–10 **2.180**；B 报告档全 <2（最差 2–3 1.916）但 lt10 **1.200** >1.10；R 1.940 | run `v11-company-betaonly-endo-focal3/20261108-194051_84c11f03_s20261020_a446cd`；eval `20261108-221118_84c11f03_s20261020_a33152` |

### 4.2 v2 runs

| experiment | config | status | <primary metric> | run | note |
|---|---|---|---|---|---|
| ranksim-best | `v11/ranksim-best.yaml` | completed | val worst 2.454；test 3.23–4.81 | `20261030-212317_09bdfb94_s20261020_17e52f` | 80 epoch；旧管线 |
| baseline-best | `v11/baseline-best.yaml` | completed | test lt10 B 1.37/D 1.30/R 2.96/clean 1.24 | `20261030-231251_24bb601b_s20261020_b91f9a` | 40ep case_bin；旧管线；修复后重跑中 |
| ranksim-repro | `v11/ranksim-repro.yaml` | completed | test lt10 B 2.25/D 1.47/R 3.40/clean 1.74 | `20261030-235746_f6bbc540_s20261020_f0e4a2` | 40ep；旧管线；证明管线差异（D-0018） |
| baseline-shuffle（旧管线） | `v11/baseline-shuffle.yaml` | failed | — | `20261031-004307*` | D-0018 前启动，已终止 |
| ranksim-align | `v11/ranksim-align.yaml` | running | - | 队列自动 | 40ep；**修复后管线**；评 best+best_ema |
| backbone-partial / frozen / proj | `v11/*-best.yaml` | queued | - | - | 40ep；修复后管线 |
| 旧 baseline / ranksim / proj / long80 | — | deleted | — | — | 旧训练策略全部作废，已删除 |

## 5. Open questions

- [ ] RankSim 在 DLEF 迁移后是否仍能复现 PolypMeasure 的单 seed 收益？
- [ ] 9–10 mm 训练帧不足，是否需要补开发侧病例？
- [x] 预测的**系统性压缩**（item 45）能否用 company-v1/val 拟合的单调映射消除？→ **否**（item 46：val 交叉拟合 lt10 1.356→log-linear 1.429、isotonic 1.891；company test 1.107→1.097 只是噪声级，超标档反而 1→2）。压缩不是全局刻度错误。
- [x] 标定只能修刻度、不能修分辨力（D-0068）→ 已试**换监督信号**：分档头（item 49，D-0069）→ **判否**（D-0070）：bin_acc 18.1%/32.0% 反低于 base 的 28.9%/36.0%，val worst_bin 4.455 > 3.973；命中档 MAE 仅 0.135 → 头/损失不是瓶颈，**特征分辨力不足**。几何标度 k 按队列归一仍未做，留作 B 的候选修法。
- [ ] R 的 1–2 mm 档在 company_train 里**一个样本都没有**；开发侧是否需要补小息肉病例，还是承认该档不可外推？
- [ ] 两臂结论（item 47/48）：**门控不是成因**（floor=0.5 把 wg 钉在 0.52，偏差曲线却与 base 重合）；**锚定让 wg 饱和到 1.00 → 退化成纯 β·D·S，压缩显著减轻（company 偏差 +1.95/−2.46 → +1.80/−1.68，worst_bin 2.48→1.83，首次全档 <2mm，且 val 同步 worst_bin 3.97→3.33）但中档与 B 变差**。所以压缩在 **β 分支内部**，不在融合权重。分档头已判否（item 49，D-0070）——`scripts/bin_accuracy.py` 的读点成立（base 的 bin_acc 只有 28.9%/36.0%，命中档 MAE 仅 0.41/0.33），但 ordinal 头把 bin_acc 压得更低（18.1%/32.0%）→ **下一步转特征/输入侧**；备选「直接回归 k = truth/(D·S)」同属输出侧，优先级下调。
- [ ] diranchor 是否补第二个 seed（20261021/22）以确认 company 全档达标 + 量化中档退化？目前只是单 seed，按规则不能采纳。
- [ ] B 是只剩的阻塞项且失败档极小（2–3=3 例、7–8=1 例）；纯 β 路线在 B 上退化最多（1.062→1.205，B 的 k 是 company 的 2–8 倍）→ 是否需要**按队列归一的几何标度**，或 diranchor + 仅在跨队列时保留 direct 逃生口？
- [ ] 第三臂：`company-v1-big`（真值上限 15 mm）**从零**重训（去掉 D-0064 的 warm start 混淆）能否解决「company-v1 train 上限 10.00 mm、9–10 档坐在标签边缘」？改数据版本+benchmark，需 owner 确认后再跑。
- [ ] B 的 7–8 档只有 `polyp-size-04` 一例（S/W=0.184，而 B 自己的 8–9=0.414、9–10=0.448，队列内部就不自洽），且 Polyp-Size 的 42 段视频**就是** B、无法补数据 → 该档是否应承认为不可达，而不是继续优化？
- [x] span / diranchor 是否欠训练？→ **否**：span val worst 在 **ep4 达 3.598** 后单调升到 **3.847**（ep15），同种子控制 s21 在 ep5 达 3.405 后升到 3.621，40 轮 diranchor 在 **ep10 达 3.414** 后升到 3.607（只有 lt10 仍缓慢改善 1.42→1.32）；同期 train loss 持续下降到 ~0.7–0.9 → 是泛化/过拟合与域偏移，不是优化没跑够。给 span 再加轮数预计不会改善 worst_bin。
- [x] span 在 R 上（D-0075 补充）：lt10 **1.750**（同种子对照 s21 1.964、base 三 seed 1.930）、worst **3.113**（7–8；对照最差 1–2 3.101）；1–2/2–3/4–6 改善、7–8/8–9/9–10 变差。R 不要求，只作观察（eval `20261105-214446_e267761a_s20261021_5c0998`）。

## 6. Next actions

- [x] 删除所有旧训练策略的 run，只保留当前 best
- [x] `v11-ranksim-best` 跑完 80 epoch 并评四个 benchmark
- [x] baseline-best（40ep case_bin）+ ranksim-repro（40ep）完成并评估——锁定管线差异（D-0018）
- [x] 小档标签/框约定审计（D-0047 第③步）——见 item 45，结论指向系统性压缩
- [x] **读 `output/reports/overnight_20261104.md`**：第 1–5 节（D-0066）与第 6 节（D-0067 两臂）结果已落档，见 item 46–48 与 D-0068
- [x] 三 seed 全距：company 0.044 / B 0.028 / R 0.211（**≤0.2**）→ ctx/big 的否决成立；**endo 需修正**（B 上 Δ+0.048 超出 0.028 → B 确实更差；company/R 在噪声内）
- [x] diranchor 跟进完成（item 50，D-0073/D-0074）：s21 company 全档 <2mm 复现；d40 company 1.119/1.744（全档 <2，当前最好）；但 **val worst 3.62/3.61 > 3.333、bin_acc 仍低于 base** → 判据未全过，不采纳；B 7–8（3.19–3.51）仍是唯一阻塞
- [x] span-normalization 完成（D-0075）：预登记 val 判据未达到（9–10 +0.227、2–3 +0.342、4–5 +0.694）；company lt10 **1.108** 但 9–10 **2.347**（全档 <2 被打破）、B lt10 **1.127** / 7–8 **2.983**（仍 >2）→ 不采纳；B 7–8 仍是唯一 blocker
- [x] item 52 队列完成（D-0079，22:43 启动）：d40-sel s21 → spaninput s20/s21 → betaonly s20/s21；40 轮 + 官方 eval + benchmark2 图 + 小档审计；结果见 `output/reports/stability_protocol_20261105.md` 与 `stability_s3_s6_20261105.md`
- [x] 步骤 0：S1–S6 协议 + R 新口径复算 + S3/S6/bin_acc 现有 run 表：`stability_protocol_20261105.md`、`benchmark2_r_recheck_20261105.md`、`stability_s3_s6_20261105.md`、`stability_bin_accuracy_20261105.md`、`stability_existing_20261105.md`
- [x] 步骤 1：判据病例盲审包已生成（16 例=8 目标+8 对照，匿名联系表/worksheet/key）：`output/review/judgment_cases_20261105/`；人工独立测量尚未进行
- [x] **采纳 beta-only**（D-0081，owner 2026-11-06）：`fusion: none` 作为算法主体；s20/s21 已支持
- [x] B 7–8 单例处置（D-0083，owner 2026-11-06）：检测框与存储框一致、S/W 队列内不自洽 → 按离群从 benchmark2 报告剔除；**冻结 `v11-b-test-v2` 不变**。新口径 β-only s20/s21：company 1.146/1.090（全档 <2）、B lt10 **1.054/1.029**（报告档全 <2）——见 `output/reports/benchmark2_b_no78_20261106.md`
- [x] item 53 完成（D-0080/D-0082）：big-scratch、base448、beta-only@448 均完成 → 448 未采纳（company 1.206 / B lt10 1.150 / S1 3.319，均差于 224 同种子）；条件臂保持 held，见 `output/reports/conditional_448_20261106.md`
- [x] 单一视图 image-only（D-0084）：15 轮 company lt10 1.547 / B lt10 1.671（新口径），明显差于双视图 β-only → 深度/几何/β 分支不可省
- [x] Endo β-only 完成（D-0085，tag `d0085`）：company **1.109**（与 DA 同噪声）、B lt10 **1.642**（Δ +0.59，FAIL）、R 2.196；S1=B 7–8 **1.706**（通过）但 B 2–3 3.00 / 9–10 2.16 崩 → **不采纳 Endo，主配方保持 DA β-only**；报告 `output/reports/endo_betaonly_20261106.md`
- [x] Endo 米制刻度诊断（只读，2026-11-06）：B/company 隐含 f 比值 **1.084 [0.944, 1.224]**、q_mm 21.4 vs 21.8 → **B 不是常数因子偏移**；R 才是 2.49× 偏移。DA 的 B/company 反而只有 0.31×。B 的 Endo 误差是 2–5mm 高估、8–10mm 低估的尺寸相关压缩，不能靠队列级常数标定修（且 B 只有 test，标定=test tuning）→ 报告 `output/reports/endo_scale_20261106/findings.md`
- [x] Endo token/β 输入 OOD 诊断（只读，2026-11-06）：Endo B 的深度结构反而最接近 company（`mad_rel` 0.056 vs company 0.066；DA B 0.042）；β 输入线性探针 AUC Endo 0.929、DA 0.999，但 B 上逐帧 OOD–误差 Spearman 均为 **−0.08**、四分位 MAE 持平 → **误差不是 OOD 帧驱动，是队列级 k 校准压缩**（Endo B 2–3 k_pred/k_true 2.01、9–10 0.80）；物理项混合 α=0.25 可把 B lt10 1.644→1.153，但 company 1.109→1.814 → 不能全局用 → 报告 `output/reports/endo_token_ood_20261106/findings.md`
- [x] Metric-Beta 完成（D-0088，2026-11-06 22:40）：company lt10 **1.089**（case 1.063）、B lt10 **1.722**、R 2.229；B 2–3 3.13 → 预注册 B>1.30 触发**否决**，且比 Endo 相对 token（1.642）更差。保留米制 token 只把 company 拉到 DA 同级，没修好 B 的 k 压缩 → 流程上 Endo 已关闭；owner 仍要求再试一次 head 层约束（D-0089）。
- [x] Metric-Anchor 完成（D-0090，2026-11-07 01:13）：metric token + `k_phys(log D, log S)`（±1.0）· 有界残差（±0.3）→ company lt10 1.110 但 **9–10 2.47 >2**；B lt10 **1.491**（2–3 2.76、9–10 2.28 仍 >2）→ **B>1.20 且 company 有档 >2，预注册否决**。锚定把 B 2–5 拉回、却压死 company 5–10 → company 校准 vs B 稳健的硬 tradeoff；**Endo 三次尝试全部否决，Endo 线彻底关闭，主配方保持 DA β-only**。run `v11-company-betaonly-endo-anchor/20261106-232535_8bb60ae1_s20261020_d5e6d1`；eval `20261107-011359_8bb60ae1_s20261020_119b5e`
- [ ] 后续（待 owner）：人工盲审 `output/review/judgment_cases_20261105/`；可选 448 40 轮；把 PROJECT.yaml 的 success 口径同步为 benchmark2 company+B（B 7–8 按 D-0083 不报告）
- [x] benchmark2 全量可视复核包（2026-11-08）：`output/review/visioncheck/`，β-only s21 `final_ema.pt` 推理 1,926 帧（company 1,557 / B 205 / R 164），按尺寸档 + 单帧 MAE<2/≥2 分目录；三队列 lt10 与官方 eval 逐位一致（company 1.0903 / B 1.0288 / R 1.7780）；`summary.md` + `predictions.csv` 同目录
- [x] 下一臂已跑完：**分档头 `v11-company-ordinal`**（item 49，D-0069/D-0070）→ **FAIL**；备选「回归目标换成 `k = truth/(D·S)`」押后
- [x] 分档头判定完成（D-0070）：bin_acc 未升反降（company 18.1% vs 28.9%、B 32.0% vs 36.0%），val worst_bin 4.455 > 3.973 → **头不是瓶颈，特征分辨力不足**
- [ ] 特征侧下一臂（owner 2026-11-05 押后，先跑 item 50）：**靶子已量化**——紧视图是 1.25× 框裁剪再 letterbox 到 224，病灶恒占 ~179px，与真实尺寸无关；按 stride 32 折算 **2–3mm 档 ≈2.2 特征格/mm，9–10mm 档只有 ≈0.59 格/mm**（整图视图更糟：284px 框 → 1–2 格），即**失败的顶档恰好 mm 分辨力最低**。候选：`input_size 224→448`（模型不用改，avgpool 已是 Identity+空间均值）、或「常量像素尺度」高分辨率紧视图（不按框比例缩放，保留绝对尺寸）。注意：改分辨率会改 `DataSpec.schema` → **benchmark_id 变**，必须同时重训 base@448 当锚（禁止跨 benchmark_id 比数字）
- [x] DataLoader 并行化：新增 `data.num_workers/pin_memory`，company-base/ordinal 已设 4/true；实测 **726→190 ms/step（3.8×）**，15 轮约 2h15→~35min
- [ ] 若还要更快：半分辨率解码（`cv2.IMREAD_REDUCED_COLOR_2`，需同步缩放 crop_box/bbox 与 S）——会改像素重采样，先小样本验证再决定
- [ ] owner 确认后把 `PROJECT.yaml:9` 的 success 改成 benchmark2 的 company+B 口径（见 §1）
- [ ] R 小档盲审：`output/review/lt3mm_r/`（96 帧 / 51 例联系表）人工核对框与标签是否可信
- [ ] ranksim-align（修复后管线）对齐 rank2-official（B≈1.2/D≈1.13/R≈2.37）
- [ ] 队列跑完 baseline×2（sampler 干净对照）+ backbone×2/proj
- [ ] 提交 DLEF 与项目的全部修复（两边都是 dirty）
- [ ] 为 best 结果创建 claim，并生成 report
- [ ] review `make gc-dry`

## 7. Known issues / gotchas
- **`v11-company-betaonly-endo-focal/20261108-152112_3f6cc575_s20261020_4deeab` 是中断 run**：
  2026-11-08 16:28 手工 SIGTERM 停止（epoch 20），manifest 仍为 `running`，**不是结果**；
  v1 否决依据见 D-0092（epoch 18 val lt10 2.86）。不要用 `ls -dt` 把它当成正式 run。
- **`v11-company-ordinal/20261105-103728_e4c55d52_s20261020_4510de` 是中断 run**：11:37 被外部 SIGKILL（epoch 7/15），manifest 仍为 `running`，只有 epoch-1 `best.pt`，**不是结果**；正式结果是 `20261105-130723_e4c55d52_s20261020_bfeda1`（15 轮 completed）。
- **`smallbin_audit.py` 对无 gate/pred_beta 的模型（ordinal）会崩**（`np.isfinite(None)`）：已修 `med()` 与 `beta_share_med`；重跑后 `smallbin_audit_ordinal/summary.{md,json}` 正常，`k/train=None` 属预期（未同时评 company_train）。
- **DataLoader 并行化（2026-11-05）**：`data.num_workers=4`、`pin_memory=true` 已写入 company-base/ordinal；shuffle 仍由主进程 generator 决定，样本内容/顺序不变。旧 run 的吞吐（~0.75 s/step）来自 `num_workers=0` 单线程解码 1080p/4K JPEG + npz，不是 GPU 限制。
- **benchmark2 的 R 队列自 2026-11-05 起排除 1–2mm**：suite 成员带 `min_truth_mm: 2.0`，`plot_experiment.measure()` 过滤 manifest 行；冻结 `v11-r-test-v2` 与官方 `eval.py` 不变（仍含全帧）。历史图表/`_evalmode.json` 未重写（仍含 1–2）——见 D-0076。
- **benchmark2 的 B 队列自 2026-11-06 起排除 7–8mm**：suite 成员带 `exclude_bins: [7]`，`filter_rows()` 按 `int(truth_mm)` 剔除（8.0 归 8–9）；company 与 R 不变。这是**报告口径**，冻结 `v11-b-test-v2` 与官方 `eval.py` 仍逐帧含该档。历史图表/`_evalmode.json` 未重写——见 D-0083。
- **`v11-company-diranchor15-448` 现在是 beta-only@448 的别名**（D-0082）：配置里 `model.fusion: none`，experiment 名保留只是为了 append 队列的 `find_run` 能自动接 eval/chart/audit；看 run 的 `config.resolved.json` 里的 `fusion` 才是真实结构。
- **2026-11-05 23:40 上线前复盘**：① 新 `v11` 指标在 d40 官方 test eval 上逐位复现旧值（lt10 1.1185、worst 1.7436；新增 `case_mae_lt10`/`robust_worst_bin`）；② 448 benchmark 独立 id `sha256:e3cb0ad9…`，base448/diranchor15-448 verify 通过，**448 batch4×accum4 smoke 通过**；③ big eval 的 `data.version=company-v1` 覆盖 verify 通过（冻结 benchmark `sha256:d3ddb232…`）；④ 全套 103 测试通过；⑤ 运行中的 d40-sel s21 的 resolved config **没有 `input_size`**（启动早于该字段），默认 224，旧 resolved 配置的 eval 已验证可用。
- **Makefile 硬编码 `python`**，本机 PATH 里没有 `python`（只有 venv 里的解释器）→
  `make status` / `make check` / `make test` 直接 exit 127。目前只能显式用 venv python 调
  `core/tools/*` 与 `scripts/*`。待修：给 Makefile 加 `PY ?=` 变量。

- **`company-base` 的早停等于关闭**：`early_stopping.min_epochs == epochs == 15`，永远不停；
  且 `best.pt`（按 smoothed `worst_bin` 选）比 `final_ema.pt` 明显更差（公司 test 1.401 vs
  **1.107**）→ 报告一律用 `final_ema.pt`，不要拿 `best.pt` 当"最好"。

- **`v11-company-base` 下有三个 run，两个 `status: failed`**：`ls -dt ... | head -1` 会取到
  最旧/失败的目录。脚本必须校验 `manifest.json` 的 `status == completed`（过夜队列的
  `find_run()` 已这么做）；两臂脚本还额外排除 `"debug": "true"`，因为 CPU 冒烟 run 与正式
  run **同 seed**（20261020）、同 experiment 目录。

- **`model.gate_beta_floor` / `model.direct_scale_anchor`（D-0067）默认关闭且不新增参数**：
  关闭时前向与旧模型**位级一致**、`state_dict()` 键完全相同（`tests/test_decompression_arms.py`），
  所以旧 checkpoint 和在跑的 run 不受影响；两臂只支持 `model.name=v11`，`gate_beta_floor`
  必须在 [0,1)。改这两个开关会改变初值（`direct_scale_anchor` 把 `head_dir` 的 bias 从
  `log(expm1(truth_median))` 换成 `log(beta_ref)`），因此**不能**与旧 run 混为同一噪声底。

- **链式队列的等待条件**：`run_overnight_arms_20261105.sh` 等 tail pid 3424030 退出，再用
  `pgrep -f "core/commands/(train|eval).py .*company-"` 兜底。不要在队列期间手工跑 company-*
  的训练/评测，否则兜底循环会一直等。

- **light 增广实验不等价于「增广无效」**（D-0040）：`light` 只翻转 RGB
  （`torch.flip(dims=[-1])`），深度图与横向位置标量未同步变换，输入对应关系不一致。
  该实验只能说明「这个实现没有收益」，增广这条路仍待验证（需同步翻转深度与 x 类标量）。

- GPU 上同时跑 PolypMeasure 训练时会 OOM；迁移期先 CPU 冒烟
- 数据集移动过两次；引用路径统一走 `$DLEF_DATA_ROOT` 的相对路径
- `mae_ge10` 在没有 ≥10 样本时是 NaN；空区间也显式写 NaN（无法评价）
- 旧训练策略 run 已全部删除；当前只有 `v11-ranksim-best` 一个 run
- best 的 EMA / case-bin sampler / warmup 代码目前仍是未提交改动，run 的 `code_version` 会带 `dirty`
- RankSim/proj 必须保持 `init_checkpoint=""`；旧的非 ImageNet 初始化 run 已删除
- test access 的历史记录仍在；report 会标 `WARN`，不要重复评 test 来“刷新”

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
