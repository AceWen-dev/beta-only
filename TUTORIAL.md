# Beta-regression 项目 runbook

这份文件是 `beta-regression` 项目的操作手册，不是 DLEF 模板教程。

```text
安装 → 检查数据/benchmark → 训练 baseline → val 选模
→ 四个冻结 benchmark 做 test eval → variant/sweep → claim → report → gc
```

项目规则以 [`AGENTS.md`](AGENTS.md) 为准；稳定的研究问题在
[`PROJECT.yaml`](PROJECT.yaml)，当前快照在 [`STATE.md`](STATE.md)，历史决策在
[`DECISIONS.md`](DECISIONS.md)。

---

## 1. 安装

```bash
# DLEF core（只安装一次）
pip install -e /path/to/DLEF

# 本项目
cd /path/to/Betaregression_ep
pip install -e .

export DLEF_DATA_ROOT=/path/to/Betaregression_data
make test
```

数据目录至少要包含：

```text
$DLEF_DATA_ROOT/
  curated/v11_frames/single-v2/manifest.json
  curated/v11_frames/single-v2/...       # 图像 + depth feature
```

本机数据根目录是 `Betaregression_data/`，里面有 `pm/` 和 `images_new/` 的
符号链接。数据字节不提交到 git。

`make test` 只验证代码；需要 torch 和数据根目录的 v11 测试在缺环境时会 skip。

---

## 2. 看当前状态

```bash
make status
make runs
```

`make status` 显示 phase、run 数量、下一步和 git 状态。
`make runs` 是实验级摘要；看单个 run 用：

```bash
make inspect RUN=<run_id>
```

当前项目默认配置是：

```text
input/configs/experiments/v11/baseline.yaml
```

`input/configs/base.yaml` 只是 synthetic smoke，不是项目 baseline。
`make train` / `make check` / `make audit` 已经默认使用 v11 baseline。

---

## 3. 数据和冻结 benchmark

数据口径：

- 数据集：`v11_frames` / `single-v2`
- 样本按 `case_id` 分组，帧不跨 train/val/test
- 标签：单张图的镜下息肉尺寸，单位 mm
- 四个测试集：B / D / R / 清洗公司

冻结 manifest 在：

```text
input/benchmarks/
  v11-b-test-v2.json
  v11-d-test-v2.json
  v11-r-test-v2.json
  v11-clean-test-v2.json
```

benchmark 包含：

```text
test membership + label version + eval transform
+ metric protocol + aggregation + thresholds
```

任意一项改变，必须新建 benchmark，不能改旧 JSON。

检查：

```bash
make audit
make check
```

`make audit` 检查 split 大小、NaN/Inf、重复 sample_id 和 group overlap。
`make check` 检查 `PROJECT.yaml`、config、benchmark、`STATE.md` phase 和测试。

---

## 4. 训练 baseline

```bash
make train
# 等价于：
python core/commands/train.py \
  --config input/configs/experiments/v11/baseline.yaml \
  --tag baseline
```

跑完后看：

```bash
make runs
make inspect RUN=<run_id>
```

重点检查：

- `manifest.json` 的 `status`、`code_version`、`seed`、`tags`
- `config.resolved.json` 是否符合预期
- `metrics.jsonl` / `logs/train.log` 的 val `v11/worst_bin`
- `code_version` 不要是 `no-git`

如果 `code_version` 是 `no-git`，这个 run 只能算 exploratory，不能作为最终
baseline。先提交项目代码，再重跑。

---

## 5. val 选模

```bash
python core/commands/eval.py \
  --checkpoint output/runs/beta-regression/v11-baseline/<run_id>/checkpoints/best.pt \
  --split val
```

val 只用于模型选择，不能用来调 test。启用早停时用 `best.pt`；`last.pt` 是
最后一个 epoch，只用于 resume。`train.py` 的最终 val eval 会自动加载
`best.pt`。

---

## 6. 四个冻结 benchmark 的 test eval

test 必须显式给 benchmark：

```bash
python core/commands/eval.py \
  --checkpoint output/runs/beta-regression/v11-baseline/<run_id>/checkpoints/best.pt \
  --split test \
  --benchmark input/benchmarks/v11-d-test-v2.json
```

其余三个换 `--benchmark`，并加对应的 `--set data.test_set=...`；`eval.py`
不会从 benchmark 文件推断 cohort：

```text
B     --benchmark input/benchmarks/v11-b-test-v2.json     --set data.test_set=b
R     --benchmark input/benchmarks/v11-r-test-v2.json     --set data.test_set=r
clean --benchmark input/benchmarks/v11-clean-test-v2.json --set data.test_set=clean
```

如果当前数据和 benchmark 的 `benchmark_id` 不一致，`eval.py` 会拒绝运行。
这是保护机制，不要绕过。

每次 test 访问都会记录在 `output/test_access/`；同一个 `test_set_id` 多次访问
会在 report 里标 `WARN`。

---

## 7. 训练 variant / sweep

已有配置：

```text
input/configs/experiments/v11/baseline.yaml
input/configs/experiments/v11/ranksim.yaml
input/configs/experiments/v11/proj.yaml
input/configs/experiments/v11/backbone-full.yaml
input/configs/experiments/v11/backbone-partial.yaml
input/configs/experiments/v11/backbone-frozen.yaml
```

单独跑一个 variant：

```bash
make train CONFIG=input/configs/experiments/v11/ranksim.yaml
make check CONFIG=input/configs/experiments/v11/ranksim.yaml
```

新建 variant：

```bash
make new-experiment NAME=lr-0.0001 EXPERIMENT=v11
```

不要为每个 seed 新建 YAML。seed 是 run 维度，放在 sweep 参数里：

```python
params={
    "optim.lr": Choice((0.0002, 0.0001)),
    "seed": Choice((1, 2, 3)),
}
```

真实 GPU 实验用 `SubprocessSweep`；一次 trial = 一个 run。

---

## 8. 看 run 和实验摘要

```bash
make runs
make runs-all
make configs
make inspect RUN=<run_id>
```

每个 run 目录：

```text
manifest.json
config.resolved.json
metrics.jsonl
logs/train.log
checkpoints/last.pt    # 最后 epoch（resume 用）
checkpoints/best.pt    # 早停最佳（启用时）
evals/<eval_id>.json
```

RunStore 是唯一事实源；TensorBoard 只是镜像。

TensorBoard：

```bash
make tensorboard
# 等价于
python -m tensorboard.main --logdir output/runs --port 6006
```

打开 `http://localhost:6006`。远程机器可以：

```bash
# 服务器只监听 localhost，本地建隧道：
ssh -L 6006:localhost:6006 user@host

# 或让 TensorBoard 对外监听（注意网络权限）：
make tensorboard TENSORBOARD_ARGS=--bind_all
```

会看到：

- `train/loss` 和命名 loss terms；
- `val/v11/mae`、`val/v11/worst_bin`、`val/v11/acc10` 以及逐区间指标；
- Text 页签里的 `run/config`、`run/manifest`。

TensorBoard 只用于看曲线；最终数字以 `manifest.json`、`evals/*.json`、
`metrics.jsonl` 和 `logs/train.log` 为准。

---

## 9. Claim / report

报告数字不能手抄。先创建引用 eval id 的 claim：

```python
from dlef.claim import Claim, ClaimStore

ClaimStore("output/claims").add(
    Claim(
        claim_id="main-result",
        text="v11 baseline on v11-d-test-v2",
        eval_ids=("<eval-id>",),
        metric="v11/worst_bin",
        benchmark_id="sha256:6f6c3a3965f9cc4c9d260ed3bfd311e8bb23df9619aeab728b779afdb265e818",
    )
)
```

生成报告：

```bash
make report
# 或
python -m dlef.cli report \
  --runs output/runs \
  --claims output/claims \
  --out output/reports/report.md
```

---

## 10. 清理

先 dry-run：

```bash
make gc-dry
```

确认 plan 后加 `--apply`。`baseline` / `reported` tag 会保护 run。
不要手工 `rm -rf output/runs/...`。

---

## 11. 交给 coding agent

交接前确认：

```bash
make status
make handoff
make check
```

交接内容：

- `PROJECT.yaml`、`STATE.md`、`DECISIONS.md` 已更新；
- 当前 run/eval id 和指标有证据；
- 代码已 git commit，不处于 `no-git`；
- failed run 保留，没有手工删除；
- 四个 benchmark 的 `benchmark_id` 没有被改过。

---

## 12. 常见错误

- `make train` 以前会跑 `base.yaml`：现在默认已改为 v11 baseline；
  如需 synthetic smoke，显式传 `CONFIG=input/configs/base.yaml`。
- `make check` 失败：先看 `PROJECT.yaml`、config、benchmark 和 `STATE.phase`
  是否一致。
- 新机器缺 `cv2` / `numpy` / `torchvision`：确认执行了 `pip install -e .`。
- `code_version=no-git`：先 commit，再重跑；不要把它作为正式 baseline。
- test 访问次数 > 1：report 会标 `WARN`，不要为了“刷新”重复访问。
- `mae_ge10 = NaN`：当前 split 没有 ≥10 mm 样本时是正常现象，报告里要说明。

---

## 13. 完成标准

1. config 严格校验，selector 都能解析。
2. `DataSpec` 存在，split 已 materialize，batch 已验证。
3. model/loss/metric 契约实现，loss terms 有名字。
4. `make audit` 已检查，split/sample/group 无异常。
5. smoke + benchmark identity + `make check` 通过。
6. train 生成 manifest、resolved config、metrics、checkpoint。
7. 四个 test eval 都绑定冻结 benchmark 和 checkpoint digest。
8. 需要对比时，sweep 每个 trial 一个 run。
9. claim → eval → checkpoint → run 可追溯。
10. `make gc-dry` 只显示预期删除项。
11. `STATE.md` 和 `DECISIONS.md` 已更新，代码已提交。
