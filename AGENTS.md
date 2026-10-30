# AGENTS.md — beta-regression project rules

This is an **existing DLEF project**. The `dlef` core is installed as a package;
it is not vendored here. Read this file before editing.

- Stable brief: [`PROJECT.yaml`](PROJECT.yaml)
- Current snapshot: [`STATE.md`](STATE.md)
- Accepted decisions: [`DECISIONS.md`](DECISIONS.md) — do not re-litigate
- Operational runbook: [`TUTORIAL.md`](TUTORIAL.md)

---

## 0. Read before planning

### 0.1 Project memory

- `PROJECT.yaml` is the stable brief: question, unit, data contract, splits,
  benchmark, primary metric and success criterion.
- `STATE.md` is the current snapshot. Read it before planning and update it at
  the end of every session. Replace stale values; do not turn it into a log.
  Keep its YAML front matter (`project`, `phase`, `updated`, `primary_metric`,
  `best_run`, `benchmark`) valid; `make check` parses it.
- `DECISIONS.md` is append-only. Never rewrite or delete an old entry; add a new
  one and mark the old one `superseded by D-000X`. IDs are never reused.
- If `STATE.md` and the RunStore disagree, the RunStore wins; fix STATE.

### 0.2 Reading order

1. `AGENTS.md` — this file.
2. `STATE.md` — what exists, what is frozen, what is next.
3. `DECISIONS.md` — accepted/superseded choices.
4. `PROJECT.yaml` — stable project brief.
5. `README.md` and `TUTORIAL.md` — overview and operational runbook.
6. `core/config.py` — typed schema and selectors.
7. `core/task.py` — composition root.
8. `input/configs/experiments/v11/baseline.yaml` and `input/benchmarks/`.
9. Open only the component files the task actually changes.

**First response = plan, not code.** For non-trivial changes, state the task, the
comparison, data/split assumptions, the frozen benchmark, files to change, and
the 1–2 step smoke test. Do not write code until the plan is confirmed.

### 0.3 Stable facts

- Project id: `beta-regression`.
- Data: `v11_frames`, version `single-v2`, under `$DLEF_DATA_ROOT`.
- Membership: `curated/v11_frames/single-v2/manifest.json`; train/val/test are
  case-grouped by `case_id`.
- Primary metric: `v11`; `v11/worst_bin` is the worst non-empty 0–10 mm bin MAE.
  `v11/mae_lt10` and `v11/acc10` are reported alongside.
- Frozen tests: `v11-b-test-v2`, `v11-d-test-v2`, `v11-r-test-v2`,
  `v11-clean-test-v2`; the default development benchmark is `v11-d-test-v2`.
- Default experiment config: `input/configs/experiments/v11/baseline.yaml`.
- `input/configs/base.yaml` is the synthetic smoke config only. `make train`,
  `make audit` and `make check` default to the v11 baseline.
- Data counts, current run/eval ids and benchmark hashes live in `STATE.md`; do
  not copy them here.

---

## 1. Setup and commands

```bash
pip install -e /path/to/DLEF   # core + adapters
pip install -e .               # project dependencies
export DLEF_DATA_ROOT=/path/to/Betaregression_data
```

Daily commands:

```bash
make status        # phase, runs, next actions, git
make test          # unittest discovery over core/tests
make audit         # data/split/label/leakage first pass
make check         # brief + config + benchmark + phase gate + full tests
make configs       # validate every config in input/configs
make runs          # experiment summary
make inspect RUN=<run_id>
make handoff
```

- The suite is **unittest**, not pytest. Run one module with:
  `python -m unittest core.tests.test_v11_smoke -v`.
- `core/tests/test_project_memory.py` checks that this file names
  `PROJECT.yaml`, `STATE.md`, `DECISIONS.md` and keeps the `Reading order` /
  `0.2` headings; preserve them when editing.
- v11 tests need `torch` and `DLEF_DATA_ROOT` pointing at the manifest; install
  project deps first (a bare interpreter can fail on imports instead of skip).
- `make check` is stricter than `make test`: if `STATE.md.phase` is `baseline`
  or later, it requires a completed run tagged `baseline` (and later phases
  require a non-baseline, non-debug run).
- `make audit` samples 20 batches per split by default and audits only the
  config's single `data.test_set` (default `d`). Use `--limit 0` for all
  batches; use a config with another `test_set` to audit another cohort.

Train the official baseline with an explicit tag — `make train` does **not**
attach `--tag baseline`:

```bash
python core/commands/train.py \
  --config input/configs/experiments/v11/baseline.yaml \
  --tag baseline
```

Evaluate:

```bash
# val, for model selection
python core/commands/eval.py \
  --checkpoint output/runs/beta-regression/v11-baseline/<run_id>/checkpoints/best.pt \
  --split val

# test on the default D benchmark
python core/commands/eval.py \
  --checkpoint output/runs/beta-regression/v11-baseline/<run_id>/checkpoints/best.pt \
  --split test \
  --benchmark input/benchmarks/v11-d-test-v2.json
```

`eval.py` reuses the training run's `config.resolved.json` (found next to the
checkpoint); `--config` is only a fallback. It does **not** infer `data.test_set`
from `--benchmark`, so for B/R/clean add the matching override:

```bash
python core/commands/eval.py \
  --checkpoint .../checkpoints/best.pt \
  --split test \
  --benchmark input/benchmarks/v11-b-test-v2.json \
  --set data.test_set=b
```

When early stopping is enabled, `train.py` writes `checkpoints/best.pt` at the
best validation epoch and loads it for its final val eval. `last.pt` is the
final epoch and remains the resume checkpoint; use `best.pt` for reported test
evals.

Use `--set KEY=VALUE` on `train.py`/`eval.py` for typed one-off overrides, e.g.
`--set model.name=linear --set loss.name=huber --set data.max_samples=8`.
`CONFIG=...` overrides the Makefile default for `make train`/`check`/`audit`.

---

## 2. Architecture and extension points

```text
core/config.py          typed ProjectConfig schema
core/task.py            composition root only
core/runtime.py         policy / runner / output-root wiring
core/data/build.py      dataset + split + transform selection
core/data/datasets/     v11_frames.py (real data), regression.py (smoke)
core/models/v11.py      ResNet50 + RGB views + depth/geometry tokens
core/losses/            mm_case_bin, mm_rank, mse, huber
core/optimizers/        v11_adamw (backbone/head LRs), adamw, adam, sgd
core/metrics/v11.py     per-bin MAE, worst_bin, acc10
core/commands/          train.py, eval.py, sweep.py only
core/tools/             generic maintenance; scripts/ is project-specific
input/configs/          one file per experiment variant
input/benchmarks/       four immutable frozen manifests
output/                 generated evidence only
```

| Adding | Location | Also update |
|---|---|---|
| dataset reader | `core/data/datasets/<name>.py` | `data/build.py`, `DataConfig.validate` |
| transform | `core/data/transforms/<name>.py` | `data/build.py`, `DataConfig.validate` |
| split strategy | `core/data/splits.py` | `SPLITTERS`, `DataConfig.validate` |
| model | `core/models/<name>.py` | `models/build.py`, `ModelConfig.validate` |
| loss | `core/losses/<name>.py` | `losses/build.py`, `LossConfig.validate` |
| optimizer / scheduler | `core/optimizers/`, `core/schedulers/` | matching `build.py`, config |
| metric | `core/metrics/<name>.py` | `metrics/build.py`, `metrics` list |
| experiment variant | `input/configs/experiments/v11/` | `make new-experiment`, `make configs` |
| frozen benchmark | `input/benchmarks/` | `make new-benchmark` / `core/tools/benchmark.py` |
| project-specific script | `scripts/` | keep it thin |
| generic maintenance | `core/tools/` | do not dump project scripts there |

- `core/task.py` only calls `build_model`, `build_loss`, `build_optimizer`,
  `build_scheduler`, `build_metrics`; data loading belongs to
  `core/data/build.py`; entry points only wire.
- Do not add project-specific code to the installed `dlef` package. Reusable
  framework changes belong in the DLEF repo, not here.
- `core/commands/` must contain only `train.py`, `eval.py`, `sweep.py`
  (enforced by `test_command_surface.py`); put one-off automation in `scripts/`.
- `model.name=v11` returns a `TorchModel`; the pooled feature is exposed through
  `ModelOutput.aux["features"]` for RankSim. `build_v11` loads the optional
  `init_checkpoint`; after construction, forward does no I/O/config/device work.
- `mm_case_bin` is mm-domain, case/bin-balanced SmoothL1; `mm_rank` adds the
  RankSim term from `aux["features"]`. Every loss returns named `LossOutput`
  terms, which are logged.
- `metrics: [v11]` yields `v11/mae`, `v11/mae_lt10`, `v11/mae_ge10`,
  `v11/acc10`, `v11/mae_2_3 … v11/mae_9_10`, and `v11/worst_bin`. Metrics do
  not know the split; `EvalSpec` binds it.
- `v11_adamw` uses separate backbone/head learning rates; `cosine` is the epoch
  scheduler, `constant` is for smoke tests.
- New model/loss/metric code needs a shape or golden test; checkpoint changes
  need a round-trip test.

---

## 3. Data and config gotchas

- `data.split=manifest`: membership comes from the manifest, not a runtime
  random split. Never call `random_split`. A new split or data version is a new
  run, not an edit.
- `data.version=single-v2` is the curated version. `data.test_set` is `b`, `d`,
  `r`, or `clean`; the dataset maps it to `test_<test_set>` in the manifest.
- `data.views` and `model.views` both exist and **must match** (2 or 3); the
  2.5× third view is code-capable but has no current config.
- `model.init_checkpoint` is resolved relative to `$DLEF_DATA_ROOT` (the
  baseline uses `pm/runs/...`). Clear it for smoke tests or when the file is
  absent; otherwise model construction fails.
- `data.seed` is the data-construction seed; the top-level `seed` is the run
  seed. For `v11_frames`, membership still comes from the manifest.
- `output_root` resolves relative to the **project root**, not the current
  working directory. `DLEF_OUTPUT_ROOT` only changes Makefile inspection/report
  paths, not `train.py`'s `output_root`.
- `benchmark.name` in the config must agree with `PROJECT.yaml` and the JSON
  manifest; `make check` verifies this. Changing membership, labels, eval
  transform, metric version, aggregation, or thresholds requires a new
  benchmark.
- `make new-benchmark NAME=...` freezes the **baseline config's**
  `data.test_set` (currently `d`). For B/R/clean, use the underlying tool:

  ```bash
  python core/tools/benchmark.py create \
    --config input/configs/experiments/v11/baseline.yaml \
    --out input/benchmarks/v11-b-test-v2.json \
    --set data.test_set=b --set benchmark.name=v11-b-test-v2
  ```
- Unknown config keys are errors. Add validation in the relevant `validate()`;
  every behavior-affecting choice must be typed config, not a hidden default.
- Seeds are run dimensions, not config variants. One seed per run; never loop
  seeds inside a run. Sweep trial configs go under
  `output/sweeps/<sweep_id>/configs/`.
- `v11/mae_ge10` is `NaN` when the evaluated split has no `>=10 mm` samples;
  report that explicitly rather than treating it as zero.

---

## 4. Training, eval, and evidence

- `Task.train_step` returns `StepOutput(loss=...)`; it must not call
  `backward`, `optimizer.step`, or `zero_grad`. AMP, accumulation, clipping and
  DDP go through `TorchTrainingPolicy`; `custom_train_step` is an escape hatch
  only.
- Checkpoints are written by `Runner` + `TorchCheckpointCodec`; never hand-write
  `.pt` files. Resume restores model, optimizer, scheduler, scaler, RNG,
  epoch/step and policy state; loading only model weights is not a resume.
- DDP: `torchrun --nproc_per_node=N core/commands/train.py ...`; rank 0 writes
  the run. Multi-rank eval/resume is a v0 gap.
- Training eval is for model selection. Final test eval is a separate run named
  `<train-experiment>-eval` and must pass `--benchmark`.
- `eval.py` refuses a benchmark whose `benchmark_id` does not match current data
  and protocol. Do not bypass that check.
- Test access is recorded under `output/test_access/`; a repeated access to the
  same test set produces a `WARN` in the report.
- A `Claim` must reference at least one eval id and pin its `benchmark_id`.
  Reports are generated from claims; never hand-copy numbers:

  ```bash
  python -m dlef.cli report \
    --runs output/runs --claims output/claims \
    --out output/reports/report.md
  ```

- `TensorBoard` is an optional mirror. `manifest.json`, `config.resolved.json`,
  `metrics.jsonl` and `logs/train.log` are the source of truth.
- Before an official run, commit the code. `code_version: no-git` means the run
  is exploratory, not reportable.

---

## 5. Experiment workflow and gates

Identifiers: `project` (beta-regression), `experiment` (v11, ranksim, proj,
backbone…), `run` (one resolved config × one seed), `eval` (one checkpoint ×
one eval spec × one split), `claim` (one reported number/table/figure).

1. Run `make status`; read `STATE.md` §4.1 and pick one item.
2. Make sure `make check` is green before a real baseline.
3. Create a variant with `make new-experiment NAME=<variant> EXPERIMENT=v11`;
   do not hand-edit an old config.
4. Smoke it with `--set epochs=1 --set max_steps=1` or
   `--set data.max_samples=8`.
5. Train the baseline with `--tag baseline`; inspect manifest, resolved config,
   metrics and checkpoint.
6. Evaluate val for model selection; test only through frozen benchmarks.
7. For comparisons, use a sweep; one trial = one run. Use `SubprocessSweep` for
   real GPU work; `LocalSweep` is only for lightweight in-process tasks.
8. Create a claim for each reported number, then regenerate the report.
9. Run `make gc-dry` and update `STATE.md` / `DECISIONS.md`.

Phase gates in `make check` (do not advance `STATE.md.phase` early):

- `baseline`: at least one completed run tagged `baseline`.
- `ablation`: at least one completed non-baseline, non-debug run.
- `benchmark`: frozen manifest exists and `STATE.md` names its id.
- `report`: at least one claim and `output/reports/report.md`.

Rules:

- One active experiment at a time.
- Never tune on test; do not report a single lucky seed.
- Tag runs at launch (`baseline`, `reported`, `debug`).
- Failed runs remain recorded; do not delete them manually.
- Cleanup is dry-run first: `make gc-dry` prints the plan; rerun the printed
  command with `--apply`. `baseline`/`reported` tags protect runs; never
  `rm -rf output/runs/...`.

---

## 6. Forbidden

- Do not copy `dlef` core code into this project or add project code to the
  installed package.
- Do not call `loss.backward()` / `optimizer.step()` inside a Task.
- Do not write run manifests, metrics, eval records or checkpoints by hand.
- Do not edit completed manifests/metrics/eval records; add a new run instead.
- Do not delete runs/artifacts manually; use `dlef gc`.
- Do not edit a frozen benchmark; create a new version/file.
- Do not run `--split test` without `--benchmark`.
- Do not compare numbers across different `benchmark_id`s.
- Do not tune on test or select a result from one seed.
- Do not make TensorBoard/W&B/MLflow the only record.
- Do not put secrets or absolute data paths in config, logs or project memory.
- Do not put project scripts in `core/commands/` or `core/tools/`.
- Do not leave `STATE.md` stale after a run.

---

## 7. Definition of done

1. Config validates; every selector resolves; one batch/step smoke passes.
2. `make audit` reviewed (splits, sample ids, labels, group overlap) and
   `make check` passes; if a phase gate is unmet, do not claim done.
3. Train wrote manifest, resolved config, metrics and checkpoints (`last.pt`,
   plus `best.pt` when early stopping is enabled).
4. Test eval used a frozen benchmark and wrote an eval record bound to the
   checkpoint digest and `benchmark_id`.
5. Claim → eval → checkpoint → run is traceable; report is generated, not
   hand-copied.
6. `make gc-dry` shows only intended deletions; `STATE.md` is current;
   `DECISIONS.md` records hard-to-reverse choices; code is committed.
