# Beta-regression — endoscopic polyp size estimation

This is a DLEF research project. It estimates polyp diameter in millimetres from
a single endoscopic frame plus a detected polyp box, using a frozen relative-depth
feature and ten geometric scalars. The primary question is whether the `<10 mm`
range can be measured with **per-1-mm-bin MAE < 2 mm** on four frozen test sets.

The framework is installed as the `dlef` package; it is **not** vendored in this
repository. Project code lives in `core/`, declarative inputs in `input/`, and
generated evidence in `output/`.

**Read first**

- [`PROJECT.yaml`](PROJECT.yaml) — stable brief: question, unit, data contract,
  splits, benchmark, primary metric, success criterion.
- [`STATE.md`](STATE.md) — current snapshot: what exists, what is frozen, what
  is next.
- [`DECISIONS.md`](DECISIONS.md) — append-only design/experiment decisions.
- [`AGENTS.md`](AGENTS.md) — project rules for coding agents and humans.
- [`TUTORIAL.md`](TUTORIAL.md) — step-by-step runbook.

## What this project tests

- Unit of analysis: **case** (`case_id`); frames from one case never cross splits.
- Labels: a single measured polyp size in **mm** per frame.
- Input: whole-frame RGB + tight 1.25× RGB view, frozen relative-depth features,
  and ten geometric scalars.
- Output: diameter in mm and a `<10 / >=10 mm` decision.
- Primary metric: `v11` (`v11/worst_bin` is the worst non-empty 0–10 mm bin MAE;
  `v11/mae_lt10` and `v11/acc10` are reported alongside).
- Test sets: four frozen benchmarks, one per evaluation cohort.

| Benchmark | Test set | `benchmark_id` |
|---|---|---|
| `v11-b-test-v2` | B | `sha256:911defbf…6ecb` |
| `v11-d-test-v2` | D (dev/default) | `sha256:2c4d4bee…8147` |
| `v11-r-test-v2` | R | `sha256:983c48b6…24be` |
| `v11-clean-test-v2` | cleaned company set | `sha256:6a76a698…97ad` |

Full ids are in [`STATE.md`](STATE.md) and the JSON manifests under
`input/benchmarks/`. Numbers are comparable only within the same
`benchmark_id`.

## Layout

```text
input/
  configs/
    base.yaml                       # synthetic smoke only; not the project default
    experiments/v11/                # baseline, ranksim, proj, backbone variants
    sweeps/                         # optional parameter-space specs
  benchmarks/                       # four frozen benchmark manifests
core/
  config.py                         # typed project config
  data/
    datasets/v11_frames.py          # manifest + RGB views + depth + scalars
    datasets/regression.py          # synthetic smoke dataset
    splits.py                       # holdout / manifest
    transforms/
    paths.py                        # $DLEF_DATA_ROOT + layout
    spec.py, loaders.py, benchmark.py, build.py
  models/
    v11.py                          # ResNet50 + views + depth/geometry tokens
    linear.py, mlp.py               # smoke/baseline alternatives
  losses/
    mm_case_bin.py                  # mm-domain, case/bin-balanced SmoothL1
    mm_rank.py                      # mm_case_bin + RankSim
    huber.py, mse.py
  optimizers/v11_adamw.py           # backbone/head learning-rate groups
  schedulers/cosine.py
  metrics/v11.py                    # per-bin MAE, worst_bin, acc10
  task.py                           # composition root only
  runtime.py                        # policy / runner / output-root wiring
  commands/train.py, eval.py, sweep.py
  tools/                            # generic maintenance utilities
  tests/                            # v11 contract, smoke, benchmark tests
scripts/                            # project-specific scripts, intentionally free-form
output/
  runs/ artifacts/ sweeps/ test_access/ claims/ reports/
PROJECT.yaml, STATE.md, DECISIONS.md
AGENTS.md, TUTORIAL.md, README.md
pyproject.toml, Makefile
```

## Install

The DLEF core is installed separately from this repository:

```bash
pip install -e /path/to/DLEF   # dlef + adapters/torch
pip install -e .               # this project (torch, torchvision, numpy, cv2, ...)
export DLEF_DATA_ROOT=/path/to/Betaregression_data
```

Dataset bytes are not committed. `$DLEF_DATA_ROOT` contains at least:

```text
curated/v11_frames/single-v2/manifest.json
curated/v11_frames/single-v2/...   # images + depth feature files
```

The manifest rows use paths relative to `$DLEF_DATA_ROOT`.

## Quick start

```bash
make test          # unit tests; v11 tests need torch + DLEF_DATA_ROOT
make audit         # data/split/leakage first pass on the default config
make check         # brief + config + benchmark + phase gate + tests

make train         # v11 baseline from input/configs/experiments/v11/baseline.yaml
make runs          # experiment summary
make status        # phase / runs / next actions / git
```

`make train` uses the real v11 baseline. `input/configs/base.yaml` is the
synthetic 1-step smoke config and is not the default for `train`, `check` or
`audit`.

Override the config for any command:

```bash
make train CONFIG=input/configs/experiments/v11/ranksim.yaml
make check CONFIG=input/configs/experiments/v11/ranksim.yaml
```

## Train / eval

Train and tag a run:

```bash
python core/commands/train.py \
  --config input/configs/experiments/v11/baseline.yaml \
  --tag baseline
```

Training does an in-run validation pass for model selection. The final test
evaluation is a separate run and must use a frozen benchmark:

```bash
python core/commands/eval.py \
  --checkpoint output/runs/beta-regression/v11-baseline/<run_id>/checkpoints/best.pt \
  --split test \
  --benchmark input/benchmarks/v11-d-test-v2.json
```

Training writes `best.pt` at the best validation epoch when early stopping is
enabled; `train.py` loads it for its final val eval. Use `best.pt` for reported
test evals; `last.pt` is the final epoch and remains the resume checkpoint.

For the other cohorts, change `--benchmark` **and** add the matching
`--set data.test_set=b|r|clean`; `eval.py` does not infer the cohort from the
benchmark path. `eval.py` re-verifies the current test membership, labels,
transform and metric protocol against the stored manifest; a mismatch is
refused, not silently accepted.

## Configs

One config file per experiment variant, not per run or seed:

```text
input/configs/experiments/v11/
  baseline.yaml          # mm_case_bin, full backbone
  ranksim.yaml           # mm_rank
  proj.yaml              # independent projection + RankSim
  backbone-full.yaml
  backbone-partial.yaml
  backbone-frozen.yaml
```

A seed is a run dimension. Do not create `seed1.yaml`, `seed2.yaml`. Generated
sweep trial configs belong under `output/sweeps/<sweep_id>/configs/`, not in
this repository.

The resolved config for a run is `output/runs/.../config.resolved.json`; the
manifest records its digest and original source. Use:

```bash
python core/tools/configs.py list
python core/tools/runs.py --experiment v11-baseline
python core/tools/run_info.py --run <run_id>
```

## Reproducibility

`train.py` and `eval.py` call `dlef.seed.seed_everything(config.seed)` before
building data and models. `Runner.fit` applies the seed again before the first
step. Checkpoints store Python, NumPy, Torch and CUDA RNG state when those
libraries are present.

This means a run has two pieces of identity that must travel together:
the resolved config digest and the code version. Commit code before official
runs. A run with `code_version: no-git` is exploratory, not reportable.

## TensorBoard

`tensorboard.enabled: true` writes event files under each run directory:

```text
output/runs/<project>/<experiment>/<run_id>/tensorboard/
```

Start the local UI:

```bash
make tensorboard
# equivalent to:
python -m tensorboard.main --logdir output/runs --port 6006
```

Open <http://localhost:6006>. Remote machine:

```bash
make tensorboard TENSORBOARD_ARGS=--bind_all
# or keep it local and tunnel from your laptop:
ssh -L 6006:localhost:6006 user@host
```

What is logged:

- `train/loss` and named loss terms (`train/mm_case_bin`, `train/rank`);
- `val/v11/mae`, `val/v11/worst_bin`, `val/v11/acc10` and per-bin metrics;
- `run/config` and `run/manifest` as text (Text tab).

Runs are grouped by their `output/runs/...` path; use the experiment/run names
to compare. TensorBoard is a **mirror**, not the record: `manifest.json`,
`config.resolved.json`, `metrics.jsonl` and `logs/train.log` are the source of
truth. Set `tensorboard.enabled: false` to disable it. Use
`TENSORBOARD_PORT=6007` to change the port.

## Inspect a run

```bash
make runs
make runs-all
make inspect RUN=<run_id>
```

Each run directory contains:

```text
manifest.json          # status, seed, code/config digest, data versions, tags
config.resolved.json   # exact config used for this run
metrics.jsonl          # machine-readable train/val metrics
logs/train.log         # human-readable log, loss terms, early stopping
checkpoints/last.pt    # final epoch; model + optimizer/scheduler/policy + RNG
checkpoints/best.pt    # best validation epoch, when early stopping is enabled
evals/<eval_id>.json   # eval records bound to checkpoint + benchmark
```

Do not edit completed manifests, metrics or eval records. A correction is a new
run or an append-only annotation.

## Variants and benchmark lifecycle

```bash
make new-experiment NAME=lr-0.005 EXPERIMENT=v11
make new-benchmark NAME=v11-d-test-v2
```

`new-experiment` creates `input/configs/experiments/v11/lr-0.005.yaml` and
validates it. `new-benchmark` freezes a new test manifest; it never edits an old
one. Changing test membership, label version, eval transform or metric protocol
requires a new benchmark.

## Claims and report

A reported number must resolve through `claim → eval → checkpoint → run`:

```bash
python -m dlef.cli report \
  --runs output/runs \
  --claims output/claims \
  --out output/reports/report.md
```

Do not hand-copy numbers into a paper or report.

## Cleanup

```bash
make gc-dry
# inspect the plan; only then add --apply
```

`baseline` and `reported` runs are protected by tag. Never delete `output/runs`
by hand.

## Current status

See [`STATE.md`](STATE.md). The original `single-v1` boxes were corrupted
(938/2044 train frames and all 206 clean-test frames used the whole content
region), so v1 runs are retained only for traceability. `single-v2` restores the
source boxes and has four new frozen benchmarks; the authoritative baseline must
be rerun on v2.
