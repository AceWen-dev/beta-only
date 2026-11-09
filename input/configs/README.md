# Config layout

One config file per **experiment variant**, not per run and not per seed.

```text
input/configs/
  base.yaml                         # synthetic smoke only; not the default
  experiments/
    v11/
      baseline-best.yaml          # project default: mm_case_bin + best
      ranksim-best.yaml             # mm_rank + best
      proj-best.yaml              # projection + mm_rank + best
      backbone-full-best.yaml
      backbone-partial-best.yaml
      backbone-frozen-best.yaml
  sweeps/
    <sweep>.yaml                    # parameter space, optional
```

Rules:

- `base.yaml` is a synthetic 1-D regression smoke config. It is useful for
  checking the framework wiring, not for reporting project results.
- `make train`, `make check`, `make audit` and `make sweep` default to
  `input/configs/experiments/v11/baseline-best.yaml`; override with `CONFIG=...`.
- A seed is a run dimension. Do not create `seed1.yaml`, `seed2.yaml`, ...
- A sweep describes a parameter space; generated per-trial configs belong under
  `output/sweeps/<sweep_id>/configs/`, not in the repo.
- Keep one variant per file; do not mutate an old config for a new experiment.
- The `experiment` field identifies the variant: `experiments/v11/lr.yaml`
  should set `experiment: v11-lr`. `make new-experiment` does this for you.
- `output_root` and `tensorboard` are workspace/output settings, not comparison
  dimensions.
- The run manifest records `config_source` and `config_digest`; use
  `core/tools/runs.py` to list runs and `core/tools/run_info.py` to inspect one.

```bash
python core/tools/configs.py list
python core/tools/runs.py --experiment v11-baseline
python core/tools/run_info.py --run <run_id>
```
