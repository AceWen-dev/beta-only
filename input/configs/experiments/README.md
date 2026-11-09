# Experiment configs

One directory per experiment, one YAML per variant:

```text
experiments/
  v11/
    baseline-best.yaml
    ranksim-best.yaml
    proj-best.yaml
    backbone-full-best.yaml
    backbone-partial-best.yaml
    backbone-frozen-best.yaml
```

Do not create per-seed files; a seed is a run. Each file is a complete config
(the project has no YAML inheritance); copy `../base.yaml` only for a synthetic
smoke variant, otherwise copy the v11 baseline and change the variant fields.

`baseline-best.yaml` is the default for `make train`, `make check` and `make audit`.
