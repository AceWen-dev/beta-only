# Experiment configs

One directory per experiment, one YAML per variant:

```text
experiments/
  v11/
    baseline.yaml
    ranksim.yaml
    proj.yaml
    backbone-full.yaml
    backbone-partial.yaml
    backbone-frozen.yaml
```

Do not create per-seed files; a seed is a run. Each file is a complete config
(the project has no YAML inheritance); copy `../base.yaml` only for a synthetic
smoke variant, otherwise copy the v11 baseline and change the variant fields.

`baseline.yaml` is the default for `make train`, `make check` and `make audit`.
