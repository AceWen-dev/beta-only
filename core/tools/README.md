# Tools

Generic project maintenance utilities shipped with the template. They are not
experiment commands and not project-specific scripts.

```text
core/tools/
  benchmark.py   # create / verify a frozen benchmark manifest
  check.py       # initialization + phase gate
  audit_data.py  # data/split/leakage first pass
  configs.py     # list / validate experiment configs
  new_experiment.py  # create a variant from base.yaml
  new_benchmark.py   # freeze a named benchmark manifest
  status.py      # one-screen status / handoff
  runs.py        # experiment summary / run index (filters, --all, --json)
  run_info.py    # one run: config -> benchmark -> source files
  _bootstrap.py  # internal: find the core when running in-repo
```

Rules:

- `core/commands/` is only train/eval/sweep; `core/tools/` is only generic maintenance.
- Project-specific automation goes under `scripts/`, not here.
- Tools do not write run manifests, metrics or checkpoints.
