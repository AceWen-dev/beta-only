# Data package

Reusable data code lives here; project-specific scripts under `scripts/` stay thin.

```text
datasets/    # runtime Dataset classes returned by build.py
ingest/      # source readers / acquisition helpers (create when needed)
prepare/     # cleaning, label mapping, dedup helpers (create when needed)
transforms/  # per-sample train/eval transforms
splits.py    # materialized membership + leakage checks
paths.py     # $DLEF_DATA_ROOT resolution
spec.py      # DataBundle + DataSpec
loaders.py   # DataLoader assembly
benchmark.py # frozen test set + metric protocol
build.py     # data.name / transform registry + make_loaders
```

Example: a source-acquisition CLI belongs under `scripts/` (organize it however
that project needs); its reusable reader/decoder belongs in
`core/data/ingest/`. The CLI writes raw data under
`$DLEF_DATA_ROOT/raw/...`; it does not write into `output/runs/`.
