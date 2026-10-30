# Input

Immutable experiment inputs:

```text
input/
  input/configs/       # experiment variants
  input/benchmarks/    # frozen benchmark manifests
```

Dataset bytes are not stored here; they live outside the repo under
`$DLEF_DATA_ROOT` (see `core/data/paths.py`). `input/` contains the declarations
and manifests that pin the data.
