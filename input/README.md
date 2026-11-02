# Input

Immutable experiment inputs:

```text
input/
  configs/       # experiment variants
  benchmarks/    # frozen benchmark manifests
  datasets/      # dataset cards: version, manifest hash, split counts
```

Dataset bytes are not stored here; they live outside the repo under
`$DLEF_DATA_ROOT` (see `core/data/paths.py`). `input/` contains the declarations
and small manifests that pin the data.

The v11 dataset card is:

```text
input/datasets/v11_frames/single-v2.yaml
```
