# Frozen benchmarks

One JSON file per frozen benchmark:

```text
v11-b-test-v2.json
v11-d-test-v2.json
v11-r-test-v2.json
v11-clean-test-v2.json
```

These are the four project test sets (B / D / R / cleaned company set). The
default development benchmark is `v11-d-test-v2`.

A benchmark freezes:

```text
test membership + label version + eval transform
+ metric implementation/version + aggregation + thresholds
```

The JSON is immutable. If membership, labels, transform or metric protocol
changes, create a new benchmark file/version with:

```bash
python core/tools/benchmark.py create \
  --config input/configs/experiments/v11/baseline.yaml \
  --out input/benchmarks/v11-d-test-v2.json
```

Never edit an existing benchmark in place. `eval.py --split test` verifies the
manifest against the current data and protocol before running; a mismatch is
refused rather than silently accepted.
