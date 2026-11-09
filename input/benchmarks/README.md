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
  --config input/configs/experiments/v11/baseline-best.yaml \
  --out input/benchmarks/v11-d-test-v2.json
```

Never edit an existing benchmark in place. `eval.py --split test` verifies the
manifest against the current data and protocol before running; a mismatch is
refused rather than silently accepted.

## Reporting suites (`benchmark2*.json`)

`benchmark2.json` is a **reporting suite** (company + B + R) consumed by
`scripts/plot_experiment.py --suite`; it is not a frozen benchmark manifest.
Since 2026-11-05 the R member carries `"min_truth_mm": 2.0`: the R 1–2 mm bin
(no company-v1 training samples, D-0047) is excluded from the suite's
per-bin/`lt10`/MAE calculations. Since 2026-11-06 the B member carries
`"exclude_bins": [7]`: the PolypSize B 7–8 mm bin is a single flat-lesion case
(5 frames, detector boxes identical to the stored ROI, S/W 0.156–0.169 vs
0.414/0.448 for B 8–9/9–10) and is treated as an outlier, so that bin is not
reported in benchmark2. `benchmark2-big.json` and `benchmark2-endo.json` carry
the same R/B filters.

The frozen `v11-r-test-v2.json` is **unchanged**; an official
`eval.py --benchmark v11-r-test-v2.json` still evaluates every frozen frame.
If an official R benchmark without 1–2 mm is required, create a new version
(new split/filter), never edit the frozen file.
