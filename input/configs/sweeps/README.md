# Sweep configs

Optional parameter-space descriptions for a sweep. Generated per-trial configs
belong under `output/sweeps/<sweep_id>/configs/`, not here.

The template's `commands/sweep.py` currently defines its parameter space in
code; if you move it to YAML, keep one sweep spec per file.
