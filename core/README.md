# Core

Layer 2: project source code.

```text
config.py     # typed schema
data/         # datasets, splits, transforms, loaders, benchmark manifest builder
models/       # architecture builders
losses/       # loss builders
optimizers/
schedulers/
metrics/
task.py       # composition root
runtime.py    # policy / runner / output-root wiring
tensorboard.py # optional TensorBoard mirror callback
commands/     # train / eval / sweep (core entries only)
tools/        # generic maintenance utilities
tests/        # tests for this project
```

Reusable logic lives here; the sibling `scripts/` directory stays thin. If code
is reusable across projects, move it to DLEF core/adapters instead.
