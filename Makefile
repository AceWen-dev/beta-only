PYTHON ?= python
CONFIG ?= input/configs/experiments/v11/baseline.yaml
OUTPUT ?= $(if $(DLEF_OUTPUT_ROOT),$(DLEF_OUTPUT_ROOT),output)
EXPERIMENT ?= v11
GC_METRIC ?= v11/worst_bin
TENSORBOARD_PORT ?= 6006
TENSORBOARD_ARGS ?=

.PHONY: train eval inspect runs runs-all configs sweep report gc-dry check audit new-experiment new-benchmark status handoff test doctor tensorboard

train:
	$(PYTHON) core/commands/train.py --config $(CONFIG)

eval:
	@test -n "$(CHECKPOINT)" || (echo "usage: make eval CHECKPOINT=$(OUTPUT)/runs/.../checkpoints/last.pt"; exit 2)
	$(PYTHON) core/commands/eval.py --checkpoint $(CHECKPOINT)

inspect:
	@test -n "$(RUN)" || (echo "usage: make inspect RUN=<run_id>"; exit 2)
	$(PYTHON) core/tools/run_info.py --runs $(OUTPUT)/runs --run $(RUN)

runs:
	$(PYTHON) core/tools/runs.py --runs $(OUTPUT)/runs

runs-all:
	$(PYTHON) core/tools/runs.py --runs $(OUTPUT)/runs --all

tensorboard:
	$(PYTHON) -m tensorboard.main --logdir $(OUTPUT)/runs --port $(TENSORBOARD_PORT) $(TENSORBOARD_ARGS)

configs:
	$(PYTHON) core/tools/configs.py list --check

new-experiment:
	@test -n "$(NAME)" || (echo "usage: make new-experiment NAME=<variant> [EXPERIMENT=<experiment>]"; exit 2)
	$(PYTHON) core/tools/new_experiment.py --name $(NAME) --experiment $(EXPERIMENT)

new-benchmark:
	@test -n "$(NAME)" || (echo "usage: make new-benchmark NAME=<benchmark>"; exit 2)
	$(PYTHON) core/tools/new_benchmark.py --name $(NAME)

status:
	$(PYTHON) core/tools/status.py

handoff:
	$(PYTHON) core/tools/status.py --handoff

sweep:
	$(PYTHON) core/commands/sweep.py --config $(CONFIG)

report:
	$(PYTHON) -m dlef.cli report --runs $(OUTPUT)/runs --claims $(OUTPUT)/claims --out $(OUTPUT)/reports/report.md

gc-dry:
	$(PYTHON) -m dlef.cli gc --runs $(OUTPUT)/runs --sweeps $(OUTPUT)/sweeps --keep-last 3 --keep-best 1 --metric-key $(GC_METRIC) --keep-tag reported --keep-tag baseline --artifact-ttl-days 30 --sweep-ttl-days 30

check:
	$(PYTHON) core/tools/check.py --project PROJECT.yaml --config $(CONFIG)
	$(PYTHON) -m unittest discover -s core/tests -v

audit:
	$(PYTHON) core/tools/audit_data.py --config $(CONFIG) --project PROJECT.yaml

test:
	$(PYTHON) -m unittest discover -s core/tests -v

doctor:
	$(PYTHON) -m dlef.cli doctor
