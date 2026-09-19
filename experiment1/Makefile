PY ?= python3
export PYTHONPATH := .

.PHONY: help selftest verify-tasks smoke pilot analyze prompts clean

help:
	@echo "selftest      validate detectors against scripted agents (no API keys)"
	@echo "verify-tasks  check task solvability, impossibility and the twin diff"
	@echo "smoke         run the mock grid end to end"
	@echo "pilot         run configs/pilot.yaml (needs API keys and docker)"
	@echo "analyze       summarise results/smoke"
	@echo "prompts       print every rung of the ladder"
	@echo "clean         remove results/ and caches"

selftest:
	$(PY) -m harness selftest

verify-tasks:
	$(PY) -m harness verify-tasks

smoke:
	rm -rf results/smoke
	$(PY) -m harness run --config configs/smoke.yaml --yes

pilot:
	$(PY) -m harness run --config configs/pilot.yaml

analyze:
	$(PY) -m harness analyze results/smoke --csv results/smoke/cells.csv

prompts:
	@for rung in P0 P1 P2 P3 P4 P5 P6; do \
		echo "############ $$rung ############"; \
		$(PY) -m harness show-prompt $$rung --task perf_memo | sed -n '1,40p'; \
		echo; \
	done

clean:
	rm -rf results .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
