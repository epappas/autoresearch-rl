# Common workflows for autoresearch-rl. All targets are thin wrappers
# around `uv run` invocations — nothing here changes behavior, just ergonomics.
#
# Quick reference:
#   make test                  # full test suite
#   make test-fast             # skip the slow integration tests
#   make lint                  # ruff
#   make typecheck             # mypy
#   make check                 # test + lint + typecheck
#   make showcase              # run examples/parallel-cancel-showcase end-to-end
#   make showcase-clean        # delete showcase artifacts
#   make validate CONFIG=path  # run config_validate against any yaml
#   make help                  # this list

.DEFAULT_GOAL := help
.PHONY: help test test-fast lint typecheck check showcase showcase-clean validate sync

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[1m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Install runtime + dev dependencies via uv
	uv sync --extra dev

test: ## Run the full pytest suite (~95 s — includes showcase determinism)
	uv run pytest -q

test-fast: ## Run pytest excluding showcase determinism (~30 s)
	uv run pytest -q --ignore=tests/test_showcase_determinism.py

lint: ## Ruff lint check
	uv run ruff check src/ tests/

typecheck: ## Mypy on src/
	uv run mypy src/

check: lint typecheck test ## All three: lint + typecheck + tests

showcase: ## Run the parallel-cancel-showcase end-to-end (~13 s)
	bash examples/parallel-cancel-showcase/run.sh

showcase-clean: ## Wipe showcase artifacts/ and data/
	rm -rf examples/parallel-cancel-showcase/artifacts \
	       examples/parallel-cancel-showcase/artifacts-serial \
	       examples/parallel-cancel-showcase/data

showcase-chart: ## Regenerate examples/parallel-cancel-showcase/progress.png from latest run (needs --extra chart)
	uv run python scripts/progress_chart.py \
	    examples/parallel-cancel-showcase/artifacts/results.tsv \
	    -o examples/parallel-cancel-showcase/progress.png \
	    --direction min

validate: ## Validate a config: make validate CONFIG=path/to/config.yaml
	@if [ -z "$(CONFIG)" ]; then echo "usage: make validate CONFIG=path/to/config.yaml"; exit 2; fi
	uv run autoresearch-rl validate $(CONFIG)
