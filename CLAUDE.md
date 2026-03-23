# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install (requires Python >= 3.10, uv is the primary tool)
uv sync --extra dev

# Optional: Basilica GPU cloud support
uv sync --extra basilica

# Run tests
uv run pytest -q                          # all tests
uv run pytest tests/test_contract.py -q   # single test file
uv run pytest -k test_name -q             # single test by name

# Lint & type check
uv run ruff check src/ tests/
uv run mypy src/

# Run the CLI
uv run autoresearch-rl examples/minimal-trainable-target/config.yaml
uv run autoresearch-rl validate examples/minimal-trainable-target/config.yaml
uv run autoresearch-rl print-config examples/minimal-trainable-target/config.yaml
uv run autoresearch-rl examples/minimal-trainable-target/config.yaml --override controller.max_wall_time_s=10
```

## Architecture

Two independent loop systems coexist in the codebase:

### 1. Continuous CLI loop (primary, actively used)
Runtime path: `cli.py` -> `controller/continuous.py` -> `target/*` -> `telemetry/*`

- **Targets** (`target/`): Pluggable adapters implementing `TargetAdapter` protocol (run + eval). Three types:
  - `CommandTarget`: runs local/Docker commands, injects params via `AR_PARAMS_JSON` and `AR_PARAM_<NAME>` env vars
  - `HttpTarget`: calls remote endpoints (vLLM/sglang)
  - `BasilicaTarget` (`target/basilica.py`): deploys each training iteration as a containerized GPU job on Basilica cloud; handles health-check bootstrapping, log polling, and cleanup
  - Registry in `target/registry.py` builds the correct adapter from config
- **Pipeline step** (`prepare_cmd`): Optional config field. When set, the target runs `prepare_cmd` once before the iteration loop (CommandTarget) or once per container before `train_cmd` (BasilicaTarget). This is the frozen data/evaluation boundary — `prepare.py` produces data files that `train.py` reads. No Python import between them.
- **Frozen/mutable boundary**: `prepare.py` (frozen) owns data loading, answer extraction, reward computation, evaluation — "what is correct". `train.py` (mutable) owns the training algorithm, reward function, optimizer, generation strategy — "how to get there". The LLM can modify `train.py` via code diffs but can never touch `prepare.py`. This prevents the LLM from gaming evaluation. In hybrid mode, when param search stalls, the LLM proposes diffs to `train.py` (e.g., adding partial-credit rewards, gradient accumulation, or a different sampling strategy).
- **Policy**: Parameter proposal strategies. Five types across two modules:
  - `policy/search.py`: `GridPolicy` (exhaustive combos), `RandomPolicy` (seeded uniform), `StaticPolicy` (no overrides)
  - `policy/llm_search.py`: `LLMParamPolicy` calls any OpenAI-compatible chat API to propose params from full experiment history; falls back to seeded random on failure; retries on 429/502/503 with exponential backoff + jitter
  - `policy/llm_diff.py`: `LLMDiffPolicy` proposes code modifications as unified diffs with correction retry on validation failure
  - `policy/hybrid.py`: `HybridPolicy` starts with param exploration, switches to code diffs when params stall, falls back to param mode on consecutive diff failures
  - `policy/learned.py` + `policy/learned_search.py`: learned policy using PPO-style updates
- **Controller** (`controller/continuous.py`): Orchestrates the loop with stop guards (wall time, no-improvement streak, failure rate). Each iteration: propose params -> train -> eval -> keep/discard -> emit telemetry
- **Config** (`config.py`): Pydantic models for all config sections. YAML config validated via `RunConfig.model_validate()`

### 2. Legacy contract/sandbox loop (not used by CLI)
Runtime path: `controller/loop.py` -> `sandbox/runner.py` -> `eval/*`

- **Sandbox** (`sandbox/`): Validates diffs (`validator.py`, `ast_policy.py`), applies patches via git worktrees, runs trials with early stopping and power-law forecasting
- **Contract** (`controller/contract.py`): Enforces frozen/mutable file boundaries - diffs can only touch the designated mutable file
- **Eval** (`eval/`): `judge.py` does heuristic next-state voting, `scoring.py` computes composite scores, `metrics.py` parses stdout for val_bpb/loss
- **Policy** (`policy/baselines.py`, `policy/learned.py`): Diff-proposal policies - `GreedyLLMPolicy` proposes code changes, `LearnedDiffPolicy` learns weights via PPO-style updates

### Telemetry (`telemetry/`)
Shared by both loops:
- `events.py`: JSONL trace emission
- `ledger.py`: TSV results ledger with comparability metadata
- `manifest.py`: per-run manifest files
- `comparability.py`: hardware fingerprinting and budget-mode checks (strict mode blocks mismatched runs)
- `distill.py`: distillation sample collection (legacy loop only)

### Key design patterns
- **Keep/discard**: iterations that beat the best score are "kept" with versioned artifacts in `artifacts/versions/v####/`
- **Comparability enforcement**: runs record hardware fingerprint + budget mode; strict mode rejects budget mismatches
- **Objective direction**: `direction: min` or `max` in config; internally normalized via `_score()` so lower is always better
- Ruff line length: 100
