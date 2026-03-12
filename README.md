# AutoResearch-RL Scaffold

Minimal production-oriented scaffold for autonomous training-script research loops.

## Goals
- Deterministic, auditable experiment loop
- Safe code-edit sandboxing
- Fixed-budget comparable evaluations
- Trace-first observability
- Next-state-aware scoring (evaluate turn _t_ using state from _t+1_)
- Dual-signal optimization hooks (evaluative + directional)

## Layout
- `src/autoresearch_rl/controller` – run orchestration and state machine
- `src/autoresearch_rl/sandbox` – patch validation + constrained execution
- `src/autoresearch_rl/policy` – proposal interface (LLM/RL plug points)
- `src/autoresearch_rl/eval` – scoring, early-stop forecasting hooks
- `src/autoresearch_rl/telemetry` – trace/event emitters
- `configs/` – runtime and benchmark configs
- `scripts/` – CLI entry helpers
- `tests/` – unit/integration tests

## Quickstart (uv-first)
```bash
# install deps + dev extras from pyproject.toml
uv sync --extra dev

cp configs/example.yaml configs/local.yaml
uv run python scripts/run_once.py --config configs/local.yaml
```

## Quickstart (pip fallback)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp configs/example.yaml configs/local.yaml
python scripts/run_once.py --config configs/local.yaml
```

## Example target projects
Concrete targets with real `train.py` are included in:

- `examples/minimal-trainable-target/train.py` (synthetic deterministic target)
- `examples/deberta-prompt-injection/train.py` (real Hugging Face DeBERTa fine-tuning target)

Run direct trials:

```bash
uv run python examples/minimal-trainable-target/run.py
uv run python examples/deberta-prompt-injection/run.py
```

Run DeBERTa benchmark sweep:

```bash
uv run python scripts/benchmark_deberta_example.py
```

Config: `examples/deberta-prompt-injection/deberta-example.yaml`

## Comparability enforcement
The scaffold supports strict fair-comparison mode for benchmark runs:
- `budget_mode: fixed_wallclock`
- run budget must match configured `expected_budget_s`
- optional hardware fingerprint lock (`expected_hardware_fingerprint`)
- strict mode blocks non-comparable runs and records comparability metadata in `results.tsv`

## Safety defaults
- mutable scope limited to target file list
- no network in runner (expected to be enforced by runtime/container)
- hard wall-clock timeout per run
- every run emits structured JSONL trace

## v0.2 architecture extensions
- event-driven async pipeline: proposal → trial → judge/score
- next-state judging with majority vote hooks
- composite scoring (`val_bpb` + status penalties + evaluative score + hint bonus)
- optional git-backed patch apply + rollback in trial runner (auto-inits git if missing)
- optional early-stop threshold checks in trial runner
- richer telemetry (`event_id`, `episode_id`, `sample_type`) and replayable manifests

## Contract and policy files
- Three-file contract doc: `docs/THREE_FILE_CONTRACT.md`
- Default agent policy: `programs/default.md`
- Canonical results ledger: `results.tsv` (auto-initialized by loop)
- Fixed-budget comparability policy: `experiment.comparability` in config
- Strict contract mode blocks out-of-scope mutations (frozen/program/non-mutable files)

## Research notes
- `docs/research/SDFT-Softmax-Divergence-Fine-Tuning.md`
- `docs/research/SDPO-Self-Distilled-Policy-Optimization.md`
- `docs/research/AutoResearch-RL-Perpetual-Self-Evaluating-RL-Agents.md`
