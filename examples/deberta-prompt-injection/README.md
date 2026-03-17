# deberta-prompt-injection

Fine-tune DeBERTa for prompt-injection classification using the `hybrid` policy:
LLM-guided hyperparameter search for the first N iterations, then code diff proposals when stalled.

## Prerequisites

```bash
export CHUTES_API_KEY="your-key"
pip install -r examples/deberta-prompt-injection/requirements.txt
```

## Run (local)

```bash
bash examples/deberta-prompt-injection/run.sh
```

## Agentic workflow

```bash
# Check experiment state (JSON output — agent-readable)
uv run autoresearch-rl status \
  --config examples/deberta-prompt-injection/config.yaml --last 5

# Inject explicit hyperparameters and run one iteration
uv run autoresearch-rl run-one \
  --config examples/deberta-prompt-injection/config.yaml \
  --params '{"learning_rate": 0.00002, "epochs": 2, "batch_size": 8}'

# Inject a code diff and run one iteration
uv run autoresearch-rl run-one \
  --config examples/deberta-prompt-injection/config.yaml \
  --diff path/to/change.patch
```

## Deploy (Basilica)

```bash
export BASILICA_API_TOKEN="your-token"
python3 examples/deberta-prompt-injection/deploy.py

# LLM-only param search
python3 examples/deberta-prompt-injection/deploy.py --policy llm

# Grid search (no LLM required)
python3 examples/deberta-prompt-injection/deploy.py --policy grid
```

## How it works

- **Policy**: `hybrid` — starts with multi-turn LLM hyperparameter proposals
  (`hybrid_param_explore_iters=5`), switches to code diff proposals after
  `hybrid_stall_threshold=3` no-improvement iterations, falls back to params after
  `hybrid_diff_failure_limit=3` consecutive diff failures.
- **Multi-turn**: the LLM's prior reasoning is included in each new API call.
- **Contract**: diffs may only touch `train.py`; `prepare.py` is frozen.
- Metric: `val_bpb = 1 - f1` (lower is better).

## Files

| File | Role |
|------|------|
| `train.py` | Mutable — the LLM modifies this |
| `prepare.py` | Frozen — data loading infrastructure, not modified |
| `program.md` | Task spec provided to the LLM |
| `data/` | Train/val JSONL splits |

## Artifacts

- `artifacts/deberta/results.tsv` — per-iteration scores
- `artifacts/deberta/versions/` — kept iterations
- `artifacts/deberta/checkpoint.json` — resumable state
