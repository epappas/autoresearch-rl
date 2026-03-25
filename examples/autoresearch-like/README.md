# autoresearch-like

Code evolution via `llm_diff` — the LLM proposes unified diffs to `train.py` each iteration,
building cumulative reasoning from the full experiment history across turns.

## Prerequisites

```bash
export CHUTES_API_KEY="your-key"   # or any OpenAI-compatible API key
```

## Run (local)

```bash
bash examples/autoresearch-like/run.sh
```

For offline testing without an LLM API key:

```bash
bash examples/autoresearch-like/run.sh --override policy.type=grid \
  --override 'policy.params={"learning_rate":[0.0020,0.0026,0.0032],"grad_clip":[0.7,0.85,1.0]}'
```

## Agentic workflow

The controller exposes `status` and `run-one` for agent-driven loops:

```bash
# Check experiment state (JSON output — agent-readable)
uv run autoresearch-rl status \
  examples/autoresearch-like/config.yaml --last 5

# Inject a pre-computed code diff and run one iteration
uv run autoresearch-rl run-one \
  examples/autoresearch-like/config.yaml \
  --diff path/to/change.patch

# Inject explicit hyperparameters and run one iteration
uv run autoresearch-rl run-one \
  examples/autoresearch-like/config.yaml \
  --params '{"learning_rate": 0.0026, "grad_clip": 0.85}'
```

## Deploy (Basilica)

```bash
export BASILICA_API_TOKEN="your-token"
python3 examples/autoresearch-like/deploy.py

# Custom Docker image
python3 examples/autoresearch-like/deploy.py --image-tag your-registry/ar-like:latest

# Grid search (no LLM required)
python3 examples/autoresearch-like/deploy.py --policy grid
```

## How it works

- **Policy**: `llm_diff` — the LLM receives the current `train.py` source + full iteration
  history + task spec and proposes a unified diff.
- **Multi-turn**: prior proposals are included in each new API call, enabling cumulative reasoning.
- **Keep/discard**: accepted diffs are permanently applied to `train.py`; discarded ones are
  rolled back before the next iteration.
- **Contract**: diffs may only touch `train.py` (mutable); `prepare.py` is frozen.

## Pipeline

```
prepare.py  -->  [data/config on disk]  -->  train.py  -->  [metrics]
(runs once)       (frozen boundary)         (each iter)    (keep/discard)
```

`prepare.py` is a config-driven pipeline step (`prepare_cmd` in config.yaml). It runs once
before the iteration loop and produces data files that `train.py` reads. There is no Python
import between them -- they communicate via the filesystem.

## Files

| File | Role |
|------|------|
| `train.py` | Mutable -- the LLM modifies this each iteration |
| `prepare.py` | Frozen -- pipeline step, runs once via `prepare_cmd` |
| `program.md` | Task spec provided to the LLM as context |

## Model Persistence

When `model_output_dir` is set in config, each iteration receives `$AR_MODEL_DIR` and
can save trained checkpoints there. After a campaign, push to HuggingFace:

```bash
uv run autoresearch-rl upload examples/autoresearch-like/config.yaml --repo user/model
```

## Artifacts

- `artifacts/autoresearch-like/results.tsv` -- per-iteration scores
- `artifacts/autoresearch-like/versions/` -- kept iterations (with model_dir if configured)
- `artifacts/autoresearch-like/checkpoint.json` -- resumable state
