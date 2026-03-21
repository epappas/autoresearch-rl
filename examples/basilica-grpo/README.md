# basilica-grpo: Qwen2.5-0.5B on GSM8K

Fine-tune Qwen2.5-0.5B-Instruct with GRPO on GSM8K using multi-turn LLM-guided hyperparameter
search — the LLM builds cumulative reasoning across iterations from the full experiment history.

## Prerequisites

```bash
export BASILICA_API_TOKEN="your-basilica-token"
export CHUTES_API_KEY="your-key"
export HF_TOKEN="your-huggingface-token"
pip install basilica-sdk
```

## Run (Basilica required)

```bash
bash examples/basilica-grpo/run.sh
```

## Agentic workflow

```bash
# Check experiment state (JSON output — agent-readable)
uv run autoresearch-rl status \
  examples/basilica-grpo/config.yaml --last 5

# Inject explicit hyperparameters and run one Basilica iteration
uv run autoresearch-rl run-one \
  examples/basilica-grpo/config.yaml \
  --params '{"learning_rate": 5e-6, "max_steps": 30, "num_generations": 4}'
```

## Deploy (Basilica)

```bash
python3 examples/basilica-grpo/deploy.py

# Custom Docker image
python3 examples/basilica-grpo/deploy.py --image-tag your-registry/grpo:latest

# Grid search (no LLM required)
python3 examples/basilica-grpo/deploy.py --policy grid
```

## How it works

1. The `hybrid` policy starts with LLM-guided hyperparameter search, then switches to code
   diffs when param exploration stalls.
2. For each iteration, a Basilica deployment is created with a GPU container.
3. Inside the container: `prepare.py` runs first (via `prepare_cmd`) to download GSM8K data
   and write formatted JSONL files to `/app/data/`.
4. Then `train.py` reads the prepared data, trains with GRPO, evaluates pass@1, and prints
   metrics to stdout.
5. The controller parses metrics from deployment logs and applies keep/discard.

## Pipeline

```
prepare.py  -->  /app/data/{train,eval}.jsonl  -->  train.py  -->  [metrics]
(runs once)       (frozen data boundary)            (each iter)    (keep/discard)
```

`prepare.py` is a config-driven pipeline step (`prepare_cmd` in config.yaml). It runs once
per container before `train.py`. There is no Python import between them -- they communicate
via JSONL data files on disk.

## Files

| File | Role |
|------|------|
| `train.py` | Mutable -- GRPO training, modified by LLM in diff mode |
| `prepare.py` | Frozen -- pipeline step, produces data files via `prepare_cmd` |
| `program.md` | Task spec provided to the LLM as context |

## GPU Requirements

- 1× A100 or H100 (24GB+ VRAM)
- ~32GB system memory
- ~15–20 minutes per iteration

## Artifacts

- `artifacts/basilica-grpo/results.tsv` — per-iteration scores
- `artifacts/basilica-grpo/versions/` — kept iterations
- `artifacts/basilica-grpo/checkpoint.json` — resumable state
