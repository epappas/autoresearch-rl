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
  --config examples/basilica-grpo/config.yaml --last 5

# Inject explicit hyperparameters and run one Basilica iteration
uv run autoresearch-rl run-one \
  --config examples/basilica-grpo/config.yaml \
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

1. The `llm` policy proposes hyperparameters using multi-turn conversation, incorporating the
   full iteration history into each API call.
2. For each iteration, a Basilica deployment is created with a GPU container.
3. The container runs `train.py` which loads Qwen2.5-0.5B, trains with GRPO on GSM8K,
   evaluates pass@1, and prints metrics to stdout.
4. The controller parses metrics from deployment logs and applies keep/discard.
5. `program.md` provides the task spec to the LLM as persistent context.

## Files

| File | Role |
|------|------|
| `train.py` | GRPO training script |
| `prepare.py` | Frozen — dataset loading and prompt formatting |
| `program.md` | Task spec provided to the LLM |

## GPU Requirements

- 1× A100 or H100 (24GB+ VRAM)
- ~32GB system memory
- ~15–20 minutes per iteration

## Artifacts

- `artifacts/basilica-grpo/results.tsv` — per-iteration scores
- `artifacts/basilica-grpo/versions/` — kept iterations
- `artifacts/basilica-grpo/checkpoint.json` — resumable state
