# Basilica GRPO: Qwen2.5-0.5B on GSM8K

Fine-tune Qwen2.5-0.5B-Instruct using GRPO (Group Relative Policy Optimization)
on GSM8K math reasoning, deployed on Basilica GPU cloud.

## Prerequisites

```bash
export BASILICA_API_TOKEN="your-basilica-token"
export HF_TOKEN="your-huggingface-token"
pip install basilica-sdk
```

## Quick Start

```bash
# Via CLI
uv run autoresearch-rl --config examples/basilica-grpo/example.yaml

# Or via run script
python3 examples/basilica-grpo/run.py
```

## How It Works

1. The autoresearch-rl loop proposes hyperparameter combinations (learning_rate, batch_size, max_steps)
2. For each iteration, a Basilica deployment is created with a GPU container
3. The container runs `train.py` which:
   - Loads Qwen2.5-0.5B-Instruct
   - Fine-tunes with TRL's GRPOTrainer on GSM8K
   - Evaluates pass@1 on GSM8K test set
   - Prints metrics to stdout
4. The controller parses metrics from deployment logs
5. Keep/discard based on val_bpb improvement (1 - pass@1, lower is better)

## Search Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| learning_rate | 3e-6, 5e-6, 1e-5 | GRPO learning rate |
| batch_size | 4, 8 | Per-device batch size |
| max_steps | 20, 30 | Training steps |

## GPU Requirements

- 1x A100 or H100 (24GB+ VRAM)
- ~32GB system memory
- ~15 minutes per iteration

## Output

Results are tracked in:
- `artifacts/basilica-grpo/results.tsv` -- per-iteration scores
- `artifacts/basilica-grpo/versions/` -- kept iterations
- `traces/basilica-grpo/events.jsonl` -- telemetry
