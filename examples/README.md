# Examples

Ready-to-run examples for the **continuous CLI**. Each example follows a standard layout:

| File | Purpose |
|------|---------|
| `config.yaml` | Single YAML config |
| `run.sh` | Local entrypoint (calls `uv run autoresearch-rl`) |
| `deploy.py` | Basilica cloud deployment (build, push, run) |
| `Dockerfile` | Container image for Basilica |
| `train.py` | Training script |
| `program.md` | LLM agent instructions (if applicable) |
| `README.md` | Documentation |

Policy and target variations are handled via `--override` flags.

## Minimal trainable target
Deterministic toy target, no GPU, completes in seconds.
```bash
bash examples/minimal-trainable-target/run.sh
```

## Autoresearch-like
Time-bounded deterministic training loop.
```bash
bash examples/autoresearch-like/run.sh
```

## Basilica GRPO
Qwen2.5-0.5B GRPO fine-tuning on GSM8K via Basilica GPU cloud.
```bash
export BASILICA_API_TOKEN="your-token"
bash examples/basilica-grpo/run.sh
```

## DeBERTa prompt injection
DeBERTa classifier for prompt-injection detection. Supports LLM-guided and grid search, local and Basilica.
```bash
# Local (default: LLM-guided)
bash examples/deberta-prompt-injection/run.sh

# Local with grid search
bash examples/deberta-prompt-injection/run.sh --override policy.type=grid

# Basilica cloud
export BASILICA_API_TOKEN="your-token"
python3 examples/deberta-prompt-injection/deploy.py
```
