# minimal-trainable-target

Minimal deterministic example — no external dependencies, runs in milliseconds.
Demonstrates `llm_diff` code evolution and the full agentic workflow interface.

## Prerequisites

```bash
export CHUTES_API_KEY="your-key"   # required for llm_diff
```

## Run (local)

```bash
bash examples/minimal-trainable-target/run.sh
```

For offline testing without an LLM API key:

```bash
bash examples/minimal-trainable-target/run.sh --override policy.type=grid \
  --override 'policy.params={"learning_rate":[0.0020,0.0026,0.0032],"grad_clip":[0.7,0.85,1.0]}'
```

## Agentic workflow

```bash
# Check experiment state (JSON output — agent-readable)
uv run autoresearch-rl status \
  --config examples/minimal-trainable-target/config.yaml --last 5

# Inject a code diff and run one iteration
uv run autoresearch-rl run-one \
  --config examples/minimal-trainable-target/config.yaml \
  --diff path/to/change.patch

# Inject explicit hyperparameters and run one iteration
uv run autoresearch-rl run-one \
  --config examples/minimal-trainable-target/config.yaml \
  --params '{"learning_rate": 0.0026}'
```

## Deploy (Basilica)

```bash
export BASILICA_API_TOKEN="your-token"
python3 examples/minimal-trainable-target/deploy.py
```

## How it works

- **Policy**: `llm_diff` — the LLM proposes code diffs to `train.py` each iteration.
- **Multi-turn**: the LLM's prior proposals are included in each new call.
- **Keep/discard**: accepted diffs persist; discarded ones are rolled back.
- Each iteration completes in milliseconds — ideal for rapid loop validation.

## Files

| File | Role |
|------|------|
| `train.py` | Mutable — the LLM modifies this |
| `prepare.py` | Frozen — evaluation oracle, not modified |
| `program.md` | Task spec provided to the LLM as context |

## Purpose

Use for:
- End-to-end loop validation without GPU or network
- Testing `status` / `run-one` agentic commands locally
- Rapid iteration on policy and controller logic
