# AutoResearch-RL

Continuous autoresearch RL runner for long-lived model training loops.

## Goals
- Always-on continuous runs (with safety stop guards)
- Modular targets (local command, HTTP, Basilica GPU cloud)
- Pluggable search policies (grid, random, LLM-guided, PPO-learned)
- Keep/discard decisions with versioned artifacts
- Trace + ledger output for auditability

## Layout (core)
- `src/autoresearch_rl/cli.py` -- CLI entrypoint
- `src/autoresearch_rl/controller/continuous.py` -- continuous loop
- `src/autoresearch_rl/target/` -- target adapters (command/http/basilica)
- `src/autoresearch_rl/policy/` -- parameter search policies (grid/random/llm/learned)
- `src/autoresearch_rl/telemetry/` -- trace + ledger

## Install
```bash
uv sync --extra dev

# Optional: Basilica GPU cloud support
uv sync --extra basilica
```

## Quickstart
```bash
# Local run with grid search
autoresearch-rl --config examples/deberta-prompt-injection/example.yaml

# Local run with LLM-guided search (requires API key)
CHUTES_API_KEY="..." autoresearch-rl --config examples/deberta-prompt-injection/example-llm.yaml

# Basilica GPU cloud with LLM-guided search
CHUTES_API_KEY="..." autoresearch-rl --config examples/deberta-prompt-injection/basilica-llm.yaml
```

## Targets

### Command target (local/Docker)
```yaml
target:
  type: command
  train_cmd: ["python3", "train.py"]
  eval_cmd: ["python3", "eval.py"]
```

**Parameter injection:** passes params via environment variables:
- `AR_PARAMS_JSON` (full dict)
- `AR_PARAM_<NAME>` (uppercased keys)

### HTTP target (remote/vLLM/sglang)
```yaml
target:
  type: http
  url: "http://localhost:8080/train"
  headers:
    Authorization: "Bearer ..."
```

### Basilica target (GPU cloud)
```yaml
target:
  type: basilica
  train_cmd: ["python3", "/app/train.py"]
  timeout_s: 900
  basilica:
    image: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
    gpu_count: 1
    gpu_models: ["A100", "H100"]
    memory: "32Gi"
    setup_cmd: "pip install transformers datasets"
```

Deploys each training iteration as a containerized GPU job. Includes automatic health-check bootstrapping, log polling, and cleanup.

## Policies

### Grid search (`type: grid`)
Cycles through all combinations of the param space exhaustively.
```yaml
policy:
  type: grid
  params:
    learning_rate: [0.00001, 0.00002, 0.00003]
    epochs: [1, 2, 3]
```

### Random search (`type: random`)
Uniform random sampling from the param space with a fixed seed.
```yaml
policy:
  type: random
  seed: 42
  params:
    learning_rate: [0.00001, 0.00002, 0.00003]
    epochs: [1, 2, 3]
```

### LLM-guided search (`type: llm`)
Calls any OpenAI-compatible chat completions API to propose hyperparameters based on full experiment history. Falls back to seeded random on any failure (missing key, timeout, parse error). Zero extra dependencies (stdlib `urllib` only).

```yaml
policy:
  type: llm
  params:
    learning_rate: [0.00001, 0.00002, 0.00003]
    epochs: [1, 2, 3]
    weight_decay: [0.0, 0.01, 0.1]
  llm_api_url: "https://llm.chutes.ai/v1"
  llm_model: "MiniMaxAI/MiniMax-M2.5-TEE"
  llm_api_key_env: "CHUTES_API_KEY"
  llm_timeout_s: 30
```

Works with any provider: Chutes, OpenAI, Anthropic, local vLLM, etc. The API key is read from the environment variable named in `llm_api_key_env`.

### Learned search (`type: learned`)
PPO-based policy that learns from trajectory feedback using GAE and optional sDPO distillation.

### Static (`type: static`)
No param overrides. Useful for single-config baseline runs.

## Output
Each iteration emits:
- `traces/events.jsonl` -- JSONL trace of proposals, outcomes, episode summaries
- `artifacts/results.tsv` -- ledger with metrics, comparability metadata, hardware fingerprint
- `artifacts/runs/` -- per-iteration stdout/stderr + manifest
- `artifacts/versions/` -- versioned artifacts for `keep` decisions only

## Examples

### DeBERTa prompt-injection classifier
Fine-tunes `protectai/deberta-v3-base-prompt-injection-v2` on a small binary classification dataset. Four config variants:

| Config | Target | Policy | Search space |
|--------|--------|--------|--------------|
| `example.yaml` | local | grid | 108 combos |
| `example-llm.yaml` | local | llm (Chutes) | 108 combos |
| `basilica.yaml` | Basilica GPU | grid | 108 combos |
| `basilica-llm.yaml` | Basilica GPU | llm (Chutes) | 108 combos |

See `examples/deberta-prompt-injection/` for configs and training script.

## CLI helpers

```bash
autoresearch-rl validate --config configs/example.yaml
autoresearch-rl print-config --config configs/example.yaml
autoresearch-rl --config configs/example.yaml --override controller.max_wall_time_s=10
```
