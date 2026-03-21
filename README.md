# AutoResearch-RL

Autonomous ML experiment loop. An LLM proposes hyperparameters or code changes,
trains on local or cloud GPU (Basilica), evaluates, keeps or discards, and repeats.

```
prepare.py  -->  [data]  -->  train.py  -->  [metrics]  -->  keep/discard  -->  repeat
 (frozen)                     (mutable)       eval_score       |
                                  ^                            |
                                  |     LLM proposes next      |
                                  +------- params or diff -----+
```

## Install

```bash
uv sync --extra dev
uv sync --extra basilica   # optional: Basilica GPU cloud
```

## Quickstart

```bash
# Local (no GPU, deterministic, milliseconds per iteration)
bash examples/minimal-trainable-target/run.sh

# Basilica GPU cloud (GRPO post-training on GSM8K)
export BASILICA_API_TOKEN="..." CHUTES_API_KEY="..."
python3 examples/basilica-grpo/deploy.py
```

## How it works

**Pipeline.** Each example has a `prepare.py` (frozen) and `train.py` (mutable).
`prepare.py` runs once via `prepare_cmd` and produces data files.
`train.py` runs each iteration, reads the data, trains, and prints `key=value` metrics
to stdout. The controller parses metrics and decides keep or discard.

**Targets.** Where training runs: locally (`command`), against a remote API (`http`),
or on Basilica GPU cloud (`basilica`). Same config, different `target.type`.

**Policies.** How the next experiment is chosen:

| Policy | Description |
|--------|-------------|
| `grid` | Exhaustive combinations |
| `random` | Seeded uniform sampling |
| `llm` | LLM proposes params from experiment history (any OpenAI-compatible API) |
| `llm_diff` | LLM proposes code diffs to `train.py` with correction retry |
| `hybrid` | Starts with param search, switches to code diffs when stalled |
| `learned` | PPO-based policy that learns from trajectory feedback |

**Stop guards.** Wall time, max iterations, no-improvement streak, failure rate.

**Checkpoint/resume.** State persisted after every iteration. Survives crashes and restarts.

## Examples

| Example | Policy | Task |
|---------|--------|------|
| [minimal-trainable-target](examples/minimal-trainable-target/) | `llm_diff` | Deterministic toy (no GPU) |
| [autoresearch-like](examples/autoresearch-like/) | `llm_diff` | Synthetic training loop |
| [basilica-grpo](examples/basilica-grpo/) | `hybrid` | GRPO post-training: Qwen2.5-0.5B on GSM8K |
| [deberta-prompt-injection](examples/deberta-prompt-injection/) | `hybrid` | DeBERTa security classifier |

Each example: `config.yaml`, `prepare.py`, `train.py`, `program.md`, `deploy.py`,
`Dockerfile`, `run.sh`, `README.md`.

## Config

```yaml
target:
  prepare_cmd: ["python3", "prepare.py"]   # runs once, produces data
  train_cmd: ["python3", "train.py"]       # runs each iteration
  type: basilica                           # or: command, http
  basilica:
    gpu_models: ["A100"]
    setup_cmd: "pip install transformers datasets"

policy:
  type: hybrid                             # or: llm, llm_diff, grid, random
  params:
    learning_rate: [3e-6, 5e-6, 1e-5]
  mutable_file: train.py
  frozen_file: prepare.py
  program_file: program.md
  llm_api_url: "https://llm.chutes.ai/v1"
  llm_model: "deepseek-ai/DeepSeek-V3-0324"
  llm_api_key_env: "CHUTES_API_KEY"

objective:
  metric: eval_score
  direction: max

controller:
  checkpoint_path: artifacts/checkpoint.json
  no_improve_limit: 10
```

## CLI

```bash
uv run autoresearch-rl --config config.yaml                    # run the loop
uv run autoresearch-rl validate --config config.yaml           # validate config
uv run autoresearch-rl status --config config.yaml --last 5    # check state (JSON)
uv run autoresearch-rl run-one --config config.yaml \
  --params '{"learning_rate": 5e-6}'                           # single iteration
```

## Output

```
artifacts/results.tsv          # per-iteration scores + comparability metadata
artifacts/versions/v0001/      # kept iterations (versioned artifacts)
artifacts/checkpoint.json      # resumable state
traces/events.jsonl            # structured event trace (proposals, outcomes)
```

## Progress chart

```bash
python scripts/progress_chart.py artifacts/results.tsv -o progress.png --direction max
```

Generates a Karpathy-style scatter plot: gray (discarded), green (kept), step function
(running best).
