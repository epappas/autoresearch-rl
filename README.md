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

## Quickstart

```bash
uv sync --extra dev
uv run autoresearch-rl run examples/minimal-trainable-target/config.yaml
```

## The Two Scripts

Every experiment has two scripts connected by the filesystem, never by imports:

**`prepare.py`** (frozen) -- runs once via `prepare_cmd`. Produces data files, defines
the evaluation protocol (answer extraction, reward computation). The LLM cannot modify
this file. It is the trust boundary: evaluation integrity is guaranteed by freezing it.

**`train.py`** (mutable) -- runs each iteration. Reads the prepared data, trains the
model, prints metrics to stdout. The LLM can modify this file in `llm_diff` or `hybrid`
mode. This is where the training algorithm, reward function, optimizer, and generation
strategy live. When hyperparameter tuning stalls, the LLM proposes code diffs to
`train.py` -- improving the reward function, adding gradient accumulation, changing the
sampling strategy -- autonomously.

The boundary is deliberate: `prepare.py` owns "what is correct" (data, evaluation),
`train.py` owns "how to get there" (training algorithm, reward shaping). The LLM can
evolve the "how" but never redefine the "what".

## How it works

**Targets.** Where training runs: locally (`command`), against a remote API (`http`),
or on Basilica GPU cloud (`basilica`). Same config, different `target.type`.

**Policies.** How the next experiment is chosen:

| Policy | What it proposes | When to use |
|--------|-----------------|-------------|
| `grid` | Exhaustive param combinations | Small spaces, baselines |
| `random` | Uniform random params | Large spaces, baselines |
| `llm` | LLM-guided params from history | Medium spaces, fast convergence |
| `llm_diff` | Code diffs to `train.py` | Algorithmic improvements |
| `hybrid` | Params first, code diffs when stalled | Best of both worlds |
| `learned` | PPO-based policy with trajectory feedback | Long campaigns |

**Hybrid mode** is the most powerful: it starts with param exploration (find the right
learning rate and batch size), then when the no-improvement streak hits `stall_threshold`,
it switches to code diffs. The LLM reads `train.py`, `program.md` (task guidance), and
the full experiment history, then proposes a unified diff. If the diff fails validation,
the error is sent back for correction (up to 2 retries). If diff proposals fail
consecutively, it falls back to param mode.

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
  prepare_cmd: ["python3", "prepare.py"]   # frozen: runs once, produces data
  train_cmd: ["python3", "train.py"]       # mutable: runs each iteration
  type: basilica                           # or: command, http

policy:
  type: hybrid                             # param search -> code diffs on stall
  params:                                  # search space for param mode
    learning_rate: [3e-6, 5e-6, 1e-5]
  mutable_file: train.py                   # LLM can modify this in diff mode
  frozen_file: prepare.py                  # LLM cannot modify this
  program_file: program.md                 # task guidance for the LLM
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
uv run autoresearch-rl run config.yaml                     # run the loop
uv run autoresearch-rl validate config.yaml                # validate config
uv run autoresearch-rl status config.yaml --last 5         # check state (JSON)
uv run autoresearch-rl run-one config.yaml \
  --params '{"learning_rate": 5e-6}'                       # single iteration
uv run autoresearch-rl run-one config.yaml \
  --diff reward_improvement.patch                          # apply a code diff
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
