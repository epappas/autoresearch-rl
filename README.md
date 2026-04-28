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

Common workflows are wrapped in a `Makefile`:

```bash
make help       # list targets
make check      # lint + typecheck + full tests (~95 s)
make test-fast  # tests excluding the slow integration suite (~30 s)
make showcase   # run examples/parallel-cancel-showcase end-to-end
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

**Stop guards.** Wall time, max iterations, no-improvement streak, failure rate
(`cancelled` iters do not count as failures).

**Checkpoint/resume.** State persisted after every iteration. Survives crashes and restarts.

**Cooperative cancellation** (`controller.intra_iteration_cancel.enabled`). The
trial calls `from autoresearch_rl.target.progress import emit_progress` per step;
the engine drains progress reports and runs them through the power-law forecaster.
When a trial cannot beat the current best, the engine writes a control file and
the trial's next `emit_progress` call exits with code 42. Status becomes
`cancelled` (graceful early-out, distinct from `failed`).

**Parallel iterations** (`controller.parallel.enabled`). `K` trials run
concurrently inside a `ThreadPoolExecutor`, admitted by a resource pool. Diff and
hybrid policies stay serial — k concurrent diffs would fight the contract.
`LLMParamPolicy.propose_batch` issues ONE chat call asking for k diverse
proposals (vs k independent calls). Reward feedback to learnable policies is
buffered and drained in submission order so PPO sees a stable trial-time sequence.

**Timeline export** (`telemetry.timeline_path`). Writes a Chrome-trace JSON file
openable directly in `chrome://tracing` or `ui.perfetto.dev`. Spans:
`policy.propose_batch`, `executor.execute`, `llm.chat_completion`, all
`basilica.*` phases.

**Diff guardrails** (`policy.required_calls`, default `["emit_progress"]`).
The diff validator AST-walks the post-patch source and rejects any diff that
strips a required call. Used to keep load-bearing instrumentation intact across
LLM-proposed code changes.

**Runtime config validation** runs on every `validate` and `run`. Eight checks
covering reserved env-var prefixes, missing files / API keys / GPU models,
unwritable dirs, budget alignment, and positive-presence of `emit_progress` when
intra-iteration cancel is enabled. Blocking errors exit code 2 before any trial
starts.

## Examples

| Example | Policy | Task |
|---------|--------|------|
| [minimal-trainable-target](examples/minimal-trainable-target/) | `llm_diff` | Deterministic toy (no GPU) |
| [parallel-cancel-showcase](examples/parallel-cancel-showcase/) | `random` | End-to-end demo: parallel + cancel + timeline + config validation (no GPU, ~13 s) |
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

  # Optional: cancel doomed trials mid-flight via the power-law forecaster.
  # Trial must call emit_progress(step=, step_target=, metrics=...) per step.
  intra_iteration_cancel:
    enabled: false               # opt-in
    min_steps: 5                 # don't cancel before this many trial steps
    poll_interval_s: 5.0         # how often the guard re-evaluates
    min_reports_before_decide: 5 # need at least this many progress reports

  # Optional: run K iterations concurrently. Diff/hybrid policies stay serial.
  parallel:
    enabled: false               # opt-in
    max_concurrency: 4
    resources: {gpu: 4}          # ResourcePool admits trials by their resource_cost
    submit_poll_interval_s: 0.5

telemetry:
  trace_path: traces/events.jsonl
  ledger_path: artifacts/results.tsv
  artifacts_dir: artifacts/runs
  versions_dir: artifacts/versions
  timeline_path: traces/timeline.json   # null disables; openable in chrome://tracing
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
uv run autoresearch-rl upload config.yaml \
  --repo user/my-security-judge                            # push best model to HF
```

## Output

```
artifacts/results.tsv          # per-iteration scores + comparability metadata
artifacts/versions/v0001/      # kept iterations (versioned artifacts)
  version.json                 # params, metrics, model_dir path
artifacts/checkpoint.json      # resumable state
artifacts/runs/run-XXXX/
  progress.jsonl               # per-step emit_progress(...) reports
  control.json                 # cancel signal (only when guard fired)
  manifest-*.json              # per-iter snapshot
traces/events.jsonl            # structured event trace (proposals, progress, iterations, summary)
traces/timeline.json           # Chrome trace JSON (when telemetry.timeline_path set)
/data/models/v0001/            # trained model checkpoint (if model_output_dir set)
```

**Reading the timeline.** Open `traces/timeline.json` in `chrome://tracing` or
[ui.perfetto.dev](https://ui.perfetto.dev) to see per-iteration spans
(`policy.propose_batch`, `executor.execute`), Basilica deployment phases
(`create_deployment`, `wait_ready`, `poll_for_metrics`, `download_model`,
`cleanup`), and LLM call latencies (`llm.chat_completion` with attempt counts
and terminal status as args).

**Model persistence.** When `model_output_dir` is set in config, the framework injects
`AR_MODEL_DIR` into each iteration. The training script saves the model there. On
Basilica, the bootstrap HTTP server exposes `/model/files` (listing) and
`/model/download/<path>` (file download). The controller downloads the model from the
running container before cleanup. The best model's path is recorded in `version.json`.

After a campaign, push the best model to HuggingFace Hub:
```bash
uv run autoresearch-rl upload config.yaml --repo user/my-model
```

## Progress chart

```bash
python scripts/progress_chart.py artifacts/results.tsv -o progress.png --direction max
```

Generates a Karpathy-style scatter plot: gray (discarded), green (kept), step function
(running best).

## Architecture and design notes

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module-by-module walkthrough.
- [`docs/research/`](docs/research/) — RLix-adoption arc: comparison, plan,
  remediation, deferral notes, velocity log, end-to-end reports.
- [`CHANGELOG.md`](CHANGELOG.md) — phase-by-phase change log.
