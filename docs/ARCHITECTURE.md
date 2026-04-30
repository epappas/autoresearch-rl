# AutoResearch-RL Architecture (continuous CLI)

## Goal
Provide a continuous RL loop that can run against any training target (local command, Docker wrapper, Basilica GPU cloud, or remote HTTP endpoint) with keep/discard decisions, versioned artifacts, and pluggable parameter search policies.

## Runtime path
`autoresearch-rl run` -> `controller/continuous.py` -> `target/*` -> `telemetry/*`

## Config (`config.py`)
All configuration is validated by Pydantic models in `config.py` and assembled into `RunConfig`. Sections:

| Model | Purpose |
|---|---|
| `ObjectiveConfig` | `metric` name + `direction` (min/max) |
| `TargetConfig` | target `type` (command/http/basilica), `train_cmd`, `eval_cmd`, `url`, `timeout_s`, `basilica` sub-config |
| `BasilicaConfig` | GPU cloud settings: image, gpu_count, gpu_models, memory, cpu, storage, ttl_seconds, setup_cmd, `ready_timeout_s` (default 600 — bump for parallel-mode setup contention), `post_trial_sleep_s` (default 90 — bootstrap stays alive this long after trial exits so the controller can pull model files before container shutdown) |
| `PolicyConfig` | `type` (grid/random/static/learned/llm/llm_diff/hybrid) + param space + LLM fields + `mutable_file` / `frozen_file` / `program_file` for diff modes + `required_calls` (default `["emit_progress"]`) for the diff guardrail + hybrid stall thresholds |
| `ControllerConfig` | `seed`, `max_wall_time_s`, `no_improve_limit`, `failure_rate_limit`, `failure_window`, `checkpoint_path`, plus sub-configs `intra_iteration_cancel` (Phase 2) and `parallel` (Phase 4) |
| `IntraIterationCancelConfig` | `enabled`, `min_steps`, `poll_interval_s`, `min_reports_before_decide` — controls cooperative cancellation |
| `ParallelConfig` | `enabled`, `max_concurrency`, `resources` dict, `submit_poll_interval_s` — controls the parallel engine |
| `ComparabilityConfig` | `budget_mode` (`fixed_wallclock` or `parallel_wallclock`), `expected_budget_s`, `expected_hardware_fingerprint`, `strict` |
| `ScoringConfig` | Weights for composite score components (val_bpb, loss, fail_penalty, etc.) |
| `TelemetryConfig` | Paths for trace, ledger, artifacts, versions; `model_output_dir`, `timeline_path` (Chrome-trace JSON), file rotation limits |

Runtime semantic validation lives in `config_validate.py::validate_runtime` (Phase 6). Eight checks (reserved env-var prefixes, file existence, missing API keys, basilica creds, writable dirs, budget alignment, positive-presence of `emit_progress` when intra-iteration cancel is enabled). `cli.py::run` and `cli.py::validate` both invoke it; blocking errors exit code 2.

## Core modules

### `cli.py`
CLI entrypoint. Parses `--config`, optional `--override key=value` flags, and sub-commands (`validate`, `print-config`). Builds `RunConfig` and calls `run_continuous`.

### `controller/continuous.py`
Thin dispatcher built on `controller/engine.py::run_experiment` and (when `controller.parallel.enabled`) `controller/parallel_engine.py::run_experiment_parallel`. Routes between four modes based on `policy.type`: param-only (uses `TargetExecutor`), diff-only (uses `DiffExecutor` from `controller/diff_executor.py`), hybrid (uses `HybridExecutor`), and parallel param-search (parallel engine). Diff and hybrid modes are intentionally serial — k concurrent diffs fight the frozen/mutable contract.

### `controller/engine.py`
Where the canonical serial loop body lives. Orchestrates each iteration:
1. `propose` (timed via `policy.propose` span) -> set per-iter `AR_PROGRESS_FILE`/`AR_CONTROL_FILE` env vars -> optionally start an `IntraIterationGuard` -> `executor.execute(proposal, run_dir)` (timed via `executor.execute` span).
2. If the guard cancelled mid-run, override `outcome.status = "cancelled"`.
3. Drain `run_dir/progress.jsonl` into `traces/events.jsonl` as `progress` events (Phase 1.4).
4. Score, keep/discard, save version, append to ledger + manifest, attach `progress_series` (downsampled to 20 points) to the history entry.
5. Stop guards: wall-time, no-improve streak, failure rate (excludes `cancelled`), power-law early stop.
6. On exit: emit `episode_summary` event, close `TimelineRecorder`, save `LoopCheckpoint`.

### `controller/parallel_engine.py`
Sibling to `engine.py`. Same observable contract but multiple iterations execute concurrently inside a `ThreadPoolExecutor`, admitted by a `ResourcePool`, with results processed in submission order via two queues (`in_flight` for executing, `completed` for done-but-awaiting-in-order processing). Specifics:
- `propose_batch(policy, state, k)` is called per submission window; falls back to k single proposals when the policy has no native batch impl.
- `BestValueRef` is a thread-safe shared float that workers' guards consult on every evaluate; iters submitted before any best exists still cancel as soon as a sibling's keep updates the ref.
- `pending_rewards: dict[iter_idx -> float]` is drained in monotonic iter order so `Learnable.record_reward` sees a stable trial-time sequence (R3.a).
- Under `comparability.budget_mode=parallel_wallclock`, the ledger's `budget_s` carries per-trial elapsed_s instead of loop wall (R3.b). Description column is annotated `<label>|conc=K`.

### `controller/intra_iteration.py`
`IntraIterationGuard` wraps `forecasting.should_early_stop` applied to a live progress series. Watcher thread drains the `ProgressReader`, accumulates a cumulative metric series across ticks (drain clears the reader buffer), and writes the cancel control file when the forecast says abandon ship. Honors `direction=max` by negating. `BestValueRef` (in same module) lets workers read the engine's live best value.

### `controller/diff_executor.py`
`DiffExecutor.execute` validates a `DiffProposal` (token guard + AST policy via `sandbox.validator.validate_diff` -> ContractConfig file-bounds -> Phase 7.3 `validate_required_calls` for `emit_progress` and any other `policy.required_calls`), applies the diff via an in-memory ephemeral git repo, writes the modified source to the mutable file, runs the target, then restores the original (the `on_keep` callback persists the diff if accepted). `HybridExecutor` dispatches `ParamProposal -> TargetExecutor`, `DiffProposal -> DiffExecutor`.

### `controller/executor.py`
`Executor` Protocol (`execute(proposal, run_dir) -> Outcome`). Implementations: `TargetExecutor` (param mode), `DiffExecutor` and `HybridExecutor` in `diff_executor.py`. `Outcome` adds `status="cancelled"` and `judge_signals` to `RunOutcome`.

### `controller/resource_pool.py`
`ResourcePool` for the parallel engine. Threadsafe dict-of-int capacity (e.g. `{"gpu": 4, "memory_gb": 128}`) with `try_acquire(iter_idx, cost)` / `release(iter_idx)` / `wait_for_capacity`. No partial allocation, no fairness layer. Inspired by RLix's `ResourceManager` but stripped to autoresearch-rl's needs.

### `controller/helpers.py`
Stop-guard logic (`check_no_improve`, `check_failure_rate` — excludes `cancelled`) and `current_commit`.

### `controller/shutdown.py`
`ShutdownHandler`: `SIGINT`/`SIGTERM` graceful shutdown.

### `controller/types.py`
`LoopResult` dataclass returned by both engines.

## Targets (`target/`)

All targets implement `TargetAdapter` (`target/interface.py`): `run(run_dir, params)` and `eval(run_dir, params)` both returning `RunOutcome`.

`RunOutcome` fields: `status` (ok/failed/timeout), `metrics`, `stdout`, `stderr`, `elapsed_s`, `run_dir`.

**Parameter injection (command + basilica targets):** params are passed via `AR_PARAMS_JSON` (full JSON dict) and individual `AR_PARAM_<NAME>` env vars. The engine additionally sets `AR_PROGRESS_FILE` and `AR_CONTROL_FILE` to per-iter paths (absolutized so the subprocess `cwd` change to `target.workdir` does not break resolution).

`resource_cost(target, params) -> dict[str, int]` (in `target/interface.py`): targets may declare a `resource_cost(self, params)` method consumed by the parallel engine's `ResourcePool`. Default `{"gpu": 1}`.

### `target/progress.py` (Phase 1)
`emit_progress(step=, step_target=, metrics={...})`: trial-side helper that writes one JSON line to `$AR_PROGRESS_FILE`. No-op when the env var is unset, so existing scripts keep working. Reads `$AR_CONTROL_FILE` on each call; on `{"action": "cancel"}` exits with code 42 (cooperative cancel). `ProgressReport` dataclass.

### `target/progress_reader.py` (Phase 1)
`ProgressReader`: controller-side daemon thread that tails the trial's JSONL. Thread-safe `drain()` returns + clears the buffer; `latest()` is a non-clearing peek. Used by `CommandTarget` (metric backfill) and by `IntraIterationGuard` (live series for forecaster).

### `target/command.py` -- `CommandTarget`
Runs `train_cmd` / `eval_cmd` as local subprocesses with per-call env (no `os.environ` writes). Spawns a `ProgressReader` against `$run_dir/progress.jsonl`. Parses `key=value` pairs from stdout and backfills `outcome.metrics` from the latest progress report when stdout was silent. Honors engine-set `AR_PROGRESS_FILE`/`AR_CONTROL_FILE`; falls back to `$run_dir/{progress.jsonl,control.json}` when unset.

### `target/http.py` -- `HttpTarget`
POSTs params JSON to a remote URL. Expects a JSON response with a `metrics` key.

### `target/basilica.py` -- `BasilicaTarget`
Deploys each training iteration as a containerized GPU job on Basilica cloud:
1. Wraps the user command in a bootstrap Python script (built via `string.Template` so JSON literals stay literal) that starts an HTTP server on port 8080 exposing: `GET /` (health), `GET /progress` (returns the JSONL trial wrote inside the container), `POST /control` (writes the cancel file the trial reads via `emit_progress`), `GET /model/files`, `GET /model/download/<path>`.
2. Creates a `Deployment` via `basilica-sdk` (typed via a local `_Deployment` Protocol so mypy works without the SDK installed), waits for ready status.
3. `_poll_for_metrics` adapts: 5 s when `/progress` shows live activity, backs off to 20 s when stalled. Pulls `/progress` snapshots into local `run_dir/progress.jsonl`. Calls `_propagate_control` each tick to upload `run_dir/control.json` to the deployment's `POST /control` (size-cached for idempotence) — this is how Phase 2 cooperative cancel reaches a remote container.
4. Falls back to extracting metrics from raw logs on timeout or failure.
5. Downloads the trained model to `run_dir/model/` via the `/model/files` + `/model/download/<path>` endpoints before cleanup.
6. Cleans up (deletes) the deployment after each iteration.
7. Optional `setup_cmd` runs inside the container before the training command.

### `target/registry.py`
Builds the correct `TargetAdapter` from a `TargetConfig`.

## Policies (`policy/`)

All policies implement the `Policy` Protocol in `policy/interface.py`: `propose(state) -> Proposal`. Optional `propose_batch(state, k) -> list[Proposal]` (Phase 4A); the module-level `propose_batch(policy, state, k)` helper prefers the native impl when present, falls back to k `propose()` calls otherwise.

`Proposal` is a base dataclass with `rationale`. Subtypes: `ParamProposal` (params dict) and `DiffProposal` (unified diff string).

`Learnable` (runtime-checkable) Protocol: `record_reward(reward)`. Engine calls it after each keep/discard. Under parallelism, rewards are buffered and drained in submission order (R3.a) so the policy sees a stable trial-time sequence regardless of completion order.

### `policy/search.py`
- `GridPolicy`: exhaustive cartesian product, cycled with `itertools.cycle`. Native `propose_batch` advances through cells.
- `RandomPolicy`: seeded uniform random. Native `propose_batch` uses the same RNG so a serial run of K iters and `propose_batch(K)` produce bit-identical sequences.
- `StaticPolicy`: empty params. Native `propose_batch` returns k empties.

### `policy/llm_search.py` -- `LLMParamPolicy`
Calls any OpenAI-compatible `/chat/completions` endpoint to propose hyperparameters. Sends full experiment history (capped at 50 entries via `_MAX_HISTORY`) as context. Parses the JSON response and validates all values against the allowed choices with type coercion. Falls back to seeded random on any failure. Zero extra dependencies (stdlib `urllib` only).

`propose_batch(state, k)` (Phase 7.4): issues ONE chat call asking for a JSON array of exactly k diverse proposals (vs k independent calls). Strict parser `_parse_batch_response` rejects wrong count, wrong shape, or values outside the search space. Falls back to k seeded-random draws on parse failure or missing API key. The HTTP call is wrapped in a `llm.chat_completion` timeline span recording model, msg count, attempt count, and terminal status (`ok`/`http_429`/`network_error`/`max_retries`).

### `policy/llm_diff.py` -- `LLMDiffPolicy`
Multi-turn LLM conversation that proposes unified diffs against the mutable file. Sends current source, history, program.md, recent errors, recent logs. On validation failure (Phase 7.3 `validate_required_calls` rejects diffs that strip `emit_progress`), sends a correction request and retries up to `_MAX_CORRECTION_RETRIES` (2). Falls back to greedy on consecutive failures. **No `propose_batch`** — k concurrent diffs would fight the frozen/mutable contract (`controller/continuous.py` enforces this by routing diff/hybrid modes to the serial engine even when `controller.parallel.enabled`).

### `policy/_prompt_fragments.py` (Phase 7.1, 7.2)
Centralized prompt fragments shared by LLM policies: `PROGRESS_PROTOCOL_RULES`, `CANCELLATION_CONTEXT_RULES`, `BATCH_DIVERSITY_RULES`. Helpers `render_progress_summary(history)` and `render_progress_series(history, metric)` produce the user-prompt sections that surface `progress_series` and `cancellation_summary` to the LLM.

### `policy/llm_context.py`
Helpers for shrinking long history before it hits the prompt token budget (`summarize_history`, `extract_recent_errors`, `extract_recent_logs`).

### `policy/learned_search.py` -- `LearnedParamPolicy`
PPO-based policy that learns from iteration feedback:
- State features: last 8 metric scores, consecutive-keep streak, recent fail count, normalised history length.
- Action space: cartesian product of param space (all combinations).
- Calls `PPOAgent.get_action_and_value` to pick an action, then stores a `_Transition` pending reward.
- `record_reward(reward)` is called by the controller after each keep/discard decision.
- Triggers a PPO update every `update_every` (default 8) transitions via GAE + clipped surrogate loss.
- Optionally regularises against a teacher snapshot via sDPO KL penalty.
- Saves policy snapshots to `artifacts/policy_snapshots/` every `snapshot_every` updates.

### `policy/ppo.py` -- `PPOAgent` / `MLP`
Pure-numpy actor-critic. `MLP` is a simple feedforward net with ReLU activations and He initialisation. `PPOAgent.update` runs clipped PPO over multiple epochs using finite-difference SGD. Includes `compute_novelty_bonus` (k-NN distance in state space) used to augment rewards.

### `policy/gae.py`
`compute_gae`: Generalized Advantage Estimation (GAE-lambda). `compute_returns`: R_t = A_t + V(s_t).

### `policy/sdpo.py`
`compute_kl_divergence`: KL(teacher || student). `compute_sdpo_loss`: L_RL + alpha * KL. `compute_adaptive_alpha`: adapts alpha based on reward ratio.

### `policy/baselines.py` / `policy/learned.py`
Legacy diff-proposal policies used by the contract/sandbox loop (not used by continuous CLI).

## MDP primitives (`mdp.py`, `trajectory.py`)

### `mdp.py`
Frozen dataclasses: `State` (code_hash, history, metrics, resource_budget, iteration), `Action` (params or diff + rationale), `Reward` (scalar + component breakdown). `build_state` / `compute_reward` helpers.

### `trajectory.py`
`Transition` dataclass (state, action, reward, next_state, log_prob, value_estimate). `TrajectoryBuffer`: fixed-size circular buffer for storing transitions with `get_batch` / `get_episode` accessors.

## Checkpoint / snapshots (`checkpoint.py`)

- `save_checkpoint` / `load_checkpoint`: atomic JSON serialisation of `LoopCheckpoint` (episode state, best score, history, etc.). Used by continuous controller for resumable runs.
- `save_policy_snapshot` / `load_policy_snapshot` / `get_latest_snapshot_version`: versioned JSON snapshots of `PPOAgent` weights, used by `LearnedParamPolicy` for teacher regularisation.

## Promotion (`promotion.py`)

`PromotionTracker`: tracks consecutive improvements and detects degradation. `should_promote` triggers after `promotion_threshold` (default 3) consecutive keeps. `should_rollback` triggers when all recent scores are >10% worse than the all-time best. Instantiated by the controller but promotion decisions are not yet wired to policy switching.

## Forecasting (`forecasting.py`)

Power-law early stop: fits y = a * x^b + c to the score history and calls `should_early_stop` when >= 5 data points exist. If the predicted final value exceeds the current best, the loop terminates early.

## Tracking (`tracking.py`)

`LocalFileTracker` implements `ExperimentTracker` protocol. Writes to `base_dir/run_id/`: `params.json`, `metrics.jsonl`, `artifacts/`, `status.json`. Used by the controller to record per-run metadata.

## Telemetry (`telemetry/`)

### `events.py`
`emit`: appends JSONL events to the trace file. Triggers `rotate_if_needed` before each write.

### `ledger.py`
`ensure_results_tsv` / `append_result_row`: TSV results ledger with comparability metadata (hardware fingerprint, budget mode, comparable flag).

### `manifest.py`
`new_run_id` / `write_manifest`: per-iteration manifest JSON files in `artifacts_dir`.

### `run.py`
`write_run_manifest`: writes a per-run manifest with git info, platform, python version, hardware fingerprint, and full config at loop start.

### `comparability.py`
`hardware_fingerprint`: stable hash of CPU/GPU/memory. `ComparabilityPolicy` / `check_comparability`: strict mode raises if hardware or budget does not match the expected fingerprint.

### `aggregation.py`
`compute_episode_stats` / `compute_rolling_stats`: mean, median, min, max, stdev, trend slope over score history. Stats are emitted as an `episode_summary` event at loop end.

### `rotation.py`
`rotate_if_needed`: rotates a file when it exceeds `max_size_bytes`, keeping up to `max_rotated` numbered copies.

### `timeline.py` (Phase 3)
`TimelineRecorder` writes Chrome trace events (`ph='X'` complete events with explicit `dur` in microseconds) to a JSON array file openable directly in `chrome://tracing` or `ui.perfetto.dev`. Append-only, thread-safe, no-op when `telemetry.timeline_path` is unset. `set_global` / `global_span` exposes the engine-owned recorder to free-function callers (LLM policies, Basilica target). Engine instantiates one per run, closes on the loop's finally block. Spans currently emitted: `policy.propose`, `policy.propose_batch`, `executor.execute` (with `args.status` / `args.elapsed_s`), `llm.chat_completion`, `basilica.{create_deployment, wait_ready, poll_for_metrics, download_model, cleanup}`.

### `distill.py`
Distillation sample collection (used by legacy loop only).

## Distillation (`distillation/`)

Three modules implementing supervised distillation from snapshots:

- `trainer.py`: distillation training loop.
- `sdft.py`: supervised DFT (distillation fine-tuning) module.
- `sink.py`: directional sink for distillation outputs.

These are not invoked by the continuous CLI.

## Keep/discard + versioning
Iterations that beat the current best score (normalised: lower is always better internally, regardless of `direction`) are **kept**. A `version.json` record is written to `artifacts/versions/v####/` with params, metrics, and status. Discarded iterations still emit trace events and ledger rows.

## Stop guards
- `max_wall_time_s`: wall-clock budget
- `no_improve_limit`: consecutive non-improving iterations
- `failure_rate_limit` + `failure_window`: fraction of failures in rolling window (`cancelled` does **not** count as a failure)
- Power-law early stop: forecast-based (>= 5 data points required)
- Graceful shutdown: `SIGINT`/`SIGTERM` via `ShutdownHandler`

## Cooperative cancellation (Phase 2)
- Engine sets per-iter `AR_PROGRESS_FILE` and `AR_CONTROL_FILE` (absolute paths) before `executor.execute`.
- When `controller.intra_iteration_cancel.enabled`, an `IntraIterationGuard` runs per worker, accumulates the metric series from `ProgressReader`, and writes `{"action": "cancel", "reason": "..."}` to the control file when the power-law forecaster says the trial cannot beat the current best.
- The trial's next `emit_progress(...)` reads the control file and exits with code 42. The engine flips the outcome to `status="cancelled"`, `decision="cancelled"`, reward `-0.05` (between keep `+1.0` and failed `-0.1`).
- For Basilica deployments, `BasilicaTarget._propagate_control` POSTs the local control file to the running container's `/control` endpoint each poll tick (size-cached to prevent spam).
- Under parallelism, `BestValueRef` (thread-safe shared float in `controller/intra_iteration.py`) lets workers' guards read the engine's live best value, so a worker submitted before any best exists still cancels as soon as a sibling completes.

## Sandbox / diff validation
- `sandbox/validator.py::validate_diff`: token guard + AST safety check on diff additions.
- `sandbox/validator.py::validate_required_calls(pre, post, required)` (Phase 7.3): rejects diffs that remove all calls to any name in `required` (default `["emit_progress"]`). Reuses `sandbox/ast_policy.py::_dotted_name`. Hooked from `controller/diff_executor.py::DiffExecutor.execute` after the existing `validate_diff` call.
- `sandbox/ast_policy.py`: forbidden-import / forbidden-call deny-list applied to diff additions.

## Contract/sandbox loop (legacy)
The earlier contract/sandbox **loop** (`controller/loop.py`, `controller/contract.py` is shared, `eval/`) remains in the repo but is **not** used by the continuous CLI. If you want to remove or integrate it, do so explicitly. Note: `sandbox/validator.py` and `sandbox/ast_policy.py` ARE on the canonical diff path and must not be removed with the legacy loop.

## Adoption history
Six-phase RLix adoption arc shipped 2026-04-27 to 2026-04-28. See:
- [`docs/research/RLix-Comparison.md`](research/RLix-Comparison.md) — original RLix vs autoresearch-rl analysis
- [`docs/research/RLix-Adoption-Plan.md`](research/RLix-Adoption-Plan.md) — phase-by-phase plan
- [`docs/research/RLix-Adoption-Remediation.md`](research/RLix-Adoption-Remediation.md) — gaps surfaced after the plan was drafted
- [`docs/research/RLix-Phase5-Deferred.md`](research/RLix-Phase5-Deferred.md) — multi-LoRA target deferral with re-open triggers
- [`docs/research/velocity.md`](research/velocity.md) — measured per-phase wall time
- [`docs/research/concurrency-spike.md`](research/concurrency-spike.md) — threads-vs-asyncio decision for the parallel engine
- [`docs/research/baseline-2026-04-27.md`](research/baseline-2026-04-27.md) — pre-adoption test/lint/type baseline

End-to-end demo: [`examples/parallel-cancel-showcase/`](../examples/parallel-cancel-showcase/) — exercises Phase 1+2+3+4+6+7.5 in one ~13 s CPU-only run.
