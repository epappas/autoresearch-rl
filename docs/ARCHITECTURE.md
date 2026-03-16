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
| `BasilicaConfig` | GPU cloud settings: image, gpu_count, gpu_models, memory, cpu, storage, ttl_seconds, setup_cmd |
| `PolicyConfig` | `type` (grid/random/static/learned/llm) + param space + LLM fields |
| `ControllerConfig` | `seed`, `max_wall_time_s`, `no_improve_limit`, `failure_rate_limit`, `failure_window`, `checkpoint_path` |
| `ComparabilityConfig` | `budget_mode`, `expected_budget_s`, `expected_hardware_fingerprint`, `strict` |
| `ScoringConfig` | Weights for composite score components (val_bpb, loss, fail_penalty, etc.) |
| `TelemetryConfig` | Paths for trace, ledger, artifacts, versions; file rotation limits |

## Core modules

### `cli.py`
CLI entrypoint. Parses `--config`, optional `--override key=value` flags, and sub-commands (`validate`, `print-config`). Builds `RunConfig` and calls `run_continuous`.

### `controller/continuous.py`
Orchestrates the main loop:
1. Initialises policy, telemetry paths, `LocalFileTracker`, `PromotionTracker`, `ComparabilityPolicy`.
2. Optionally restores state from a `LoopCheckpoint`.
3. Each iteration: propose params -> `target.run()` -> `target.eval()` -> keep/discard -> emit trace + ledger row + manifest.
4. Stop guards: wall-time, no-improve streak, failure rate, power-law early stop.
5. On exit: emits `episode_summary` event, saves `LoopCheckpoint` if `checkpoint_path` set.

### `controller/helpers.py`
Small helpers for stop-guard logic (`check_no_improve`, `check_failure_rate`) and `current_commit`.

### `controller/shutdown.py`
`ShutdownHandler`: registers `SIGINT`/`SIGTERM` handlers for graceful shutdown.

### `controller/types.py`
`LoopResult` dataclass returned by `run_continuous`.

## Targets (`target/`)

All targets implement `TargetAdapter` (`target/interface.py`): `run(run_dir, params)` and `eval(run_dir, params)` both returning `RunOutcome`.

`RunOutcome` fields: `status` (ok/failed/timeout), `metrics`, `stdout`, `stderr`, `elapsed_s`, `run_dir`.

**Parameter injection (command + basilica targets):** params are passed via `AR_PARAMS_JSON` (full JSON dict) and individual `AR_PARAM_<NAME>` env vars.

### `target/command.py` -- `CommandTarget`
Runs `train_cmd` / `eval_cmd` as local subprocesses. Parses `key=value` pairs from stdout to extract metrics.

### `target/http.py` -- `HttpTarget`
POSTs params JSON to a remote URL. Expects a JSON response with a `metrics` key.

### `target/basilica.py` -- `BasilicaTarget`
Deploys each training iteration as a containerized GPU job on Basilica cloud:
1. Wraps the user command in a bootstrap Python script that starts an HTTP health-check server (port 8080) then runs the command as a subprocess.
2. Creates a `Deployment` via `basilica-sdk`, waits for ready status.
3. Polls deployment logs every 20 s for metric patterns (`key=value`).
4. Falls back to extracting metrics from raw logs on timeout or failure.
5. Cleans up (deletes) the deployment after each iteration.
6. Optional `setup_cmd` runs inside the container before the training command.

### `target/registry.py`
Builds the correct `TargetAdapter` from a `TargetConfig`.

## Policies (`policy/`)

All policies implement `ParamPolicy` (`policy/search.py`, `policy/interface.py`): `next(history) -> ParamProposal`.

`ParamProposal` fields: `params` dict, `rationale` string.

### `policy/search.py`
- `GridPolicy`: exhaustive cartesian product of the param space, cycled with `itertools.cycle`.
- `RandomPolicy`: uniform random sampling from the space with a fixed seed.
- `StaticPolicy`: returns empty params (no overrides).

### `policy/llm_search.py` -- `LLMParamPolicy`
Calls any OpenAI-compatible `/chat/completions` endpoint to propose hyperparameters. Sends full experiment history (capped at 50 entries) as context. Parses the JSON response and validates all values against the allowed choices with type coercion. Falls back to seeded random on any failure. Zero extra dependencies (stdlib `urllib` only).

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
- `failure_rate_limit` + `failure_window`: fraction of failures in rolling window
- Power-law early stop: forecast-based (>= 5 data points required)
- Graceful shutdown: `SIGINT`/`SIGTERM` via `ShutdownHandler`

## Contract/sandbox (legacy)
The earlier contract/sandbox loop remains in the repo (`controller/loop.py`, `controller/contract.py`, `sandbox/`, `eval/`) but is **not** used by the continuous CLI. If you want to remove or integrate it, do so explicitly.
