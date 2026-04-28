# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions are commit-sha tags rather than semver — this is an internal
research codebase, not a published library.

## [unreleased]

### Added

- **`examples/parallel-cancel-showcase/`** — end-to-end CPU-only demo
  exercising Phase 1+2+3+4+6+7.5 in one ~13 s run. Synthetic loss
  landscape with a known optimum at `lr=3e-3, batch=32`. Shows the
  IntraIterationGuard cancelling 10 of 16 trials with reason
  `forecast_above_best`. (`ef1e99b`)

### Changed

- `docs/ARCHITECTURE.md` rewritten to reflect Phase 1–4 modules,
  cooperative cancellation lifecycle, and Phase 7 LLM-prompt fragments.
  Adds an "Adoption history" section pointing at the seven research
  docs under `docs/research/`. (`e92c429`)

## [2026-04-28] — RLix-adoption arc complete

This window closes the six-phase adoption work that began on 2026-04-27.
All strong-fit + medium-fit-cheap items shipped; Phase 5 was deferred
with documentation. CI green on every commit; 453 pytest pass / ruff
clean / mypy 0 in 65 source files.

### Phase 1 — `emit_progress` protocol + LLM teaching (`64cf392`)

#### Added
- `target/progress.py::emit_progress(step=, step_target=, metrics=)`.
  Writes one JSON line to `$AR_PROGRESS_FILE` per call. No-op when the
  env var is unset. Reads `$AR_CONTROL_FILE`; on cancel, exits with
  code 42.
- `target/progress_reader.py::ProgressReader` — controller-side daemon
  thread tail with thread-safe `drain()`.
- `policy/_prompt_fragments.py` — shared `PROGRESS_PROTOCOL_RULES`,
  `CANCELLATION_CONTEXT_RULES`, `BATCH_DIVERSITY_RULES` plus
  `render_progress_summary` / `render_progress_series` helpers used by
  every LLM policy.
- Engine drains `run_dir/progress.jsonl` into `traces/events.jsonl` as
  `progress` events; attaches `progress_series` (downsampled to ≤20
  points) to each history entry.
- `examples/minimal-trainable-target` updated to call `emit_progress`.

#### Changed
- `policy/llm_diff.py::_SYSTEM_PROMPT` and
  `policy/llm_search.py::_SYSTEM_PROMPT` now teach the agent about the
  progress protocol, cancellation context, and batch diversity.
- `target/command.py::CommandTarget` spawns a `ProgressReader` per iter
  and backfills `outcome.metrics` from the latest report when stdout
  was silent.
- `target/basilica.py` bootstrap exposes `GET /progress`; polling
  adapts to live activity (5 s) vs stalled (20 s).

### Phase 2 — Cooperative cancellation + diff guardrail (`5f1c48c`)

#### Added
- `controller/intra_iteration.py::IntraIterationGuard`. Wraps
  `forecasting.should_early_stop` over the live progress series. Watcher
  thread accumulates a cumulative metric series across drain calls.
  Honors `direction='max'` by negating. Writes
  `{"action": "cancel", "reason": ...}` to the run's control file when
  the forecast says abandon ship.
- `config.IntraIterationCancelConfig` (sub-config of `ControllerConfig`)
  — off by default. Tunables: `min_steps`, `poll_interval_s`,
  `min_reports_before_decide`.
- `sandbox/validator.py::validate_required_calls(pre, post, required)`.
  AST-walks both sources, rejects diffs that drop all calls to any name
  in `required`. Hooked from `controller/diff_executor.py`.
- `policy.PolicyConfig.required_calls`, default `["emit_progress"]`.

#### Changed
- New `decision="cancelled"` distinct from `discard`. Reward to
  `Learnable` policies is `-0.05` (between keep `+1.0` and failed
  `-0.1`).
- `controller/helpers.py::check_failure_rate` no longer counts
  `cancelled` as a failure.
- `target/interface.py::RunOutcome.status` and
  `controller/executor.py::Outcome.status` accept `"cancelled"`.

### Phase 3 — Perfetto/Chrome-trace timeline (`d67a130`)

#### Added
- `telemetry/timeline.py::TimelineRecorder`. Append-only Chrome trace
  events (`ph='X'` complete events with explicit `dur` µs) to a JSON
  array file openable in `chrome://tracing` or `ui.perfetto.dev`.
  Thread-safe; no-op when `telemetry.timeline_path` is unset.
- `set_global` / `global_span` so policies and downstream targets can
  emit spans without threading the recorder through constructors.
- Spans wired: `policy.propose`, `executor.execute` (with terminal
  status + elapsed_s), `basilica.{create_deployment, wait_ready,
  poll_for_metrics, download_model, cleanup}`,
  `llm.chat_completion` (model, attempt count, terminal status).

#### Changed
- `config.TelemetryConfig.timeline_path: str | None = None`.

### Phase 4 — Concurrent / batched iterations (`f0fd4a0`, `fea8090`, `d6993d8`)

#### Added
- `controller/parallel_engine.py::run_experiment_parallel`. Sibling to
  `run_experiment`. `ThreadPoolExecutor` over batched proposals,
  admitted by `ResourcePool`, results processed in submission order via
  two queues (`in_flight` + `completed` buffer).
- `controller/resource_pool.py::ResourcePool`. Threadsafe dict-of-int
  bin-packing. Reservation keyed by `iter_idx`. No partial allocation.
- `policy/interface.py::propose_batch(policy, state, k)` helper +
  Protocol method. Native impls in `RandomPolicy`, `GridPolicy`,
  `StaticPolicy`, `LLMParamPolicy`.
- `LLMParamPolicy.propose_batch` (Phase 7.4): one chat call asking for
  k diverse proposals (LRs ≥4× apart). Strict parser. Falls back to k
  seeded random on parse failure or missing API key.
- `BestValueRef` (in `controller/intra_iteration.py`): thread-safe
  shared float; iters submitted before any best exists still cancel as
  soon as a sibling completes (R3.a follow-up).
- `config.ParallelConfig` (sub-config of `ControllerConfig`) — off by
  default. `enabled`, `max_concurrency`, `resources`,
  `submit_poll_interval_s`.
- `target/interface.py::resource_cost(target, params)` helper. Targets
  may declare a `resource_cost(self, params)` method; defaults to
  `{"gpu": 1}`.
- `comparability.budget_mode='parallel_wallclock'` (R3.b): writes
  per-trial `outcome.elapsed_s` into the ledger `budget_s` column
  instead of loop wall.
- Description column annotated `<label>|conc=K` so post-hoc analysis
  can filter without a schema change.

#### Notes
- Diff-mode policies (`llm_diff`, `hybrid`) intentionally remain serial
  — k concurrent diffs would fight the frozen/mutable contract.
  `controller/continuous.py` enforces this.
- R3.a reward ordering: `pending_rewards: dict[iter_idx -> float]`
  drained in monotonic iter order so `Learnable.record_reward` sees a
  stable trial-time sequence regardless of completion order.

### Phase 6 — Two-phase runtime config validation (`87b8b41`)

#### Added
- `config_validate.py::validate_runtime(cfg)`. Pure function; returns
  ordered `ValidationError` list with severity (`error` | `warn`).
  Eight checks: reserved `AR_*` param keys, file existence (mutable /
  frozen / program), `BASILICA_API_KEY` presence + non-empty
  `gpu_models` for basilica targets, LLM api-key env presence,
  writable parent dirs for `checkpoint_path` and `model_output_dir`,
  budget alignment vs `max_wall_time_s` (warn), and (R3.e) positive
  presence of `emit_progress(...)` calls in the trial source when
  intra-iteration cancel is enabled.
- `cli.py::run` and `cli.py::validate` both invoke it; blocking errors
  exit code 2.

### Phase 7 — LLM/agent orchestration awareness (interleaved)

Shipped alongside the host phases. See individual phase entries above
for `_prompt_fragments.py` (7.1, 7.2), `validate_required_calls` (7.3),
and `LLMParamPolicy.propose_batch` (7.4). Phase 7.5 (update
`examples/*/{train.py, program.md}`) shipped with each new mechanism.

### Type debt — M1, M2, M3 (`467dc63`)

#### Changed
- `target/basilica.py`: new local `_Deployment` and `_DeploymentStatus`
  Protocols capture the SDK surface we use, eliminating `object`-typed
  attribute access. Type-ignore tags on the runtime-only SDK imports.
- `cli.py`: `# type: ignore` on `import yaml` (stubs deferred) and the
  lazy `huggingface_hub` import.
- `sandbox/runner.py`: narrow type fixes to the legacy loop runner
  (Optional `workdir`, tuple-shape narrowing for the forecaster).

mypy went from 10 pre-existing errors to **0 errors in 65 source files**.

### Follow-ups closed

- **#19 Basilica cancel propagation** (`d6577ae`):
  `BasilicaTarget._propagate_control` POSTs `run_dir/control.json`
  contents to the deployment's `/control` endpoint each poll tick.
  Size-cached for idempotence. The bootstrap server already accepted
  POST `/control`; only the controller-side wiring was missing.
- **#24 Per-worker progress paths for parallel cancel** (`9d52c3c`):
  parallel engine no longer writes `os.environ`; CommandTarget derives
  both env vars from `$run_dir` on its per-call env dict. End-to-end
  real-subprocess parallel-cancel test passes deterministically.

### Bugs surfaced and fixed by the showcase example (`3b24976`)

- `comparability.check_comparability` rejected `parallel_wallclock` as
  `unsupported_budget_mode`. Phase 4B added the new mode but the
  gatekeeper still hard-coded `fixed_wallclock`. Every showcase ledger
  row had `comparable=0` until fixed.
- `CommandTarget` passed `AR_PROGRESS_FILE` as a relative path. When
  `target.workdir` differs from the engine's cwd, the subprocess could
  not find the file. `outcome.metrics` was empty for every iter.
- `config_validate._check_required_calls_for_cancel` required
  `policy.mutable_file` even for non-diff policies. Now resolves the
  trial source via `mutable_file` OR by walking `target.train_cmd` for
  a `.py` argument; warns (not errors) if neither resolves.

### Deferred

- **Phase 5 — Multi-LoRA iteration-sharing target**. None of the in-tree
  examples have the search-space shape that benefits. Three concrete
  re-open triggers + a ~120-line implementation sketch are documented
  in `docs/research/RLix-Phase5-Deferred.md`.

## Pre-arc baseline (2026-04-27)

Captured in `docs/research/baseline-2026-04-27.md`:
pytest 358 pass / ruff clean / mypy 10 pre-existing errors.
