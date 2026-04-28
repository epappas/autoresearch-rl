# Implementation Plan — Adopt RLix-Inspired Improvements

Companion to [`RLix-Comparison.md`](./RLix-Comparison.md). Concrete, file-grounded, ordered by dependency.

## Scope

Adopts the four **strong-fit** items and the two **medium-fit** items. Does **not** adopt Ray, the 7-tier priority enum, or any ROLL/Megatron coupling.

| # | Feature | Source inspiration | Fit |
|---|---|---|---|
| F1 | Structured `ProgressReport` protocol target → controller | RLix `ProgressReport.step_target_trajectories` | strong |
| F2 | Cooperative cancellation of in-flight iterations | RLix `resize_infer` + cooperative shrink | strong |
| F3 | Concurrent / batched iterations on shared compute | RLix gap-ratio planner across pipelines | strong |
| F4 | Perfetto / Chrome-trace timeline export | RLix `SchedulerTracer` | strong |
| F5 | Multi-LoRA "iteration sharing" target | RLix `RollMultiLoraPipeline` | medium |
| F6 | Two-phase fail-fast config validation | RLix scheduler validate-then-mutate | medium |

## Guiding constraints (non-negotiable)

- **No new heavy runtime deps** — no Ray, no asyncio in user-facing surface. Use stdlib `threading` / `concurrent.futures` only. (Matches the existing `target/basilica.py` and `sandbox/runner.py` style.)
- **The frozen/mutable contract stays sacred.** All proposals are still attributed to a single iteration; concurrency is at the *trial* level, not within `prepare.py`.
- **Backwards-compatible YAML.** Every new field defaults off. Existing configs must run unchanged.
- **Each step is independently shippable** behind a config flag.

---

## Phase 0 — Pre-work (housekeeping, blocking)

> **Status (2026-04-27)**: 0.1 done — `target/` restored from HEAD. Imports green. Baseline captured in [`baseline-2026-04-27.md`](./baseline-2026-04-27.md). See [`RLix-Adoption-Remediation.md`](./RLix-Adoption-Remediation.md) for the corrected file/line references applied below.

### 0.1 Restore the deleted `target/` modules

`git status` shows `src/autoresearch_rl/target/{__init__,basilica,command,http,interface,registry}.py` as deleted in the working tree but the imports are still live (`cli.py:17`, `controller/continuous.py:22`, `controller/executor.py:15`). Either restore from HEAD or finish the in-progress refactor before any of the work below — otherwise the codebase doesn't import.

**Action**: confirm with the user whether to `git restore src/autoresearch_rl/target/` or commit the refactor that supersedes them. **Block here until resolved.**

### 0.2 Pick a metric protocol envelope

All later phases depend on a stable `ProgressReport` schema. Decide once. Proposed minimal schema (JSONL or HTTP-GET response):

```json
{
  "type": "progress",
  "iter": 12,
  "step": 340,
  "step_target": 1000,
  "elapsed_s": 87.3,
  "metrics": {"loss": 1.42, "eval_score": 0.81},
  "should_continue": true
}
```

`should_continue` is the controller's signal back to the trial (F2).

---

## Phase 1 — F1: Structured `ProgressReport` protocol

Today metrics are extracted from stdout via `parse_metrics()` and Basilica logs via regex polling at 20 s intervals (`target/basilica.py:_poll_for_metrics` (verified: 15 s readiness wait + 20 s metrics poll)). This is lossy and slow.

### 1.1 Define the protocol (new file)

- **New**: `src/autoresearch_rl/target/progress.py` with a `ProgressReport` dataclass + `ProgressEmitter` helper for trial scripts (`emit_progress(step, step_target, metrics, **)` → writes one JSON line to `$AR_PROGRESS_FILE` and/or POSTs to `$AR_PROGRESS_URL`).
- **New**: `src/autoresearch_rl/target/progress_reader.py` — controller-side `ProgressReader` that tails the file or polls the HTTP endpoint, returns latest `ProgressReport | None`.

### 1.2 Wire into `CommandTarget` (file-based)

- `target/command.py`: pre-create a tempfile, set `AR_PROGRESS_FILE=<path>` in the subprocess env. Spawn a `ProgressReader` thread alongside the existing stdout reader.
- Drain remaining reports after the subprocess exits.

### 1.3 Wire into `BasilicaTarget` (HTTP-based)

- Extend the bootstrap script in `target/basilica.py:_BOOTSTRAP` to expose `GET /progress` on the existing health-check server (port 8080). Trivial: the trial writes JSON lines to a file the bootstrap reads.
- Replace `_poll_for_metrics`'s 20 s blind sleep with an adaptive loop: if `/progress` returns `step/step_target` evidence of progress, drop the poll interval to 5 s; if stalled, back off to 60 s.

### 1.4 Surface in telemetry

- Emit one `progress` event per report into `traces/events.jsonl` via `telemetry.events.emit` (already supports arbitrary dicts).
- **No** ledger row per report — only on iteration completion as today.
- Hook point: extract the per-iteration body of `controller/engine.py:run_experiment` (lines ~192–356) into `_run_one_iteration(...)`; emit the `progress` event from inside the watcher thread that the executor spawns. This refactor also unblocks Phase 4 (parallel engine reuses `_run_one_iteration`).

### 1.5 Tests

- `tests/test_progress.py`: `ProgressEmitter` writes valid JSONL; `ProgressReader` returns the most recent line; bootstrap `/progress` round-trip via local HTTP server.

**Acceptance**: an example trial in `examples/minimal-trainable-target/train.py` calls `emit_progress(...)` per epoch, and `traces/events.jsonl` contains the resulting `progress` events.

---

## Phase 2 — F2: Cooperative cancellation

Builds on F1. Today `forecasting.should_early_stop` (`controller/engine.py:369`) only fires *between* iterations. With F1 we can cancel the *current* one.

### 2.1 Add a cancel signal channel

- Extend `ProgressEmitter` to also *read* a sibling control file `$AR_CONTROL_FILE`. The trial polls it after each `emit_progress(...)` call and exits cleanly if it contains `{"action": "cancel", "reason": "..."}`.
- Standardize exit code `42` for cooperative cancel, distinct from failure (`1`) and timeout.

### 2.2 Controller-side intra-iteration forecaster

- New `controller/intra_iteration.py::IntraIterationGuard`. Inputs: rolling list of `ProgressReport.metrics[objective.metric]`, current best score, `min_steps_before_cancel` (default 50), `forecaster=should_early_stop`. Returns `("continue" | "cancel", reason)`.
- Reuse the existing `forecasting.should_early_stop` — it is series-agnostic.

### 2.3 Wire into the executor

- Extend `controller/executor.py::Executor` Protocol with `execute_with_cancel(proposal, run_dir, guard) -> Outcome`. Default no-op `guard`.
- Implement on `controller/executor.py::TargetExecutor` and `controller/diff_executor.py::{DiffExecutor, HybridExecutor}` (the canonical diff path, not `sandbox/runner.py`).
- `TargetExecutor`: spawn a watcher thread that polls `ProgressReader` every N s, calls `guard.evaluate(...)`, and on `"cancel"` writes the control file. Uses `threading.Event` for shutdown — same pattern as `controller/shutdown.py`.
- `BasilicaTarget`: cancel via `POST /control` on the bootstrap server (extend the same handler that exposes `/progress`).

### 2.4 Treat cancellation as a first-class outcome

- Add `status="cancelled"` to `target/interface.py::RunOutcome` *and* `controller/executor.py::Outcome`. The engine treats it like `"failed"` for keep/discard but distinguishes it in telemetry and ledger (new `decision="cancelled"`). `controller/types.py::LoopResult` and the per-iteration `history` entry surface it for the LLM policies.
- Negative reward for `Learnable` policies: e.g. `-0.05` (less than `-0.1` for crashes — a cancel was a graceful early-out, not a bug).

### 2.5 Config

```yaml
controller:
  intra_iteration_cancel:
    enabled: false
    min_steps: 50
    forecaster: power_law       # extension point
    poll_interval_s: 10
```

### 2.6 Tests

- Synthetic trial that emits a degrading metric → controller cancels at step `min_steps + 1`.
- Trial that finishes before any cancel signal → no false cancel.

**Acceptance**: replay the `basilica-grpo` example with a deliberately bad seed and confirm it cancels mid-training instead of running to wall-clock.

---

## Phase 3 — F4: Perfetto-format scheduling timeline

Independent of F2/F3 — do this early because it's the diagnostic surface you want when F3 lands.

### 3.1 Adapter only — do not adopt `tg4perfetto`

RLix uses a forked tracing lib (`tg4perfetto`). We can write the Perfetto trace JSON ourselves; the format is documented and small.

- **New**: `src/autoresearch_rl/telemetry/timeline.py::TimelineRecorder`.
  - `slice(name, category, start_ts, end_ts, args=None)` writes a Chrome trace event: `{"ph":"X", "name":..., "cat":..., "ts":..., "dur":..., "pid":..., "tid":..., "args":...}`.
  - Writes incrementally to `traces/timeline.json` as a JSON array (open `[`, append `,{...}`, close `]` on shutdown — matches Chrome's tolerant parser).
- One process per "thread", one iteration per "slice"; nested slices for sub-phases (proposal, target.run, target.eval, model_download).

### 3.2 Hook into the engine

- `controller/engine.py:run_experiment`: wrap `executor.execute(...)`, the proposal call, and the model-download phase in `with recorder.slice(...)`.
- `target/basilica.py`: emit slices for `create_deployment`, `wait_ready`, `poll_metrics`, `download_model`, `cleanup`.
- `policy/llm_search.py` and `policy/llm_diff.py`: slice the LLM call (we already log latencies — surface them visually too).

### 3.3 Config

```yaml
telemetry:
  timeline_path: traces/timeline.json   # null disables
```

### 3.4 Tests

- `tests/test_timeline.py`: synthetic recorder produces valid Chrome trace JSON (parseable by `json.loads`); slice nesting yields correct durations.

**Acceptance**: open `traces/timeline.json` in `chrome://tracing` and see one row per Basilica deployment with sub-slices for each phase.

---

## Phase 4 — F3: Concurrent / batched iterations

The biggest change. Touches the engine's main loop. Do **not** start until F1, F2, F4 are merged.

### 4.1 Decide the concurrency model

Recommendation: **bounded `ThreadPoolExecutor` of in-flight iterations**, with policy proposing in batches. Reason: `BasilicaTarget` is I/O-bound (it polls a remote API); threads are sufficient. No need for asyncio or Ray.

### 4.2 Batched policy proposals

- Add an optional `propose_batch(state, k) -> list[Proposal]` method to `policy/interface.py::Policy`. Default implementation calls `propose(state)` k times.
- Implement `propose_batch` natively for `RandomPolicy`, `GridPolicy`, and `LLMParamPolicy` (LLM batch: one prompt asking for k diverse proposals; falls back to k single proposals on parse failure).
- **Diff-mode policies stay serial.** A code diff changes the trial source; running k diffs concurrently means k worktrees and serializing the keep/discard ordering, which fights the contract. Defer.

### 4.3 New parallel engine path

- **Prerequisite (R2)**: extract the per-iteration body of `controller/engine.py:run_experiment` (lines ~192–356) into `_run_one_iteration(...)` first. Both serial and parallel engines call it. Cuts duplication.
- New `controller/parallel_engine.py::run_experiment_parallel`. Same shape as `run_experiment` but:
  - Each loop tick: ask the policy for `k = max_concurrency - in_flight` proposals (uses `propose_batch` from §4.2 / Phase 7.4).
  - Submit each to the pool. Each worker writes its own `run-{iter:04d}/` and emits its own progress/timeline slices.
  - As futures complete, run the existing keep/discard logic **in submission order** (use a min-heap on `iter_idx` so the ledger order is deterministic).
  - **Reward ordering (R3.a)**: maintain `pending_rewards: dict[int, float]` keyed by `iter_idx`. After each completion, drain in ascending order so `Learnable.record_reward` sees a stable trial-time sequence.
  - **Comparability (R3.b)**: when `comparability.budget_mode == "parallel_wallclock"`, budget is per-trial `outcome.elapsed_s`, not loop wall time. Record `max_concurrency_at_submission` in the ledger row.
- The engine still owns: best-score tracking, version-saving, `Learnable.record_reward` (in serialized order), checkpoint, comparability check.

### 4.4 Resource accounting (the gap-ratio idea, scaled down)

- Each `TargetAdapter` declares `resource_cost(params) -> dict` (e.g. `{"gpu": 1, "memory_gb": 32}`). Default `{"gpu": 1}`.
- A new `controller/resource_pool.py::ResourcePool` admits a proposal only if its cost fits remaining capacity. Trivial bin-packing — no need for the gap-ratio planner.
- Config:
```yaml
controller:
  parallel:
    enabled: false
    max_concurrency: 4
    resources: {gpu: 4, memory_gb: 128}
```

### 4.5 Cancellation interplay

When F2 cancels a doomed in-flight trial, the freed resource is returned to the pool and the next batch can fill it. This is the entire point of doing F2 first.

### 4.6 Checkpoint/resume

The current checkpoint records a single `iter_idx`. With batches in flight, record `next_iter_to_submit` and a list of `in_flight_iters`. On resume, treat unfinished iters as failed and re-propose.

### 4.7 Tests

- `tests/test_parallel_engine.py`: a fake target with deterministic 0.5 s sleep + scored metrics. Run with `max_concurrency=4`; assert wall time ≈ N/4 × per-iter and that ledger order is deterministic.
- Cancellation interleaving: doomed trial freed early triggers next submission immediately.

**Acceptance**: `random` policy on `examples/minimal-trainable-target` with `max_concurrency=4` completes 20 iterations in ≤ ~1.3× the time of 5 sequential iterations of the same target.

---

## Phase 5 — F5: Multi-LoRA iteration-sharing target (medium-fit, optional)

> **Status (2026-04-28)**: deferred. See [`RLix-Phase5-Deferred.md`](./RLix-Phase5-Deferred.md) for rationale, re-open triggers, and a concrete implementation sketch. The summary below is kept for historical context.

Only implement if a real example needs it (e.g., the security-judge campaign wants to compare 5 reward shaping variants per GPU).

### 5.1 New target type

- **New**: `src/autoresearch_rl/target/multi_lora_basilica.py::MultiLoraBasilicaTarget`.
- One Basilica deployment trains N LoRAs over a shared base model, returns `RunOutcome` whose `metrics` dict contains a per-tag dict `{"_lora_results": {"tag_a": {...}, "tag_b": {...}}}`.

### 5.2 Engine fan-out

- Engine must accept a `MultiOutcome` and unpack into N pseudo-iterations in the ledger (one row per tag, same `episode_id`, distinct `iter_idx`s in a contiguous block, `params["_lora_tag"] = ...`).
- Each tag goes through the existing keep/discard.

### 5.3 Policy support

- A new `LoraBatchPolicy` wrapper: takes a base param policy, on each call returns N proposals tagged `_lora_tag = "lora_0".."lora_{n-1}"`. The trial script uses the tag to select a reward variant or hyperparameter set per LoRA.

**Defer until requested.** This is heavy and only pays off for specific search-space shapes.

---

## Phase 6 — F6: Two-phase config validation

Cheap and high-value. Do anytime.

### 6.1 New module

- **New**: `src/autoresearch_rl/config_validate.py::validate_runtime(cfg: RunConfig) -> list[ValidationError]`.
- Pure function. Returns errors; never mutates.

### 6.2 Checks (mirroring real failure modes seen in the repo)

- `objective.metric` must be one of: a known metric name (heuristic list), or pulled from a sample `prepare.py` parse, or accepted with warning.
- `policy.params` keys must not collide with reserved env vars (`AR_*`).
- `policy.mutable_file` and `policy.frozen_file` must exist on disk.
- For `target.type == "basilica"`: `BASILICA_API_KEY` env var present, `gpu_models` non-empty, `gpu_count >= 1`.
- For `policy.type in {"llm", "llm_diff", "hybrid"}`: `policy.llm_api_key_env` resolves to a non-empty env var.
- `controller.checkpoint_path` parent dir is writable.
- `telemetry.model_output_dir` parent dir is writable when set.
- `comparability.expected_budget_s` must be ≤ `controller.max_wall_time_s` (currently silently mismatched).

### 6.3 CLI integration

- `cli.py::validate` already exists. Replace its body with `validate_runtime(cfg)` and pretty-print errors.
- The default `cli.py::run` calls `validate_runtime(cfg)` first; refuses to start if any error has `severity=error`. `severity=warn` is printed and the run continues.

### 6.4 Tests

- One test per error class (~10 tests). Pure functions, no I/O outside tmp dirs.

**Acceptance**: pasting `examples/basilica-grpo/config.yaml` with `BASILICA_API_KEY` unset prints a clear error and exits non-zero before any deployment.

---

## Phase 7 — LLM/agent orchestration awareness (cross-cutting, mandatory)

**Without this phase, the strong-fit features ship as dead infrastructure.** The LLM policies do not auto-discover `emit_progress`, cooperative cancel, parallel siblings, or `propose_batch`. They must be explicitly taught.

Today's LLM context (from `policy/llm_diff.py::_format_diff_prompt` and `policy/llm_search.py::_format_prompt`, fed by `controller/continuous.py::_make_diff_state_builder`):

```
history (≤50 entries, each: iter, status, decision, metrics, params, stdout_tail, stderr_tail)
program (program.md text)
source (current train.py)
mutable_file path
recent_errors, recent_logs
```

Nothing about progress, cancellation, batch context, or resource budget.

### 7.1 System prompt updates (per policy)

- `policy/llm_diff.py::_SYSTEM_PROMPT`: add a "Progress protocol" section:
  > *"The trial script SHOULD call `from autoresearch_rl.target.progress import emit_progress` and invoke `emit_progress(step=, step_target=, metrics={...})` at least every N steps so the controller can early-cancel doomed trials. If existing calls are present, preserve them. Removing them is a regression."*
- `policy/llm_search.py::_SYSTEM_PROMPT`: add a "Cancellation context" section:
  > *"`status='cancelled'` in history means the controller stopped the trial early because the forecast did not beat the best score. Treat the metrics from cancelled iters as partial signal, not failure."*
- New shared constant `policy/_prompt_fragments.py` so the wording stays consistent across diff, search, and hybrid policies.

### 7.2 Enrich the state dict

Extend `_make_diff_state_builder` and `_param_state_builder` in `controller/continuous.py`:

- Per history entry, add `progress_series: list[{step, metric_value}] | None` — the last N progress reports for that iteration. Lets the LLM see *trajectory shape*, not just the final number ("trial 12 was promising at step 100, plateaued at 300").
- Add a top-level `cancellation_summary: {total: int, last_reason: str | None}` so the LLM knows when its diffs keep hitting the same early-stop cliff.
- Add `resource_budget: {gpu, memory_gb, max_concurrency}` when F3 is enabled — useful so the LLM doesn't propose `gradient_accumulation_steps=64` on a 32 GB card.

### 7.3 Diff guardrails for load-bearing calls

Add a new function `validate_required_calls(pre_source, post_source, required: list[str])` to `sandbox/validator.py` (the same module that `controller/diff_executor.py:110` already calls). Reuses `sandbox/ast_policy.py::_dotted_name` and the existing `ast.walk` pattern.

- Parse the *post-patch* AST; if the pre-patch source contained `emit_progress(` calls and the post-patch source contains zero, reject with a clear correction message: *"Your diff removes all `emit_progress` calls. These are required for early-stop. Restore at least one call inside the training loop."*
- Same for any function listed in a new `policy.required_calls: list[str]` config field. Lets users mark e.g. `save_checkpoint(` as load-bearing per project.
- Hook point: in `controller/diff_executor.py::DiffExecutor.execute`, after the existing `validate_diff(diff)` call (line 110), also call `validate_required_calls(source, modified, ...)`.
- **Positive-presence check (R3.e)** — `config_validate.py` (Phase 6) refuses to start when `controller.intra_iteration_cancel.enabled` is true and the mutable_file source contains zero `emit_progress(` calls. This catches the blind spot where the source never had progress reporting in the first place.

### 7.4 Native batch proposals (for F3)

Don't issue k independent LLM calls — issue one and parse k.

- New `LLMParamPolicy.propose_batch(state, k)`:
  - One chat call with system instruction *"Return a JSON array of exactly k diverse proposals. Diversity dimensions: learning rate (1+ decade apart), batch size, sampling strategy. Do not duplicate prior history entries."*
  - Parse as JSON array; on parse failure, fall back to k seeded-random proposals (existing fallback path).
  - Cheaper (one prompt, one cache hit) and demonstrably more diverse than k independent calls.
- `LLMDiffPolicy.propose_batch` is intentionally **not implemented** — see plan §4.2 (k concurrent diffs fight the contract).

### 7.5 Update `program.md` template + examples

- New section in every `examples/*/program.md`: *"Your training script must emit progress per step using `emit_progress(...)`. The controller may cancel trials whose metric trajectory is forecast not to beat the current best."*
- Update `examples/*/train.py` to actually call `emit_progress(...)`. This matters even more than the docs: the LLM's most reliable cue is *the source it is diffing*. If the source already calls `emit_progress`, new diffs preserve it by default.

### 7.6 Tests

- `tests/test_llm_prompt_progress.py`: assert system prompts mention `emit_progress`; assert state builder includes `progress_series` when reports exist.
- `tests/test_diff_guardrail_progress.py`: feed a diff that strips all `emit_progress` calls → validator rejects with the correction message; LLM gets the message and retries (use a stub `_call_chat_api_messages`).
- `tests/test_llm_batch.py`: stub LLM returns a JSON array of 4 proposals; `propose_batch(state, 4)` returns 4 distinct dicts. Stub returns malformed → falls back to seeded random.

### 7.7 Observability

- Emit a `prompt_context` event into `traces/events.jsonl` per LLM call with `{has_progress_series, history_len, cancelled_count, batch_size}`. Lets us debug "is the agent actually receiving the new context?" without dumping full prompts.

### Acceptance for Phase 7

1. With F1+F7 enabled, `LLMDiffPolicy` produces a diff that *adds* an `emit_progress(...)` call in a `train.py` that didn't have one — verifiable in `traces/events.jsonl`.
2. With F2+F7, after 3 cancelled iterations the LLM's next proposal references the cancellation pattern in its reasoning (parse the assistant message; loose check: contains "cancel" or "early stop").
3. With F3+F7, `propose_batch(state, 4)` returns 4 proposals where no two share the same `learning_rate` value.

### Sequencing

- 7.1, 7.2 ship **with** F1 (Phase 1) — the agent learns about progress at the same moment progress exists.
- 7.3 ships **with** F2 (Phase 2) — guardrail and cancel mechanism land together.
- 7.4 ships **with** F3 (Phase 4) — batch proposals only useful with parallel execution.
- 7.5 (`program.md` + examples) ships in every phase that introduces a new mechanism the LLM must respect.

**Estimate**: ~2 days additional, spread across phases (no separate calendar block).

---

## Sequencing & estimates

**T-shirt sizes**: S = ≤1 day; M = 1–3 days; L = 3–5 days. Every phase ships in PRs each ≤ 1 day. Velocity tracked in `docs/research/velocity.md` after first 3 PRs merge.

```
Phase 0 (housekeeping)           ── S  (DONE 2026-04-27)
  └─ Phase 6 (config validation) ── S, parallelizable
  └─ Phase 1 (ProgressReport)    ── M
       │  + Phase 7.1, 7.2, 7.5  ── teach LLM about progress (S, ships with F1)
       └─ Phase 2 (cancel)       ── M
            + Phase 7.3, 7.5     ── diff guardrail + program.md (S, ships with F2)
  └─ Phase 4 (timeline)          ── S, parallelizable
       └─ Phase 3 (parallel)     ── L
            + Phase 7.4, 7.5     ── batch proposals + diversity (S, ships with F3)
            └─ Phase 5 (LoRA)    ── M, optional, DEFERRED 2026-04-28 (see RLix-Phase5-Deferred.md)
```

**Calibration note (R4.a)**: original day-estimates were gut-feel. Sizes above are bounds, not commitments. Re-estimate after 3 merged PRs.

Phase 7 is **not optional** — without it, the LLM agents won't use the new infrastructure and the investment is wasted.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| F3 breaks deterministic comparability of campaigns | Keep serial path as default; parallel path records `parallelism_at_submission` in ledger so post-hoc analysis can filter |
| F2 cancels trials too aggressively | `min_steps` floor + opt-in flag; default disabled; require `>= 5` progress reports before any cancel decision (mirrors `should_early_stop` `min_points`) |
| F1 trial-side helper adds friction to BYO scripts | `emit_progress(...)` is one import + one call per epoch; no-op when `AR_PROGRESS_FILE` is unset, so existing scripts keep working |
| F4 timeline file grows unbounded on long campaigns | Reuse `telemetry.rotation` with a `timeline_max_bytes` cap |
| F3 concurrent iterations write to the same `versions_dir` | Already iter-indexed (`v{iter:04d}`); add an explicit lock around `_save_version` |
| Touching `controller/engine.py` regresses the existing serial loop | New `parallel_engine.py` is a separate module; engine.py is unchanged unless explicitly opted in |

## Out of scope (do not do as part of this plan)

- Ray actors / Ray cluster — avoid the operational footprint until justified.
- Cross-iteration GPU sharing inside one trial (RLix's actual core feature) — meaningless for our serial-trial model.
- vLLM `sleep_level` integration — we don't own the inference engine.
- Replacing the frozen/mutable contract with anything more elaborate.

---

## Definition of done (per phase)

Each phase ships when:
1. Unit tests pass (`uv run pytest -q`).
2. `uv run ruff check src/ tests/` clean.
3. `uv run mypy src/` clean.
4. At least one example in `examples/` exercises the new path end-to-end with real evidence in `traces/events.jsonl` or `traces/timeline.json`.
5. `docs/ARCHITECTURE.md` updated with the new module and a one-paragraph description.
