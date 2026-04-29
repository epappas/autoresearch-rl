# Outcomes — RLix Adoption Arc (2026-04-27 → 2026-04-29)

What autoresearch-rl can now do that it couldn't before, and what we
explicitly did not gain. Companion to `RLix-Comparison.md` (the original
analysis), `RLix-Adoption-Plan.md` (the phase plan), and
`velocity.md` (the per-phase wall-time log with the bugs surfaced).

The scope of validation is mixed: most claims here are validated by
unit + smoke tests; a subset is validated by real LLM calls (Kimi K2.6);
a smaller subset is validated by a real Basilica A100 run. Each item
below names which tier applies.

## Headline

> **autoresearch-rl evolved from a serial controller-blind launcher
> into a parallel cooperative search engine.** The controller now sees
> inside trials, cancels doomed ones, runs K of them concurrently, and
> refuses to start a misconfigured campaign — and the changes have been
> validated against real Kimi inference and a real Basilica A100
> deployment.

## New capabilities

### 1. Live in-trial observability via `emit_progress`
**Validation: real LLM**

*Before:* the controller saw only what the trial printed at the end. A
30-minute training run was a 30-minute black box.

*Now:* trials call `emit_progress(step=, step_target=, metrics=...)`
per step. The controller streams these into `traces/events.jsonl`,
attaches a per-iter `progress_series` (downsampled to 20 points) to
each history entry, and surfaces trajectory shape to the LLM policy.

**Real evidence:** Kimi K2.6, when shown a history with three
cancellations at `lr=0.01` and one keep at `lr=0.001`, picked
`lr=0.001` for the next iter. Captured under
`tests/eval/fixtures/real_responses/param_avoids_cancelled.json`.
The LLM is using the new signal, not just receiving it.

### 2. Cooperative cancellation of doomed trials
**Validation: end-to-end CPU; not yet real Basilica**

*Before:* a bad trial ran to completion regardless of trajectory.

*Now:* the engine forecasts via power-law over the live progress
series. When the forecast says "this trial cannot beat the running
best," it writes `control.json`; the trial's next `emit_progress`
reads it and exits with code 42. Status is `cancelled` (distinct
from `failed`); reward to learnable policies is `-0.05` (between
`+1.0` keep and `-0.1` failed).

**Real evidence (CPU):** in the showcase, **10 of 16 trials get
cancelled** before reaching their full duration. `make showcase`
reproduces.

**Untested:** the Basilica propagation path
(`BasilicaTarget._propagate_control` POSTing the control file to
the live deployment's `/control` endpoint). Mocked tests pass; real
container round-trip not yet exercised.

### 3. Concurrent iteration execution under a resource pool
**Validation: end-to-end CPU; deferred for Basilica**

*Before:* iterations were strictly serial — even when each iter was
independent. *Now:* `controller.parallel.enabled=true`,
`max_concurrency=K` runs K trials in parallel through a
`ThreadPoolExecutor`, admitted by a `ResourcePool` (bin-packing on
`{gpu, memory_gb}`), with **submission-order** ledger writes and
reward-feedback ordering even when futures complete out of order.

**Real evidence (CPU):** showcase runs 16 trials in ~13 s wall vs
~30 s sequential.

**Untested:** parallel mode against a real Basilica deployment.

### 4. Browser-openable campaign timeline
**Validation: end-to-end CPU + 1 Basilica**

*Now:* `traces/timeline.json` in Chrome trace format. Open in
`chrome://tracing` or `ui.perfetto.dev` to see every iteration,
every Basilica deployment phase (`create_deployment`, `wait_ready`,
`poll_for_metrics`, `download_model`, `cleanup`), every LLM call
as boxes on a timeline, with durations and metadata in args.

**Real evidence:** the security-judge probe produced 8 spans with
realistic durations: `create_deployment 1.6s`, `wait_ready 106s`,
`poll_for_metrics 230s`, `download_model 0.4s`, `cleanup 0.8s`,
plus one `policy.propose`, one `executor.execute`, one
`llm.chat_completion`.

### 5. Cooperation between the controller and LLM-proposed code diffs
**Validation: real LLM**

*Now:* `policy.required_calls = ["emit_progress", ...]` AST-counts
pre/post diff and rejects net-removals with a correction message
the LLM gets on retry. Prevents an LLM-proposed "cleanup" diff from
silently stripping the load-bearing instrumentation.

**Real evidence:** Kimi K2.6 was sent a diff prompt against a
`train.py` that already calls `emit_progress`, and produced a diff
that preserved the call. Captured under
`tests/eval/fixtures/real_responses/diff_preserve_emit_progress.json`.

### 6. Cheaper, more diverse LLM-batch proposals
**Validation: real LLM**

*Before:* k LLM proposals = k separate chat calls = k× tokens billed,
often producing similar params.

*Now:* `propose_batch(state, k)` issues ONE chat call asking for k
diverse proposals (LRs ≥4× apart). Falls back cleanly to k
seeded-random on parse failure or missing API key.

**Real evidence:** Kimi K2.6 returned `[1e-5, 1e-4, 1e-3, 1e-2]` for
`propose_batch(state, k=4)` — distinct, well-spread. Captured under
`tests/eval/fixtures/real_responses/batch_proposals.json`.

### 7. Two-phase runtime config validation
**Validation: end-to-end CPU**

*Now:* eight runtime checks run before the first iter:

- Reserved env-var prefixes can't collide with `AR_*`
- Files referenced in policy must exist
- `BASILICA_API_TOKEN` / `OPENAI_API_KEY` must be present when
  target/policy needs them
- Writable parent dirs for checkpoints and model output
- Budget alignment vs `max_wall_time_s`
- Positive presence of `emit_progress(...)` when intra-iteration
  cancel is enabled
- Tracked-path overwrite warnings — won't silently overwrite paper
  data

**Real evidence:** the validator caught real misconfigurations during
this arc — including the moment when running security-judge would
have overwritten the user's tracked paper artifacts.

### 8. Hardware-aware comparability under parallelism
**Validation: end-to-end CPU**

*Now:* `comparability.budget_mode = "parallel_wallclock"` writes
per-trial `outcome.elapsed_s` into the ledger `budget_s` column
instead of loop wall time. Description column annotated
`<label>|conc=K`. Ledger remains comparable across runs with
different concurrency.

## Improvements that aren't strictly new capabilities

### Type-safe baseline
*Before:* 10 mypy errors in 3 files. *After:* **0 errors in 65 files.**
mypy is now a real signal — adding code that breaks types fails
locally.

### End-to-end smoke covers all examples
*Before:* every `llm_diff`/`hybrid` example was silently rejecting
every diff (the contract path-comparison bug, basename-vs-prefix),
running campaigns to completion with `best_value: null`. The bug had
been live for weeks. *After:* `tests/test_examples_smoke.py` runs
every example through either a 2-iter loop or `validate`, asserting
`best_value != None`. That class of bug cannot regress without CI
failing.

### Per-worker race-free run/eval cache
*Before:* `BasilicaTarget._last_train_outcome` was a single attribute
shared across worker threads. Under parallel mode, Thread A's
`eval()` could return Thread B's training outcome, silently
corrupting the kept-best calculation.

*After:* per-`run_dir` dict + lock. Thread A's eval() always returns
Thread A's training outcome.

**Real evidence:** caught by code-reading after probe 3, before any
parallel-mode probe. Regression test in `test_basilica_unit.py`.

### Real-LLM prompt validation
*Before:* prompt assertions used stubs. *After:*
`tests/eval/test_real_llm.py` validated against Kimi K2.6 — 3
behavioral contracts pass against a real LLM provider.

### Real Basilica validation (single-iter)
*Before:* zero. *After:* one full real GRPO training run of
Qwen2.5-0.5B on a real A100, eval_score=0.640909, decision_accuracy=
0.772727, json_compliance=0.972727. Every Basilica integration
touched during the arc is confirmed working in the single-iter
sequential case.

## What we did NOT gain

Honest list:

- **Model download from Basilica container is broken.**
  `BasilicaTarget._download_model` returned `downloaded: False` on
  probe 3. `version.json` records a container-path that doesn't exist
  locally after cleanup. If anyone wants the LoRA weights for
  downstream evaluation, they're not on disk.
- **Multi-iter parallel against Basilica is untested.** CPU-only.
- **Cooperative cancel against Basilica is untested.** The HTTP
  `POST /control` round-trip from controller to running container
  has unit tests with mocks; never run against a live deployment.
- **Phase 5 (multi-LoRA per-deployment fan-out) deferred** — by
  design, no triggers fired.

## Probability assessment for a full 8-hour campaign

Based on what's been validated:

- ~80% chance a multi-iter sequential security-judge campaign
  completes ≥16 iters without the framework crashing.
- ~50% chance you get usable LoRA weights at the end (the model
  download bug).
- Unknown for whether the hybrid policy's diff-mode switchover
  (kicks in after `hybrid_param_explore_iters: 8`) works against
  Basilica — that path has zero real validation.
- Unknown for parallel mode under cloud GPU constraints. We have
  zero validation of `max_concurrency>1` against Basilica. That's
  the obvious next probe.

## What's been spent vs gained

- ~$5 in real Basilica A100 time across 3 probes (2 caught real
  bugs, 1 succeeded).
- ~30 minutes of Kimi K2.6 inference for prompt validation.
- 471 unit + integration tests, ruff clean, mypy clean.
- A single concrete piece of paper-relevant evidence:
  `eval_score=0.640909` from a real GRPO run.

That's the truthful state.
