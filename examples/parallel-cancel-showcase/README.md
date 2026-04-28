# Parallel + Cancel Showcase

End-to-end demo exercising every Phase 1–4 + 6 feature added during the
RLix-adoption arc. CPU-only synthetic loss landscape, runs in ~15 seconds.

## What this exercises

| Feature | How |
|---|---|
| **Phase 1** — `emit_progress` protocol | `train.py` calls `emit_progress(step, step_target, metrics)` per step; engine drains into `traces/events.jsonl` |
| **Phase 2** — Cooperative cancellation | `controller.intra_iteration_cancel.enabled: true`; doomed trials get `decision="cancelled"` |
| **Phase 3** — Perfetto/Chrome timeline | `telemetry.timeline_path` set; produces `traces/timeline.json` openable in `chrome://tracing` or `ui.perfetto.dev` |
| **Phase 4** — Parallel iterations | `controller.parallel.enabled: true`, `max_concurrency: 4` |
| **Phase 4 / R3.b** — Parallel comparability | `comparability.budget_mode: parallel_wallclock`; ledger records per-trial budget |
| **Phase 6** — Config validation | `_check_required_calls_for_cancel` resolves trial source from `target.train_cmd` and verifies `emit_progress(...)` is present |
| **Phase 7.5** — Example calls `emit_progress` | Trial source acts as the LLM's diff template |

## How to run

From the repo root:

```bash
bash examples/parallel-cancel-showcase/run.sh
```

No GPU needed. Cleans previous run state, executes the campaign, prints
a one-page summary.

## What you should see

A clean run (single GPU box, CPython 3.13) measured 2026-04-28:

```
== Running parallel + cancel showcase (max_concurrency=4) ==
{
  "iterations": 16,
  "best_value": 0.200222,
  "best_score": 0.200222
}

== Ledger summary ==
Total rows:       16
Status breakdown:
     10 cancelled
      3 discard
      3 keep

== Timeline events ==
  Total spans: 41
  executor.execute: 16
  policy.propose_batch: 25

== Cancellations (per-trial control files) ==
  run-0004: {"action": "cancel", "reason": "forecast_above_best"}
  run-0006: {"action": "cancel", "reason": "forecast_above_best"}
  run-0007: {"action": "cancel", "reason": "forecast_above_best"}
  run-0008: {"action": "cancel", "reason": "forecast_above_best"}
  run-0009: {"action": "cancel", "reason": "forecast_above_best"}
  run-0010: {"action": "cancel", "reason": "forecast_above_best"}
  run-0011: {"action": "cancel", "reason": "forecast_above_best"}
  run-0013: {"action": "cancel", "reason": "forecast_above_best"}
  run-0014: {"action": "cancel", "reason": "forecast_above_best"}
  run-0015: {"action": "cancel", "reason": "forecast_above_best"}
```

Wall time: ~13 s for the full 16-iteration campaign (16 trials × ~1.8 s
each = ~29 s sequential ideal; ~13 s observed under parallelism + cancel
+ subprocess overhead).

## Anatomy of the synthetic landscape

`train.py::_loss_at_step` defines a deterministic loss curve parameterised
by `(learning_rate, batch_size)`:
- Optimum at `lr=3e-3, batch=32` → asymptote `val_loss ≈ 0.20`
- Far-from-optimum cells plateau at `val_loss ≈ 2.00` and converge slowly
- Each step takes ~60 ms, total 30 steps per uncancelled trial

The search space (random, seed=42) covers 6 LRs × 4 batch sizes = 24
cells. With `max_iterations: 16`, the campaign samples ~2/3 of the space.

The asymptote spread is what makes cancellation useful: a trial whose
`val_loss` plateaus near 2.0 cannot beat a current best of 0.93 or 0.42,
so the IntraIterationGuard's power-law forecast says "abandon ship" after
5 reports (~0.5 s of trial wall time), and `emit_progress`'s next call
exits with code 42 (cooperative cancel).

## Comparison: serial baseline

```bash
uv run autoresearch-rl run examples/parallel-cancel-showcase/config-serial.yaml
```

Serial `random` policy with cancel disabled. Same search space, same
seed. On the test machine this ran 5 iterations in 24 s before the
existing power-law forecaster (Phase 0) early-stopped the loop because
the metric series suggested the run would not improve. Best val_loss
reached: 0.93.

The parallel + cancel run, by contrast, completed all 16 iterations
in 13 s and found `val_loss = 0.20` (the global optimum). The wins:

- ~50 % less wall clock for 3.2× more samples explored (16 vs 5).
- Found the optimum (0.20) where the serial run never escaped the
  early-stop trap (0.93).
- 10 of 16 trials cancelled mid-flight, saving ~10 × 1.3 s ≈ 13 s of
  wasted compute.

(Caveat: the serial path's early-stop was triggered by the engine's
existing power-law forecaster, not by intra-iteration cancel. The
forecaster is correct in spirit — the early sample of params landed in a
weak region — but the parallel run's wider exploration found better
cells before the same forecast could fire. Different mechanism, similar
intent: don't waste compute on doomed trajectories.)

## Where the artifacts go

```
artifacts/
  results.tsv                       per-iter ledger (16 rows + header)
  runs/
    run-0000/
      progress.jsonl                30 reports (one per emit_progress)
      manifest-*.json               per-iter snapshot
    run-0004/
      progress.jsonl                ~5 reports (cancelled mid-trial)
      control.json                  {"action": "cancel", "reason": "..."}
    ...
  versions/
    v0000/version.json              kept iter metadata
    v0005/version.json
    v0012/version.json
  traces/
    events.jsonl                    per-event timeline (proposals, progress, iterations, summary)
    timeline.json                   Chrome trace JSON — open in chrome://tracing
```

## Reading the timeline

`traces/timeline.json` is Chrome-trace format. Open it directly in:
- `chrome://tracing` (Chrome / Chromium / Edge)
- `https://ui.perfetto.dev`

Each iteration shows:
- `policy.propose_batch` span (k=4 → one prompt to the LLM in real
  campaigns; here a deterministic random draw)
- `executor.execute` per trial (16 of these), with `args.status` and
  `args.elapsed_s` recorded as the span ends

Cancelled trials show as shorter `executor.execute` spans because the
trial subprocess exits early when `emit_progress(...)` reads the
`control.json` file.

## Determinism guarantee

The showcase ships two configs that exercise **two distinct determinism
contracts**, both verified by `tests/test_showcase_determinism.py`.

### Strict — `config-deterministic.yaml` (cancel disabled)

Two runs with the same seed produce **bit-identical** per-iter params,
keep / discard decisions, kept version directories, and `best_value`.
The parallel engine itself is deterministic when cancel timing
(filesystem-polling jitter) is removed.

### Relaxed — `config.yaml` (cancel enabled)

Two runs produce identical **params** and identical **`best_value`**.
The cancellation set is allowed to differ because cancel decisions
depend on filesystem-polling timing — a trial whose early-step loss
straddles the current best can be cancelled in one run and survive in
another. The synthetic landscape's `min_steps: 8` /
`min_reports_before_decide: 10` are tuned so the optimum (iter 12) is
reliably kept; lowering them resurrects the flake.

### Allowed to differ in **both** contracts

- exact `val_loss` recorded for a cancelled trial
- per-event timestamps in `traces/timeline.json` (real wall-clock)
- `episode_id` (uuid generated at run start)
- the exact ordering of timeline spans across worker threads
- `elapsed_s` per iter

## What this does NOT exercise

- **LLM-driven policies** (`llm`, `llm_diff`, `hybrid`). Use `random`
  here so the demo is offline. Same wiring; flip `policy.type` and set
  `OPENAI_API_KEY` (or your provider's equivalent) to swap in.
- **Basilica deployments**. CPU-only; the cooperative-cancel path is
  identical, but `_propagate_control` only fires when the target is
  `basilica`. See `examples/basilica-grpo` for the cloud path.
- **Diff-mode policies** under parallelism. Phase 4 deliberately
  serializes diff iters because k concurrent code edits fight the
  frozen/mutable contract. Hybrid mode automatically falls back to
  serial.
- **Multi-LoRA target sharing**. Deferred — see
  [`docs/research/RLix-Phase5-Deferred.md`](../../docs/research/RLix-Phase5-Deferred.md).

## Files

- `prepare.py` — frozen run-once data prep step
- `train.py` — mutable trial; emits progress; can be diffed by an LLM
- `program.md` — task guidance the LLM reads
- `config.yaml` — parallel + cancel + timeline
- `config-serial.yaml` — serial baseline for comparison
- `run.sh` — wrapper that cleans state, runs, prints summary
