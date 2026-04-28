# Parallel-cancel showcase

## Objective
Minimize `val_loss` over `(learning_rate, batch_size)` on a synthetic
loss landscape with a known optimum at `lr=3e-3, batch_size=32`. The
synthetic landscape is intentionally crafted so most cells in the search
space converge slowly enough that intra-iteration cancellation can fire.

## Mutable file
`train.py` — the trial. The LLM may propose code diffs to it (in
`llm_diff` or `hybrid` mode). See "Progress protocol" below for what
must be preserved.

## Frozen file
`prepare.py` — defines what the dataset looks like. The trial verifies
prepare.py ran before starting; LLM cannot modify this.

## Progress protocol
The trial must call `emit_progress(step=, step_target=, metrics={...})`
at least once per training step. The controller drains these reports
into `traces/events.jsonl` and uses the metric series to decide
whether to cooperatively cancel the trial mid-flight.

If your diff strips all `emit_progress(...)` calls, the validator will
reject it with a correction message. Restore at least one call inside
the training loop.

## Cancellation context
With `controller.intra_iteration_cancel.enabled: true`, trials whose
forecasted final `val_loss` cannot beat the current best will be
cancelled. A `status='cancelled'` history entry is partial signal,
not a failure — read the `progress_series` field on cancelled iters
to understand the trajectory shape.

## Search space hint
The optimum sits near `lr=3e-3, batch=32`. Cells far from this
plateau quickly and should be cancelled by the guard. Cells close
to it converge below the plateau within ~10 steps.
