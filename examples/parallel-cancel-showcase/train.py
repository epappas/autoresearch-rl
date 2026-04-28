"""Synthetic training trial that demonstrates Phase 1-4 features end-to-end.

Behavior is deterministic given (learning_rate, batch_size) drawn from the
search space. Some combinations converge fast and produce a low val_loss;
others sit on a high plateau (those are the ones IntraIterationGuard should
cancel mid-trial when it sees the trajectory cannot beat the running best).

Each step:
  - sleeps a tiny amount to make the trial wall-clock measurable
  - computes a loss based on the param vector and the step
  - emit_progress(step=, step_target=, metrics={'val_loss': ...})

If the controller writes $AR_CONTROL_FILE with action='cancel', the next
emit_progress call exits cleanly with code 42. The engine flips the
outcome status to 'cancelled' (decision='cancelled' in the ledger) — not
a failure.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

# emit_progress is a no-op when AR_PROGRESS_FILE is unset, so this file
# also runs standalone (e.g., `python train.py`) without autoresearch-rl
# being in the loop.
try:
    from autoresearch_rl.target.progress import emit_progress
except ImportError:  # pragma: no cover - example may run outside the SDK
    def emit_progress(**_: object) -> bool:
        return True


N_STEPS = 30
SLEEP_PER_STEP_S = 0.06


def _read_params() -> dict[str, float]:
    raw = os.environ.get("AR_PARAMS_JSON", "{}")
    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        params = {}
    return params


def _verify_prepared() -> None:
    """Refuse to start if prepare.py wasn't run for the campaign.

    prepare.py is the run-once-per-campaign frozen step; it writes the
    shared data file relative to the workdir (the subprocess cwd here).
    """
    sentinel = Path("data") / "ready.json"
    if not sentinel.exists():
        raise SystemExit(
            f"prepare.py was not run ({sentinel.resolve()} missing)"
        )


def _loss_at_step(step: int, learning_rate: float, batch_size: int) -> float:
    """Synthetic optimum at lr=3e-3, batch=32; everything else converges slower.

    Loss curve: exponential decay from ~2.0 toward an asymptote that depends
    on how far (lr, batch) is from the optimum. Far-from-optimum trials
    plateau high — exactly the case IntraIterationGuard should cancel.
    """
    log_lr_dist = math.log10(max(learning_rate, 1e-9)) - math.log10(3e-3)
    bs_dist = math.log2(max(batch_size, 1) / 32.0)
    distance = abs(log_lr_dist) + 0.4 * abs(bs_dist)

    asymptote = 0.20 + 0.55 * distance
    decay = 0.30 + 0.10 * distance  # smaller decay = slower convergence
    progress = 1.0 - math.exp(-decay * step)
    return asymptote + (2.0 - asymptote) * (1.0 - progress)


def main() -> None:
    _verify_prepared()
    params = _read_params()
    learning_rate = float(params.get("learning_rate", 1e-3))
    batch_size = int(params.get("batch_size", 32))

    final_loss = 0.0
    for step in range(1, N_STEPS + 1):
        loss = _loss_at_step(step, learning_rate, batch_size)
        final_loss = loss
        emit_progress(
            step=step,
            step_target=N_STEPS,
            metrics={"val_loss": loss},
        )
        time.sleep(SLEEP_PER_STEP_S)

    # Print final metric in key=value format so even targets that don't
    # consume progress still see the result.
    print(f"val_loss={final_loss:.6f}")


if __name__ == "__main__":
    main()
