"""Progress reporting protocol — emit_progress() for trial scripts.

Trial scripts (train.py) call emit_progress(...) per step or epoch. The
controller-side ProgressReader (target/progress_reader.py) consumes these
reports to:

1. Provide intra-iteration signal to the LLM policy (Phase 7.2).
2. Drive cooperative cancellation (Phase 2 / IntraIterationGuard).
3. Surface trajectory shape in traces/events.jsonl (Phase 1.4).

The contract is minimal: write one JSON line per report to $AR_PROGRESS_FILE.
When the env var is not set (e.g., running outside autoresearch-rl), the call
is a no-op so existing scripts keep working.

A second env var $AR_CONTROL_FILE is read on each emit; if it exists and
contains {"action": "cancel", ...}, the trial is expected to exit cleanly
with code 42 (the cooperative-cancel exit code).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CANCEL_EXIT_CODE = 42

PROGRESS_ENV = "AR_PROGRESS_FILE"
CONTROL_ENV = "AR_CONTROL_FILE"


@dataclass
class ProgressReport:
    iter: int
    step: int
    step_target: int
    elapsed_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    should_continue: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def emit_progress(
    *,
    step: int,
    step_target: int,
    metrics: dict[str, float] | None = None,
    iter: int | None = None,
    exit_on_cancel: bool = True,
) -> bool:
    """Write one progress report. Returns True if trial should continue.

    No-op when $AR_PROGRESS_FILE is not set. Reads $AR_CONTROL_FILE on each
    call; if a cancel control is found and exit_on_cancel is True, calls
    sys.exit(CANCEL_EXIT_CODE) immediately. Pass exit_on_cancel=False if the
    trial wants to handle cleanup itself.
    """
    progress_path = os.environ.get(PROGRESS_ENV)
    if not progress_path:
        return True

    iter_idx = iter if iter is not None else _infer_iter()
    elapsed = _elapsed_since_start()
    report = ProgressReport(
        iter=iter_idx,
        step=step,
        step_target=step_target,
        elapsed_s=elapsed,
        metrics=dict(metrics or {}),
        should_continue=True,
    )

    try:
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(report.to_json_line() + "\n")
    except OSError:
        # Trial filesystem failures must not crash the trial.
        return True

    control = _read_control()
    if control and control.get("action") == "cancel":
        if exit_on_cancel:
            sys.exit(CANCEL_EXIT_CODE)
        return False
    return True


def _read_control() -> dict | None:
    path = os.environ.get(CONTROL_ENV)
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_START_TIME = time.monotonic()


def _elapsed_since_start() -> float:
    return time.monotonic() - _START_TIME


def _infer_iter() -> int:
    raw = os.environ.get("AR_ITER", "0")
    try:
        return int(raw)
    except ValueError:
        return 0
