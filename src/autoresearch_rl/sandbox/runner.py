from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from autoresearch_rl.sandbox.validator import validate_diff


@dataclass
class TrialResult:
    status: str
    timeout_s: int
    diff_len: int
    elapsed_s: float
    stdout: str = ""
    stderr: str = ""


def run_trial(diff: str, timeout_s: int, command: list[str] | None = None) -> TrialResult:
    """Validate candidate diff then run bounded subprocess command."""
    v = validate_diff(diff)
    if not v.ok:
        return TrialResult(
            status="rejected",
            timeout_s=timeout_s,
            diff_len=len(diff),
            elapsed_s=0.0,
            stderr=v.reason,
        )

    cmd = command or ["bash", "-lc", "echo 'val_bpb=1.234'"]
    start = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        elapsed = time.monotonic() - start
        status = "ok" if p.returncode == 0 else "failed"
        return TrialResult(
            status=status,
            timeout_s=timeout_s,
            diff_len=len(diff),
            elapsed_s=elapsed,
            stdout=p.stdout,
            stderr=p.stderr,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        return TrialResult(
            status="timeout",
            timeout_s=timeout_s,
            diff_len=len(diff),
            elapsed_s=elapsed,
            stdout=e.stdout or "",
            stderr=e.stderr or "",
        )
