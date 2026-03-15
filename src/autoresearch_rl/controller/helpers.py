from __future__ import annotations

import subprocess
import time


def current_commit(cwd: str | None = None) -> str:
    """Return short git HEAD hash, or 'local' if unavailable."""
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if cp.returncode == 0:
            return (cp.stdout or "").strip() or "local"
    except Exception:
        pass
    return "local"


def check_wall_time(
    start_ts: float, max_wall_time_s: int | None
) -> bool:
    """Return True if wall-time limit is exceeded."""
    if max_wall_time_s is None:
        return False
    return (time.monotonic() - start_ts) >= max_wall_time_s


def check_no_improve(streak: int, limit: int | None) -> bool:
    """Return True if no-improvement streak hit the limit."""
    if limit is None:
        return False
    return streak >= limit


def check_failure_rate(
    recent_statuses: list[str],
    limit: float | None,
    window: int,
) -> bool:
    """Return True if recent failure rate hit the limit."""
    if limit is None:
        return False
    effective_window = max(1, window)
    if len(recent_statuses) < effective_window:
        return False
    fails = sum(1 for s in recent_statuses if s != "ok")
    return (fails / len(recent_statuses)) >= limit
