from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.eval.metrics import parse_metric_series, parse_metrics
from autoresearch_rl.sandbox.diff_utils import extract_touched_files_from_diff
from autoresearch_rl.sandbox.validator import validate_diff


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    check_every_s: float = 5.0
    min_runtime_s: float = 20.0
    # For minimization metrics (lower is better)
    val_bpb_threshold: float | None = None
    loss_threshold: float | None = None
    # Forecasting (power-law fit)
    forecast_enabled: bool = False
    forecast_min_points: int = 3
    forecast_t_max_s: float | None = None
    forecast_metric: str = "val_bpb"


@dataclass
class TrialResult:
    status: str
    timeout_s: int
    diff_len: int
    elapsed_s: float
    stdout: str = ""
    stderr: str = ""
    patch_applied: bool = False


def _run_git(cwd: str, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", cwd, *args],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_git_repo(workdir: str) -> tuple[bool, str]:
    wd = Path(workdir)
    if not wd.exists() or not wd.is_dir():
        return False, f"workdir does not exist: {workdir}"

    if (wd / ".git").exists():
        return True, ""

    init = _run_git(workdir, "init")
    if init.returncode != 0:
        return False, f"git init failed: {init.stderr.strip() or init.stdout.strip()}"

    _run_git(workdir, "config", "user.name", "AutoResearch Scaffold")
    _run_git(workdir, "config", "user.email", "scaffold@local")
    _run_git(workdir, "add", "-A")
    commit = _run_git(workdir, "commit", "-m", "scaffold baseline", "--allow-empty")
    if commit.returncode != 0:
        return False, f"initial git commit failed: {commit.stderr.strip() or commit.stdout.strip()}"

    return True, ""


def _apply_patch_with_git(diff: str, workdir: str, auto_init_git: bool = True) -> tuple[bool, str]:
    wd = Path(workdir)
    if not wd.exists() or not wd.is_dir():
        return False, f"workdir does not exist: {workdir}"

    if auto_init_git:
        ok, reason = _ensure_git_repo(workdir)
        if not ok:
            return False, reason

    check = _run_git(workdir, "apply", "--check", "-", input_text=diff)
    if check.returncode != 0:
        return False, f"git apply --check failed: {check.stderr.strip() or check.stdout.strip()}"

    apply = _run_git(workdir, "apply", "-", input_text=diff)
    if apply.returncode != 0:
        return False, f"git apply failed: {apply.stderr.strip() or apply.stdout.strip()}"

    return True, ""


def _rollback_patch_with_git(workdir: str, touched_files: list[str]) -> None:
    # Best-effort rollback scoped ONLY to files touched by the candidate diff.
    if not touched_files:
        return

    # Unstage touched files if needed.
    _run_git(workdir, "reset", "HEAD", "--", *touched_files)

    # Restore tracked paths to HEAD state.
    _run_git(workdir, "checkout", "--", *touched_files)

    # Remove untracked files among touched paths (e.g., files introduced by patch).
    _run_git(workdir, "clean", "-f", "--", *touched_files)


def _fit_power_law(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    # Fit y = a * t^b + c via log transform on (y-c) with simple grid for c
    if len(points) < 3:
        return None
    pts = [(t, y) for t, y in points if t > 0]
    if len(pts) < 3:
        return None
    ys = [y for _, y in pts]
    c_candidates = [min(ys) * 0.5, min(ys) * 0.8, min(ys) * 0.9]
    best: tuple[float, float, float, float] | None = None
    for c in c_candidates:
        try:
            xs = [__import__('math').log(t) for t, _ in pts]
            zs = [__import__('math').log(max(1e-8, y - c)) for _, y in pts]
        except ValueError:
            continue
        n = len(xs)
        mean_x = sum(xs) / n
        mean_z = sum(zs) / n
        num = sum((x - mean_x) * (z - mean_z) for x, z in zip(xs, zs))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            continue
        b = num / den
        a = __import__('math').exp(mean_z - b * mean_x)
        resid = sum((a * (t ** b) + c - y) ** 2 for t, y in pts)
        if best is None or resid < best[0]:
            best = (resid, a, b, c)
    if best is None:
        return None
    _, a, b, c = best
    return a, b, c


def _forecast_value(points: list[tuple[float, float]], t_max: float) -> float | None:
    fit = _fit_power_law(points)
    if not fit:
        return None
    a, b, c = fit
    return a * (t_max ** b) + c




def _create_worktree(base_dir: str) -> tuple[str | None, str]:
    import tempfile
    ok, reason = _ensure_git_repo(base_dir)
    if not ok:
        return None, reason
    tmp = tempfile.mkdtemp(prefix="ar-worktree-")
    cp = _run_git(base_dir, "worktree", "add", "-f", tmp, "HEAD")
    if cp.returncode != 0:
        return None, cp.stderr.strip() or cp.stdout.strip()
    return tmp, ""

def _remove_worktree(base_dir: str, wt: str) -> None:
    _run_git(base_dir, "worktree", "remove", "-f", wt)


def _reader_thread(stream, sink: list[str]) -> None:
    try:
        for line in iter(stream.readline, ""):
            sink.append(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def run_trial(
    diff: str,
    timeout_s: int,
    command: list[str] | None = None,
    *,
    workdir: str | None = None,
    apply_patch: bool = False,
    rollback_patch: bool = True,
    auto_init_git: bool = True,
    early_stop: EarlyStopConfig | None = None,
    use_worktree: bool = False,
) -> TrialResult:
    """Validate candidate diff, optionally apply patch in workdir, then run bounded command.

    Notes:
    - Patch application uses `git apply`.
    - If `auto_init_git=True` and no .git is present, runner initializes a local git repo.
    - If `rollback_patch=True`, rollback is scoped to files touched by the diff only.
    """
    v = validate_diff(diff)
    if not v.ok:
        return TrialResult(
            status="rejected",
            timeout_s=timeout_s,
            diff_len=len(diff),
            elapsed_s=0.0,
            stderr=v.reason,
        )

    touched_files = extract_touched_files_from_diff(diff)

    patch_applied = False
    trial_workdir = workdir
    worktree_dir = None
    if apply_patch:
        if not workdir:
            return TrialResult(
                status="rejected",
                timeout_s=timeout_s,
                diff_len=len(diff),
                elapsed_s=0.0,
                stderr="apply_patch=true requires workdir",
            )
        if use_worktree:
            worktree_dir, reason = _create_worktree(workdir)
            if not worktree_dir:
                return TrialResult(status="rejected", timeout_s=timeout_s, diff_len=len(diff), elapsed_s=0.0, stderr=reason)
            trial_workdir = worktree_dir
        ok, reason = _apply_patch_with_git(diff=diff, workdir=trial_workdir, auto_init_git=auto_init_git)
        if not ok:
            return TrialResult(
                status="rejected",
                timeout_s=timeout_s,
                diff_len=len(diff),
                elapsed_s=0.0,
                stderr=reason,
            )
        patch_applied = True

    if command is None:
        return TrialResult(
            status="rejected",
            timeout_s=timeout_s,
            diff_len=len(diff),
            elapsed_s=0.0,
            stderr="command_required",
            patch_applied=patch_applied,
        )

    cmd = command
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=trial_workdir or None,
    )

    out_lines: list[str] = []
    err_lines: list[str] = []

    t_out = threading.Thread(target=_reader_thread, args=(proc.stdout, out_lines), daemon=True)
    t_err = threading.Thread(target=_reader_thread, args=(proc.stderr, err_lines), daemon=True)
    t_out.start()
    t_err.start()

    start = time.monotonic()
    status = "ok"
    es = early_stop or EarlyStopConfig(enabled=False)

    try:
        while True:
            rc = proc.poll()
            elapsed = time.monotonic() - start

            if rc is not None:
                status = "ok" if rc == 0 else "failed"
                break

            if elapsed > timeout_s:
                proc.kill()
                status = "timeout"
                break

            if es.enabled and elapsed >= es.min_runtime_s:
                parsed = parse_metrics("".join(out_lines))
                bad_bpb = es.val_bpb_threshold is not None and parsed.val_bpb is not None and parsed.val_bpb > es.val_bpb_threshold
                bad_loss = es.loss_threshold is not None and parsed.loss is not None and parsed.loss > es.loss_threshold

                forecast_bad = False
                if es.forecast_enabled:
                    points = parse_metric_series("".join(out_lines))
                    series = [(p.t, p.val_bpb if es.forecast_metric == "val_bpb" else p.loss) for p in points]
                    series = [(t, v) for t, v in series if v is not None]
                    if len(series) >= es.forecast_min_points:
                        t_max = es.forecast_t_max_s or float(timeout_s)
                        predicted = _forecast_value(series, t_max)
                        if predicted is not None:
                            if es.forecast_metric == "val_bpb" and es.val_bpb_threshold is not None:
                                forecast_bad = predicted > es.val_bpb_threshold
                            if es.forecast_metric == "loss" and es.loss_threshold is not None:
                                forecast_bad = predicted > es.loss_threshold

                if bad_bpb or bad_loss or forecast_bad:
                    proc.terminate()
                    time.sleep(0.5)
                    if proc.poll() is None:
                        proc.kill()
                    status = "early_stopped"
                    break

            time.sleep(max(0.1, es.check_every_s if es.enabled else 0.2))

    finally:
        t_out.join(timeout=1)
        t_err.join(timeout=1)

        if patch_applied and rollback_patch and trial_workdir:
            _rollback_patch_with_git(trial_workdir, touched_files=touched_files)
        if worktree_dir and workdir:
            _remove_worktree(workdir, worktree_dir)

    elapsed = time.monotonic() - start
    return TrialResult(
        status=status,
        timeout_s=timeout_s,
        diff_len=len(diff),
        elapsed_s=elapsed,
        stdout="".join(out_lines),
        stderr="".join(err_lines),
        patch_applied=patch_applied,
    )
