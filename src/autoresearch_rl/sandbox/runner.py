from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.sandbox.validator import validate_diff


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    check_every_s: float = 5.0
    min_runtime_s: float = 20.0
    # For minimization metrics (lower is better)
    val_bpb_threshold: float | None = None
    loss_threshold: float | None = None


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


def _rollback_patch_with_git(workdir: str) -> None:
    # Best-effort rollback to clean tracked files.
    _run_git(workdir, "reset", "--hard", "HEAD")


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
) -> TrialResult:
    """Validate candidate diff, optionally apply patch in workdir, then run bounded command.

    Notes:
    - Patch application uses `git apply`.
    - If `auto_init_git=True` and no .git is present, runner initializes a local git repo.
    - If `rollback_patch=True`, tracked files are reset with `git reset --hard HEAD` after run.
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

    patch_applied = False
    if apply_patch:
        if not workdir:
            return TrialResult(
                status="rejected",
                timeout_s=timeout_s,
                diff_len=len(diff),
                elapsed_s=0.0,
                stderr="apply_patch=true requires workdir",
            )
        ok, reason = _apply_patch_with_git(diff=diff, workdir=workdir, auto_init_git=auto_init_git)
        if not ok:
            return TrialResult(
                status="rejected",
                timeout_s=timeout_s,
                diff_len=len(diff),
                elapsed_s=0.0,
                stderr=reason,
            )
        patch_applied = True

    cmd = command or ["bash", "-lc", "echo 'val_bpb=1.234'"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workdir or None,
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
                if bad_bpb or bad_loss:
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

        if patch_applied and rollback_patch and workdir:
            _rollback_patch_with_git(workdir)

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
