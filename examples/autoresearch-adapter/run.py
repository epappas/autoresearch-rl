from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from autoresearch_rl.controller.contract import ContractConfig, validate_contract_files_exist
from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.sandbox.runner import run_trial
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy, check_comparability, hardware_fingerprint
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv

REPO_URL = "https://github.com/karpathy/autoresearch.git"
DEFAULT_REF = "master"
WORKDIR = Path("artifacts/autoresearch-adapter/workdir")
LEDGER = Path("artifacts/autoresearch-adapter/results.tsv")
TIMEOUT_S = 300


def _run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr=str(e))


def _ensure_repo(ref: str) -> Path:
    WORKDIR.parent.mkdir(parents=True, exist_ok=True)
    if not (WORKDIR / ".git").exists():
        if WORKDIR.exists():
            shutil.rmtree(WORKDIR)
        cp = _run(["git", "clone", "--depth", "1", "--branch", ref, REPO_URL, str(WORKDIR)])
        if cp.returncode != 0:
            raise RuntimeError(f"clone_failed: {cp.stderr.strip() or cp.stdout.strip()}")
    else:
        _run(["git", "fetch", "origin", ref], cwd=str(WORKDIR))
        cp = _run(["git", "checkout", ref], cwd=str(WORKDIR))
        if cp.returncode != 0:
            raise RuntimeError(f"checkout_failed: {cp.stderr.strip() or cp.stdout.strip()}")
        _run(["git", "pull", "--ff-only"], cwd=str(WORKDIR))
    return WORKDIR


def _has_uv() -> bool:
    return _run(["uv", "--version"]).returncode == 0


def main() -> None:
    ref = os.environ.get("AUTORESEARCH_REF", DEFAULT_REF)
    timeout_s = int(os.environ.get("AUTORESEARCH_TIMEOUT_S", str(TIMEOUT_S)))
    ledger_path = os.environ.get("AUTORESEARCH_LEDGER", str(LEDGER))

    repo = _ensure_repo(ref)

    contract = ContractConfig(
        frozen_file="prepare.py",
        mutable_file="train.py",
        program_file="program.md",
        strict=True,
    )
    ok, reason = validate_contract_files_exist(contract, root=str(repo))
    if not ok:
        raise RuntimeError(f"contract_invalid:{reason}")

    ensure_results_tsv(ledger_path)

    comp_policy = ComparabilityPolicy(
        budget_mode="fixed_wallclock",
        expected_budget_s=timeout_s,
        expected_hardware_fingerprint=None,
        strict=False,
    )
    hw_fp = hardware_fingerprint()
    comparable, non_comparable_reason = check_comparability(
        policy=comp_policy,
        run_budget_s=timeout_s,
        run_hardware_fingerprint=hw_fp,
    )

    # KISS: single real command path; no fake training.
    if not _has_uv():
        append_result_row(
            path=ledger_path,
            commit="external",
            val_bpb=0.0,
            memory_gb=0.0,
            status="crash",
            description="uv_not_available",
            episode_id="adapter",
            iter_idx=0,
            score=0.0,
            budget_mode=comp_policy.budget_mode,
            budget_s=timeout_s,
            hardware_fingerprint=hw_fp,
            comparable=comparable,
            non_comparable_reason=non_comparable_reason,
        )
        print({"ok": False, "reason": "uv_not_available", "workdir": str(repo), "ledger": ledger_path})
        return

    # use noop diff to exercise runner/timeout/log parsing without mutating repo state
    diff = "diff --git a/train.py b/train.py\n+ # adapter noop\n"
    trial = run_trial(
        diff=diff,
        timeout_s=timeout_s,
        command=["uv", "run", "train.py"],
        workdir=str(repo),
        apply_patch=False,
        rollback_patch=False,
    )

    parsed = parse_metrics(trial.stdout)
    val_bpb = parsed.val_bpb if parsed.val_bpb is not None else 0.0
    status = "keep" if trial.status == "ok" else "crash"

    commit_cp = _run(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo))
    commit = commit_cp.stdout.strip() if commit_cp.returncode == 0 else "external"

    append_result_row(
        path=ledger_path,
        commit=commit,
        val_bpb=float(val_bpb),
        memory_gb=0.0,
        status=status,
        description="karpathy_autoresearch_adapter_run",
        episode_id="adapter",
        iter_idx=0,
        score=float(val_bpb),
        budget_mode=comp_policy.budget_mode,
        budget_s=timeout_s,
        hardware_fingerprint=hw_fp,
        comparable=comparable,
        non_comparable_reason=non_comparable_reason,
    )

    print(
        {
            "ok": trial.status == "ok",
            "status": trial.status,
            "val_bpb": parsed.val_bpb,
            "workdir": str(repo),
            "ledger": ledger_path,
        }
    )


if __name__ == "__main__":
    main()
