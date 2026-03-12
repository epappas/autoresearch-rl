from __future__ import annotations

import os
import sys
from pathlib import Path

from autoresearch_rl.controller.contract import ContractConfig, validate_contract_files_exist
from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.sandbox.runner import run_trial
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy, check_comparability, hardware_fingerprint
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)

    contract = ContractConfig(
        frozen_file="prepare.py",
        mutable_file="train.py",
        program_file="program.md",
        strict=True,
    )
    ok, reason = validate_contract_files_exist(contract, root=".")
    if not ok:
        raise RuntimeError(reason)

    from prepare import TIME_BUDGET_S

    comp = ComparabilityPolicy(
        budget_mode="fixed_wallclock",
        expected_budget_s=TIME_BUDGET_S,
        expected_hardware_fingerprint=None,
        strict=True,
    )
    hw = hardware_fingerprint()
    comparable, mismatch = check_comparability(comp, run_budget_s=TIME_BUDGET_S, run_hardware_fingerprint=hw)
    if not comparable:
        raise RuntimeError(mismatch)

    diff = "diff --git a/train.py b/train.py\n+ # noop\n"
    trial = run_trial(
        diff=diff,
        timeout_s=TIME_BUDGET_S + 5,
        command=[sys.executable, "train.py"],
        workdir=".",
        apply_patch=False,
        rollback_patch=False,
    )

    parsed = parse_metrics(trial.stdout)
    val_bpb = parsed.val_bpb if parsed.val_bpb is not None else 0.0
    status = "keep" if trial.status == "ok" else "crash"

    ledger = here / "results.tsv"
    ensure_results_tsv(str(ledger))
    append_result_row(
        path=str(ledger),
        commit="example",
        val_bpb=float(val_bpb),
        memory_gb=0.0,
        status=status,
        description="autoresearch_like_single_run",
        episode_id="autoresearch-like",
        iter_idx=0,
        score=float(val_bpb),
        budget_mode=comp.budget_mode,
        budget_s=TIME_BUDGET_S,
        hardware_fingerprint=hw,
        comparable=True,
        non_comparable_reason="",
    )

    print({"ok": trial.status == "ok", "status": trial.status, "val_bpb": val_bpb, "ledger": str(ledger)})


if __name__ == "__main__":
    main()
