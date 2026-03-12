from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
import subprocess
import sys

from autoresearch_rl.controller.contract import ContractConfig, validate_contract_files_exist, validate_diff_against_contract
from autoresearch_rl.eval.judge import judge_next_state
from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.eval.scoring import TrialSignals, score_from_signals
from autoresearch_rl.policy.baselines import RandomPolicy
from autoresearch_rl.sandbox.runner import EarlyStopConfig, TrialResult, run_trial
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy, check_comparability, hardware_fingerprint
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.ledger import append_result_row, ensure_results_tsv
from autoresearch_rl.telemetry.manifest import new_run_id, write_manifest


@dataclass
class LoopResult:
    best_score: float
    iterations: int


def _current_commit_or_local(cwd: str | None = None) -> str:
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


def _record_scored_trial(
    *,
    trace_path: str,
    artifacts_dir: str,
    ledger_path: str,
    event: dict,
    trial: TrialResult,
    diff: str,
    parsed_val_bpb: float | None,
    commit: str,
    comparable: bool,
    non_comparable_reason: str,
    budget_mode: str,
    budget_s: int,
    hardware_fp: str,
) -> None:
    emit(trace_path, event)
    write_manifest(
        artifacts_dir,
        {
            **event,
            "run_id": f"{event['episode_id']}-{event['iter']:04d}",
            "stdout": trial.stdout,
            "stderr": trial.stderr,
            "diff": diff,
        },
    )
    append_result_row(
        path=ledger_path,
        commit=commit,
        val_bpb=float(parsed_val_bpb if parsed_val_bpb is not None else 0.0),
        memory_gb=0.0,
        status=(str(event["sample_type"]) if comparable else "non_comparable"),
        description="controller_loop_trial",
        episode_id=str(event["episode_id"]),
        iter_idx=int(event["iter"]),
        score=float(event["score"]),
        budget_mode=budget_mode,
        budget_s=budget_s,
        hardware_fingerprint=hardware_fp,
        comparable=comparable,
        non_comparable_reason=non_comparable_reason,
    )


def run_loop(
    max_iterations: int = 1,
    trace_path: str = "traces/events.jsonl",
    artifacts_dir: str = "artifacts/runs",
    early_stop: EarlyStopConfig | None = None,
    ledger_path: str = "results.tsv",
    mutable_file: str = "train.py",
    frozen_file: str = "prepare.py",
    program_path: str = "programs/default.md",
    contract_strict: bool = True,
    trial_timeout_s: int = 30,
    trial_command: list[str] | None = None,
    comparability_policy: ComparabilityPolicy | None = None,
) -> LoopResult:
    """Async scaffold loop with proposal -> trial -> judge pipeline.

    Notes:
      - Uses deterministic baseline policy by default.
      - Uses next-state judging: score turn t after observing trial output from turn t+1.
    """
    episode_id = new_run_id()
    proposal_policy = RandomPolicy(seed=7)
    ensure_results_tsv(ledger_path)
    commit = _current_commit_or_local()

    contract = ContractConfig(
        frozen_file=frozen_file,
        mutable_file=mutable_file,
        program_file=program_path,
        strict=contract_strict,
    )
    files_ok, files_reason = validate_contract_files_exist(contract)
    if contract.strict and not files_ok:
        raise ValueError(f"Contract validation failed: {files_reason}")

    effective_trial_command = trial_command or [sys.executable, mutable_file]

    comp_policy = comparability_policy or ComparabilityPolicy(expected_budget_s=trial_timeout_s)
    hw_fp = hardware_fingerprint()
    comparable, non_comparable_reason = check_comparability(
        policy=comp_policy,
        run_budget_s=trial_timeout_s,
        run_hardware_fingerprint=hw_fp,
    )
    if comp_policy.strict and not comparable:
        raise ValueError(f"Non-comparable run blocked: {non_comparable_reason}")

    proposal_q: queue.Queue[dict] = queue.Queue(maxsize=max(4, max_iterations * 2))
    result_q: queue.Queue[dict] = queue.Queue(maxsize=max(4, max_iterations * 2))

    stop_token = object()

    def runner_worker() -> None:
        while True:
            item = proposal_q.get()
            if item is stop_token:
                proposal_q.task_done()
                break

            i = int(item["iter"])
            diff = item["diff"]

            ok_contract, contract_reason = validate_diff_against_contract(diff, contract)
            if contract.strict and not ok_contract:
                trial = TrialResult(
                    status="rejected",
                    timeout_s=trial_timeout_s,
                    diff_len=len(diff),
                    elapsed_s=0.0,
                    stderr=contract_reason,
                )
            else:
                trial = run_trial(
                    diff=diff,
                    timeout_s=trial_timeout_s,
                    command=effective_trial_command,
                    early_stop=early_stop or EarlyStopConfig(enabled=False),
                )
            result_q.put({"iter": i, "diff": diff, "trial": trial})
            proposal_q.task_done()

    runner_thread = threading.Thread(target=runner_worker, daemon=True)
    runner_thread.start()

    # Stage 1: generate proposals quickly
    state = {"best_score": None}
    for i in range(max_iterations):
        diff = proposal_policy.propose_diff(state)
        proposal_q.put({"iter": i, "diff": diff})
        emit(trace_path, {"type": "proposal_created", "episode_id": episode_id, "iter": i, "diff_len": len(diff)})

    # Stage 2: collect results and apply next-state judging
    best = float("inf")
    previous: dict | None = None

    for _ in range(max_iterations):
        item = result_q.get()
        i = item["iter"]
        trial: TrialResult = item["trial"]

        parsed = parse_metrics(trial.stdout)
        current = {
            "iter": i,
            "trial": trial,
            "parsed": parsed,
            "diff": item["diff"],
        }

        emit(
            trace_path,
            {
                "type": "trial_completed",
                "episode_id": episode_id,
                "iter": i,
                "status": trial.status,
                "elapsed_s": round(trial.elapsed_s, 3),
                "training_seconds": round(trial.elapsed_s, 3),
                "total_seconds": round(trial.elapsed_s, 3),
                "budget_mode": comp_policy.budget_mode,
                "fixed_budget_s": trial_timeout_s,
                "hardware_fingerprint": hw_fp,
                "comparable": comparable,
                "non_comparable_reason": non_comparable_reason,
            },
        )

        if previous is not None:
            judge = judge_next_state(
                prev_status=previous["trial"].status,
                next_status=trial.status,
                next_stdout=trial.stdout,
                next_stderr=trial.stderr,
                vote_count=3,
            )

            signals = TrialSignals(
                status=previous["trial"].status,
                val_bpb=previous["parsed"].val_bpb,
                loss=previous["parsed"].loss,
                eval_score=judge.eval_score,
                hint=judge.hint,
            )
            score = score_from_signals(signals)
            best = min(best, score)
            state["best_score"] = best

            event = {
                "type": "trial_scored",
                "episode_id": episode_id,
                "iter": previous["iter"],
                "status": previous["trial"].status,
                "score": score,
                "eval_score": judge.eval_score,
                "hint": judge.hint,
                "sample_type": (
                    "combined"
                    if judge.eval_score != 0 and judge.hint
                    else "scalar_only"
                    if judge.eval_score != 0
                    else "directional_only"
                    if judge.hint
                    else "neutral"
                ),
            }
            _record_scored_trial(
                trace_path=trace_path,
                artifacts_dir=artifacts_dir,
                ledger_path=ledger_path,
                event=event,
                trial=previous["trial"],
                diff=previous["diff"],
                parsed_val_bpb=previous["parsed"].val_bpb,
                commit=commit,
                comparable=comparable,
                non_comparable_reason=non_comparable_reason,
                budget_mode=comp_policy.budget_mode,
                budget_s=trial_timeout_s,
                hardware_fp=hw_fp,
            )

        previous = current
        result_q.task_done()

    # flush final trial with neutral next-state score
    if previous is not None:
        signals = TrialSignals(
            status=previous["trial"].status,
            val_bpb=previous["parsed"].val_bpb,
            loss=previous["parsed"].loss,
            eval_score=0.0,
            hint="",
        )
        score = score_from_signals(signals)
        best = min(best, score)
        state["best_score"] = best

        event = {
            "type": "trial_scored",
            "episode_id": episode_id,
            "iter": previous["iter"],
            "status": previous["trial"].status,
            "score": score,
            "eval_score": 0.0,
            "hint": "",
            "sample_type": "neutral",
        }
        _record_scored_trial(
            trace_path=trace_path,
            artifacts_dir=artifacts_dir,
            ledger_path=ledger_path,
            event=event,
            trial=previous["trial"],
            diff=previous["diff"],
            parsed_val_bpb=previous["parsed"].val_bpb,
            commit=commit,
            comparable=comparable,
            non_comparable_reason=non_comparable_reason,
            budget_mode=comp_policy.budget_mode,
            budget_s=trial_timeout_s,
            hardware_fp=hw_fp,
        )

    proposal_q.put(stop_token)
    proposal_q.join()
    runner_thread.join(timeout=2)

    return LoopResult(best_score=best, iterations=max_iterations)
