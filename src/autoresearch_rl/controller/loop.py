from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.controller.contract import ContractConfig, validate_contract_files_exist, validate_diff_against_contract
from autoresearch_rl.eval.judge import judge_next_state
from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.eval.scoring import TrialSignals, score_from_signals
from autoresearch_rl.policy.baselines import GreedyLLMPolicy
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
        cp = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False)
        if cp.returncode == 0:
            return (cp.stdout or "").strip() or "local"
    except Exception:
        pass
    return "local"


def _infer_workdir(mutable_file: str) -> str:
    p = Path(mutable_file)
    parent = str(p.parent)
    return "." if parent in {"", "."} else parent


def _ensure_git_workdir(workdir: str) -> None:
    p = Path(workdir)
    if not (p / ".git").exists():
        subprocess.run(["git", "-C", workdir, "init"], check=False, capture_output=True, text=True)
        subprocess.run(["git", "-C", workdir, "config", "user.name", "AutoResearch"], check=False, capture_output=True, text=True)
        subprocess.run(["git", "-C", workdir, "config", "user.email", "autoresearch@local"], check=False, capture_output=True, text=True)
        subprocess.run(["git", "-C", workdir, "add", "-A"], check=False, capture_output=True, text=True)
        subprocess.run(["git", "-C", workdir, "commit", "-m", "baseline", "--allow-empty"], check=False, capture_output=True, text=True)


def _apply_diff_persist(workdir: str, diff: str) -> tuple[bool, str]:
    _ensure_git_workdir(workdir)
    cp = subprocess.run(["git", "-C", workdir, "apply", "--check", "-"], input=diff, text=True, capture_output=True, check=False)
    if cp.returncode != 0:
        return False, cp.stderr.strip() or cp.stdout.strip()

    cp2 = subprocess.run(["git", "-C", workdir, "apply", "-"], input=diff, text=True, capture_output=True, check=False)
    if cp2.returncode != 0:
        return False, cp2.stderr.strip() or cp2.stdout.strip()
    return True, ""


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
    decision: str,
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
        metric_name="val_bpb",
        metric_value=float(parsed_val_bpb if parsed_val_bpb is not None else 0.0),
        memory_gb=0.0,
        status=(decision if comparable else "non_comparable"),
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
    continuous: bool = False,
    max_wall_time_s: int | None = None,
    no_improve_limit: int | None = None,
    failure_rate_limit: float | None = None,
    failure_window: int = 10,
) -> LoopResult:
    episode_id = new_run_id()
    proposal_policy = GreedyLLMPolicy()
    ensure_results_tsv(ledger_path)
    commit = _current_commit_or_local()

    workdir = _infer_workdir(mutable_file)
    runtime_contract = ContractConfig(
        frozen_file=Path(frozen_file).name if workdir != "." else frozen_file,
        mutable_file=Path(mutable_file).name if workdir != "." else mutable_file,
        program_file=Path(program_path).name if workdir != "." else program_path,
        strict=contract_strict,
    )
    files_ok, files_reason = validate_contract_files_exist(runtime_contract, root=workdir)
    if runtime_contract.strict and not files_ok:
        raise ValueError(f"Contract validation failed: {files_reason}")

    mutable_basename = Path(mutable_file).name
    effective_trial_command = trial_command or [sys.executable, mutable_basename]

    comp_policy = comparability_policy or ComparabilityPolicy(expected_budget_s=trial_timeout_s)
    hw_fp = hardware_fingerprint()
    comparable, non_comparable_reason = check_comparability(comp_policy, trial_timeout_s, hw_fp)
    if comp_policy.strict and not comparable:
        raise ValueError(f"Non-comparable run blocked: {non_comparable_reason}")

    start_ts = time.monotonic()
    best = float("inf")
    incumbent_val_bpb = float("inf")
    no_improve_streak = 0
    iter_count = 0
    recent_statuses: list[str] = []
    history: list[dict] = []

    previous: dict | None = None

    while True:
        if not continuous and iter_count >= max_iterations:
            break
        if continuous and max_wall_time_s is not None and (time.monotonic() - start_ts) >= max_wall_time_s:
            break

        state = {
            "best_score": best if best < float("inf") else None,
            "incumbent_val_bpb": incumbent_val_bpb if incumbent_val_bpb < float("inf") else None,
            "no_improve_streak": no_improve_streak,
            "history": history[-32:],
            "mutable_file": mutable_file,
            "workdir": workdir,
        }

        diff = proposal_policy.propose_diff(state)
        emit(trace_path, {"type": "proposal_created", "episode_id": episode_id, "iter": iter_count, "diff_len": len(diff)})

        ok_contract, contract_reason = validate_diff_against_contract(diff, runtime_contract)
        if runtime_contract.strict and not ok_contract:
            trial = TrialResult(status="rejected", timeout_s=trial_timeout_s, diff_len=len(diff), elapsed_s=0.0, stderr=contract_reason)
        else:
            trial = run_trial(
                diff=diff,
                timeout_s=trial_timeout_s,
                command=effective_trial_command,
                workdir=workdir,
                apply_patch=True,
                rollback_patch=True,
                early_stop=early_stop or EarlyStopConfig(enabled=False),
            )

        parsed = parse_metrics(trial.stdout)
        current = {"iter": iter_count, "trial": trial, "parsed": parsed, "diff": diff}

        emit(
            trace_path,
            {
                "type": "trial_completed",
                "episode_id": episode_id,
                "iter": iter_count,
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

            decision = "discard"
            prev_val = previous["parsed"].val_bpb
            if previous["trial"].status == "ok" and prev_val is not None and prev_val < incumbent_val_bpb:
                applied, err = _apply_diff_persist(workdir=workdir, diff=previous["diff"])
                if applied:
                    incumbent_val_bpb = float(prev_val)
                    decision = "keep"
                    no_improve_streak = 0
                else:
                    event["keep_error"] = err
                    no_improve_streak += 1
            else:
                no_improve_streak += 1

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
                decision=decision,
            )

            history.append(
                {
                    "iter": previous["iter"],
                    "decision": decision,
                    "status": previous["trial"].status,
                    "val_bpb": previous["parsed"].val_bpb,
                    "stderr": (previous["trial"].stderr or "")[:200],
                }
            )
            recent_statuses.append(previous["trial"].status)
            if len(recent_statuses) > max(1, failure_window):
                recent_statuses.pop(0)

            if no_improve_limit is not None and no_improve_streak >= no_improve_limit:
                break
            if failure_rate_limit is not None and len(recent_statuses) >= max(1, failure_window):
                fails = sum(1 for s in recent_statuses if s in {"failed", "timeout", "rejected"})
                if (fails / len(recent_statuses)) >= failure_rate_limit:
                    break

        previous = current
        iter_count += 1

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

        decision = "discard"
        prev_val = previous["parsed"].val_bpb
        if previous["trial"].status == "ok" and prev_val is not None and prev_val < incumbent_val_bpb:
            applied, err = _apply_diff_persist(workdir=workdir, diff=previous["diff"])
            if applied:
                decision = "keep"
            else:
                event["keep_error"] = err

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
            decision=decision,
        )

    return LoopResult(best_score=best, iterations=iter_count)
