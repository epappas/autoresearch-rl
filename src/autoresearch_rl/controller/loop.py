from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from autoresearch_rl.eval.judge import judge_next_state
from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.eval.scoring import TrialSignals, score_from_signals
from autoresearch_rl.policy.baselines import RandomPolicy
from autoresearch_rl.sandbox.runner import TrialResult, run_trial
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.manifest import new_run_id, write_manifest


@dataclass
class LoopResult:
    best_score: float
    iterations: int


def run_loop(
    max_iterations: int = 1,
    trace_path: str = "traces/events.jsonl",
    artifacts_dir: str = "artifacts/runs",
) -> LoopResult:
    """Async scaffold loop with proposal -> trial -> judge pipeline.

    Notes:
      - Uses deterministic baseline policy by default.
      - Uses next-state judging: score turn t after observing trial output from turn t+1.
    """
    episode_id = new_run_id()
    policy = RandomPolicy(seed=7)

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
            trial = run_trial(diff=diff, timeout_s=30)
            result_q.put({"iter": i, "diff": diff, "trial": trial})
            proposal_q.task_done()

    runner_thread = threading.Thread(target=runner_worker, daemon=True)
    runner_thread.start()

    # Stage 1: generate proposals quickly
    state = {"best_score": None}
    for i in range(max_iterations):
        diff = policy.propose_diff(state)
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
            emit(trace_path, event)
            write_manifest(
                artifacts_dir,
                {
                    **event,
                    "run_id": f"{episode_id}-{previous['iter']:04d}",
                    "stdout": previous["trial"].stdout,
                    "stderr": previous["trial"].stderr,
                    "diff": previous["diff"],
                },
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
        emit(trace_path, event)
        write_manifest(
            artifacts_dir,
            {
                **event,
                "run_id": f"{episode_id}-{previous['iter']:04d}",
                "stdout": previous["trial"].stdout,
                "stderr": previous["trial"].stderr,
                "diff": previous["diff"],
            },
        )

    proposal_q.put(stop_token)
    proposal_q.join()
    runner_thread.join(timeout=2)

    return LoopResult(best_score=best, iterations=max_iterations)
