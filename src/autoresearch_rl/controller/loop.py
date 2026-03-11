from __future__ import annotations

from dataclasses import dataclass

from autoresearch_rl.sandbox.runner import run_trial
from autoresearch_rl.telemetry.events import emit
from autoresearch_rl.telemetry.manifest import write_manifest


@dataclass
class LoopResult:
    best_score: float
    iterations: int


def _extract_val_bpb(stdout: str) -> float:
    for line in stdout.splitlines():
        if "val_bpb=" in line:
            try:
                return float(line.split("val_bpb=")[-1].strip())
            except ValueError:
                pass
    return 999.0


def run_loop(max_iterations: int = 1, trace_path: str = "traces/events.jsonl", artifacts_dir: str = "artifacts/runs") -> LoopResult:
    """Minimal orchestration loop with telemetry + manifest output."""
    best = float("inf")
    for i in range(max_iterations):
        diff = "diff --git a/train.py b/train.py\n+ # candidate change"
        trial = run_trial(diff=diff, timeout_s=30)
        score = _extract_val_bpb(trial.stdout)
        best = min(best, score)
        event = {
            "type": "trial_completed",
            "iter": i,
            "status": trial.status,
            "score": score,
            "elapsed_s": round(trial.elapsed_s, 3),
        }
        emit(trace_path, event)
        write_manifest(artifacts_dir, {**event, "stdout": trial.stdout, "stderr": trial.stderr})
    return LoopResult(best_score=best, iterations=max_iterations)
