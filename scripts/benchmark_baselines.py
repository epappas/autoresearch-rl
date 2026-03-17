from __future__ import annotations

import statistics

import typer

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.policy.baselines import GreedyLLMPolicy, RandomDiffPolicy
from autoresearch_rl.sandbox.runner import run_trial

app = typer.Typer()


def _run(policy, iterations: int, timeout_s: int) -> dict:
    values: list[float] = []
    statuses: list[str] = []
    first_improve_at: int | None = None
    best = float("inf")

    for i in range(iterations):
        diff = policy.propose({"iter": i, "best_score": best if best < 999 else None}).diff
        r = run_trial(
            diff=diff,
            timeout_s=timeout_s,
            command=["python3", "examples/autoresearch-like/train.py"],
            workdir=".",
            apply_patch=False,
            rollback_patch=False,
        )
        statuses.append(r.status)

        parsed = parse_metrics(r.stdout)
        score = parsed.val_bpb if parsed.val_bpb is not None else 999.0
        values.append(float(score))

        if score < best:
            best = float(score)
            if first_improve_at is None:
                first_improve_at = i

    crash_count = sum(1 for s in statuses if s in {"failed", "timeout", "rejected"})

    return {
        "iterations": iterations,
        "best_val_bpb": min(values) if values else None,
        "median_val_bpb": statistics.median(values) if values else None,
        "stdev_val_bpb": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "first_improve_at_iter": first_improve_at,
        "crash_rate": crash_count / iterations if iterations else 0.0,
        "statuses": statuses,
    }


@app.command()
def main(iterations: int = 5, timeout_s: int = 30) -> None:
    rp = _run(RandomDiffPolicy(), iterations=iterations, timeout_s=timeout_s)
    gp = _run(GreedyLLMPolicy(), iterations=iterations, timeout_s=timeout_s)
    print({"random": rp, "greedy": gp})


if __name__ == "__main__":
    app()
