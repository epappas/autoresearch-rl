"""Shared prompt fragments for LLM policies.

Centralized so wording stays consistent across LLMParamPolicy, LLMDiffPolicy,
and HybridPolicy. Update here, every policy reads it on the next call.
"""
from __future__ import annotations

PROGRESS_PROTOCOL_RULES = (
    "PROGRESS PROTOCOL\n"
    "The training script SHOULD call:\n"
    "    from autoresearch_rl.target.progress import emit_progress\n"
    "    emit_progress(step=..., step_target=..., metrics={...})\n"
    "at least every N steps. The controller may use these reports to early-cancel\n"
    "doomed trials. If the current source already contains emit_progress(...) calls,\n"
    "PRESERVE them — removing them is a regression that will be rejected by the\n"
    "diff validator."
)


CANCELLATION_CONTEXT_RULES = (
    "CANCELLATION CONTEXT\n"
    "An iteration with status='cancelled' was stopped early by the controller\n"
    "because its metric trajectory did not appear likely to beat the current best.\n"
    "Treat metrics from cancelled iters as PARTIAL signal — the trajectory shape\n"
    "(see progress_series) matters more than the final value. Cancelled is not a\n"
    "failure; it is a graceful early-out."
)


BATCH_DIVERSITY_RULES = (
    "BATCH DIVERSITY\n"
    "When asked to propose k > 1 candidates, return a JSON array of exactly k\n"
    "diverse proposals. Diversity dimensions to vary: learning rate (>=4x apart),\n"
    "batch size, sampling strategy. Do NOT duplicate prior history entries. Do NOT\n"
    "cluster all proposals around the current best — explore."
)


def render_progress_summary(history: list[dict]) -> str:
    """Render a one-paragraph cancellation summary from history. Empty if none."""
    cancelled = [h for h in history if h.get("status") == "cancelled"]
    if not cancelled:
        return ""
    last = cancelled[-1]
    last_iter = last.get("iter", "?")
    last_params = last.get("params", {})
    return (
        f"CANCELLATION SUMMARY: {len(cancelled)} of {len(history)} iters cancelled\n"
        f"early. Most recent cancel: iter={last_iter}, params={last_params}. Avoid\n"
        f"proposing param combinations that resemble those that were cancelled."
    )


def render_progress_series(history: list[dict], metric: str, max_iters: int = 5) -> str:
    """Render up to max_iters of progress trajectories for the LLM.

    Each iter that carries `progress_series` (list of {step, value}) gets a
    short row showing first/middle/last values. Empty when no series present.
    """
    rows: list[str] = []
    iters_with_series = [h for h in history if h.get("progress_series")]
    for entry in iters_with_series[-max_iters:]:
        series = entry.get("progress_series") or []
        if not series:
            continue
        first = series[0].get("value")
        last = series[-1].get("value")
        mid = series[len(series) // 2].get("value") if len(series) > 1 else last
        steps = len(series)
        iter_n = entry.get("iter", "?")
        status = entry.get("status", "?")
        rows.append(
            f"  iter={iter_n} status={status} steps={steps} "
            f"{metric}: first={first} mid={mid} last={last}"
        )
    if not rows:
        return ""
    return "PROGRESS TRAJECTORIES (last {} with series):\n{}".format(
        len(rows), "\n".join(rows)
    )
