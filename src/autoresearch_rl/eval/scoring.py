from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreWeights:
    val_bpb: float = 1.0
    loss: float = 0.15
    fail_penalty: float = 0.8
    timeout_penalty: float = 1.2
    neutral_penalty: float = 0.05
    directional_bonus: float = 0.2


@dataclass
class TrialSignals:
    status: str
    val_bpb: float | None
    loss: float | None
    eval_score: float = 0.0
    hint: str = ""


def score_from_metrics(metrics: dict) -> float:
    """Backward-compatible scalar score: lower is better."""
    return float(metrics.get("val_bpb", 999.0))


def score_from_signals(signals: TrialSignals, weights: ScoreWeights | None = None) -> float:
    """Composite objective used by the async pipeline.

    Lower is better.
    """
    w = weights or ScoreWeights()
    score = 0.0

    score += w.val_bpb * (signals.val_bpb if signals.val_bpb is not None else 999.0)
    if signals.loss is not None:
        score += w.loss * signals.loss

    if signals.status == "failed":
        score += w.fail_penalty
    elif signals.status == "timeout":
        score += w.timeout_penalty
    elif signals.status == "rejected":
        score += w.timeout_penalty

    # next-state evaluative signal: +1 helps, -1 hurts, 0 is neutral penalty
    score -= 0.25 * float(signals.eval_score)
    if signals.eval_score == 0:
        score += w.neutral_penalty

    # directional hints slightly improve score confidence
    if signals.hint.strip():
        score -= w.directional_bonus

    return score
