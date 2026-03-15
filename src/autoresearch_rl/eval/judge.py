from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_VAL_BPB_RE = re.compile(
    r"val[_-]?bpb\s*[:=]\s*([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_LOSS_RE = re.compile(
    r"(?:^|\s)loss\s*[:=]\s*([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE | re.MULTILINE,
)
_ERROR_KEYWORDS = {"error", "exception", "fault", "panic", "fatal"}
_WARNING_RE = re.compile(r"\bwarning\b", re.IGNORECASE)
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE)


@dataclass
class JudgeVote:
    score: int  # -1, 0, +1
    hint: str = ""


@dataclass
class JudgeResult:
    eval_score: float
    hint: str
    votes: list[JudgeVote]


def _status_judge(
    prev_status: str,
    next_status: str,
    next_stdout: str,
    next_stderr: str,
) -> JudgeVote:
    """Judge based purely on status transitions."""
    failure_states = {"failed", "timeout", "rejected"}

    if next_status in failure_states:
        if prev_status in failure_states:
            return JudgeVote(
                score=-1, hint="Consecutive failures indicate regression."
            )
        return JudgeVote(
            score=-1,
            hint="Transition from non-failure to failure state.",
        )

    if next_status == "ok" and prev_status in failure_states:
        return JudgeVote(
            score=1, hint="Recovery from failure to successful state."
        )

    if next_status == "ok" and prev_status == "ok":
        return JudgeVote(
            score=1,
            hint="Consecutive successful runs suggest healthy direction.",
        )

    return JudgeVote(score=0, hint="Status transition is ambiguous.")


def _metric_judge(
    prev_status: str,
    next_status: str,
    next_stdout: str,
    next_stderr: str,
) -> JudgeVote:
    """Judge based on metric keywords and numeric values in stdout."""
    text = f"{next_stdout}\n{next_stderr}"

    bpb_matches = _VAL_BPB_RE.findall(text)
    loss_matches = _LOSS_RE.findall(text)

    has_bpb = len(bpb_matches) > 0
    has_loss = len(loss_matches) > 0

    if not has_bpb and not has_loss:
        return JudgeVote(score=0, hint="No metric signals found in output.")

    # Use last-seen values (final reported metrics)
    if has_bpb:
        last_bpb = float(bpb_matches[-1])
        if last_bpb <= 0.0:
            return JudgeVote(
                score=-1,
                hint=f"val_bpb={last_bpb} is non-positive, likely invalid.",
            )
        # Multiple bpb values => check if trend is decreasing (improving)
        if len(bpb_matches) >= 2:
            first_bpb = float(bpb_matches[0])
            if last_bpb < first_bpb:
                return JudgeVote(
                    score=1,
                    hint=f"val_bpb improved from {first_bpb} to {last_bpb}.",
                )
            if last_bpb > first_bpb:
                return JudgeVote(
                    score=-1,
                    hint=f"val_bpb regressed from {first_bpb} to {last_bpb}.",
                )
        # Single bpb present is a positive signal
        return JudgeVote(
            score=1,
            hint=f"val_bpb={last_bpb} metric reported.",
        )

    # Only loss present
    last_loss = float(loss_matches[-1])
    if len(loss_matches) >= 2:
        first_loss = float(loss_matches[0])
        if last_loss < first_loss:
            return JudgeVote(
                score=1,
                hint=f"Loss decreased from {first_loss} to {last_loss}.",
            )
        if last_loss > first_loss:
            return JudgeVote(
                score=-1,
                hint=f"Loss increased from {first_loss} to {last_loss}.",
            )

    return JudgeVote(score=0, hint=f"Loss={last_loss} with no trend data.")


def _log_quality_judge(
    prev_status: str,
    next_status: str,
    next_stdout: str,
    next_stderr: str,
) -> JudgeVote:
    """Judge based on log quality: errors, warnings, tracebacks, output length."""
    combined = f"{next_stdout}\n{next_stderr}"
    lower = combined.lower()
    lines = combined.strip().splitlines()

    traceback_count = len(_TRACEBACK_RE.findall(combined))
    if traceback_count > 0:
        return JudgeVote(
            score=-1,
            hint=f"Found {traceback_count} traceback(s) in output.",
        )

    error_hits = sum(1 for kw in _ERROR_KEYWORDS if kw in lower)
    if error_hits >= 2:
        return JudgeVote(
            score=-1,
            hint=f"Multiple error keywords ({error_hits}) found in logs.",
        )

    warning_count = len(_WARNING_RE.findall(combined))
    line_count = len(lines)

    if error_hits == 1 and warning_count > 3:
        return JudgeVote(
            score=-1,
            hint="Error keyword with excessive warnings in logs.",
        )

    if line_count < 2 and not lower.strip():
        return JudgeVote(score=-1, hint="Output is empty or near-empty.")

    if error_hits == 0 and warning_count == 0 and line_count >= 3:
        return JudgeVote(
            score=1,
            hint="Clean output with no errors or warnings.",
        )

    if error_hits == 0 and warning_count <= 2:
        return JudgeVote(
            score=0,
            hint=f"Acceptable output with {warning_count} warning(s).",
        )

    return JudgeVote(score=0, hint="Log quality is ambiguous.")


_JUDGES = [_status_judge, _metric_judge, _log_quality_judge]


def majority_vote(scores: list[int]) -> float:
    if not scores:
        return 0.0
    counter = Counter(scores)
    top_score, top_count = counter.most_common(1)[0]
    if list(counter.values()).count(top_count) > 1:
        return 0.0
    return float(top_score)


def judge_next_state(
    prev_status: str,
    next_status: str,
    next_stdout: str,
    next_stderr: str,
    vote_count: int = 3,
) -> JudgeResult:
    """Run three distinct judge strategies and aggregate via majority vote."""
    votes = [
        judge_fn(
            prev_status=prev_status,
            next_status=next_status,
            next_stdout=next_stdout,
            next_stderr=next_stderr,
        )
        for judge_fn in _JUDGES
    ]

    eval_score = majority_vote([v.score for v in votes])
    positive_hints = [
        v.hint.strip()
        for v in votes
        if v.score == 1 and len(v.hint.strip()) > 10
    ]
    hint = max(positive_hints, key=len) if positive_hints else ""

    return JudgeResult(eval_score=eval_score, hint=hint, votes=votes)
