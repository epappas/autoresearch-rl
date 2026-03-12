from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class JudgeVote:
    score: int  # -1, 0, +1
    hint: str = ""


@dataclass
class JudgeResult:
    eval_score: float
    hint: str
    votes: list[JudgeVote]


def _heuristic_vote(prev_status: str, next_status: str, next_stdout: str, next_stderr: str) -> JudgeVote:
    text = f"{next_stdout}\n{next_stderr}".lower()

    if next_status in {"failed", "timeout", "rejected"}:
        return JudgeVote(score=-1, hint="Follow-up state indicates regression/failure.")

    if "error" in text or "traceback" in text or "exception" in text:
        return JudgeVote(score=-1, hint="Follow-up logs contain runtime errors.")

    if "val_bpb" in text or "improved" in text or "success" in text:
        return JudgeVote(score=1, hint="Follow-up state indicates progress in metrics.")

    if prev_status == "ok" and next_status == "ok":
        return JudgeVote(score=1, hint="Consecutive successful runs suggest healthy direction.")

    return JudgeVote(score=0, hint="Ambiguous next-state evidence.")


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
    # Deterministic scaffold voting (identical votes for now, easy to swap for LLM judge).
    votes = [
        _heuristic_vote(prev_status=prev_status, next_status=next_status, next_stdout=next_stdout, next_stderr=next_stderr)
        for _ in range(max(1, vote_count))
    ]

    eval_score = majority_vote([v.score for v in votes])
    # longest non-trivial positive hint
    positive_hints = [v.hint.strip() for v in votes if v.score == 1 and len(v.hint.strip()) > 10]
    hint = max(positive_hints, key=len) if positive_hints else ""

    return JudgeResult(eval_score=eval_score, hint=hint, votes=votes)
