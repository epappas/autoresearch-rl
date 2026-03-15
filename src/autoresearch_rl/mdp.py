from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class State:
    """MDP state capturing the research environment."""

    code_hash: str
    history: tuple[dict, ...]
    metrics: dict[str, float]
    resource_budget: float
    iteration: int


@dataclass(frozen=True)
class Action:
    """MDP action -- either a param proposal or a diff proposal."""

    params: dict[str, object] | None = None
    diff: str | None = None
    rationale: str = ""


@dataclass(frozen=True)
class Reward:
    """MDP reward with scalar value and component breakdown."""

    value: float
    components: dict[str, float] = field(default_factory=dict)


TransitionFn = Callable[[State, Action], tuple[State, Reward]]


def build_state(
    code_hash: str,
    history: list[dict],
    metrics: dict[str, float],
    resource_budget: float,
    iteration: int,
) -> State:
    return State(
        code_hash=code_hash,
        history=tuple(history),
        metrics=dict(metrics),
        resource_budget=resource_budget,
        iteration=iteration,
    )


def compute_reward(
    prev_score: float,
    curr_score: float,
    status: str,
    fail_penalty: float = 0.8,
) -> Reward:
    delta = prev_score - curr_score
    components: dict[str, float] = {"score_delta": delta}
    value = delta
    if status in ("failed", "timeout", "rejected"):
        value -= fail_penalty
        components["status_penalty"] = -fail_penalty
    return Reward(value=value, components=components)
