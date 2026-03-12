from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from autoresearch_rl.policy.interface import Proposal


@dataclass
class RandomPolicy:
    seed: int = 7

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose(self, state: Mapping[str, object]) -> Proposal:
        _ = state
        lr = self._rng.choice(["2e-3", "2.5e-3", "2.8e-3", "3e-3"])
        return Proposal(diff=f"diff --git a/train.py b/train.py\n+ learning_rate = {lr}", rationale="random_lr_choice")

    def propose_diff(self, state: Mapping[str, object]) -> str:
        return self.propose(state).diff


@dataclass
class GreedyLLMPolicy:
    """Deterministic heuristic baseline policy.

    This is intentionally simple and deterministic so benchmark runs are repeatable.
    """

    improve_threshold: float = 1.3

    def propose(self, state: Mapping[str, object]) -> Proposal:
        best = state.get("best_score")
        if best is None:
            return Proposal(
                diff="diff --git a/train.py b/train.py\n+ use_qk_norm = True",
                rationale="bootstrap_stability",
            )

        try:
            best_f = float(best)
        except (TypeError, ValueError):
            best_f = float("inf")

        if best_f > self.improve_threshold:
            return Proposal(
                diff="diff --git a/train.py b/train.py\n+ use_qk_norm = True",
                rationale="improve_stability_before_fine_tuning",
            )

        return Proposal(
            diff="diff --git a/train.py b/train.py\n+ grad_clip = 0.8",
            rationale="tighten_optimization",
        )

    def propose_diff(self, state: Mapping[str, object]) -> str:
        return self.propose(state).diff
