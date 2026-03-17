"""Hybrid policy that switches between param search and code diff proposals.

Starts with param-based exploration to establish a baseline, then switches
to diff-based code modifications when the param search stalls. Falls back
to param mode if diff proposals fail consecutively.
"""
from __future__ import annotations

import logging
from typing import Literal

from autoresearch_rl.policy.interface import Proposal
from autoresearch_rl.policy.llm_diff import LLMDiffPolicy
from autoresearch_rl.policy.llm_search import LLMParamPolicy

logger = logging.getLogger(__name__)


class HybridPolicy:
    """Wraps a param policy and a diff policy with heuristic switching."""

    def __init__(
        self,
        param_policy: LLMParamPolicy,
        diff_policy: LLMDiffPolicy,
        *,
        param_explore_iters: int = 5,
        stall_threshold: int = 3,
        diff_failure_limit: int = 3,
    ) -> None:
        self._param_policy = param_policy
        self._diff_policy = diff_policy
        self._param_explore_iters = param_explore_iters
        self._stall_threshold = stall_threshold
        self._diff_failure_limit = diff_failure_limit
        self._diff_consecutive_failures = 0
        self._active_mode: Literal["param", "diff"] = "param"

    def propose(self, state: dict) -> Proposal:
        history: list[dict] = state.get("history", [])
        mode = self._select_mode(history)
        self._active_mode = mode
        logger.debug("Hybrid policy selected mode: %s", mode)

        if mode == "diff":
            return self._diff_policy.propose(state)
        return self._param_policy.propose(state)

    def record_reward(self, reward: float) -> None:
        if self._active_mode == "diff":
            if reward <= -0.1:
                self._diff_consecutive_failures += 1
            else:
                self._diff_consecutive_failures = 0

    def _select_mode(self, history: list[dict]) -> Literal["param", "diff"]:
        if len(history) < self._param_explore_iters:
            return "param"

        no_improve = 0
        for entry in reversed(history):
            if entry.get("decision") == "keep":
                break
            no_improve += 1

        if no_improve >= self._stall_threshold:
            if self._diff_consecutive_failures < self._diff_failure_limit:
                return "diff"

        return "param"

    @property
    def active_mode(self) -> str:
        return self._active_mode
