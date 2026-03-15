from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from autoresearch_rl.checkpoint import (
    get_latest_snapshot_version,
    load_policy_snapshot,
    save_policy_snapshot,
)
from autoresearch_rl.policy.gae import compute_gae, compute_returns
from autoresearch_rl.policy.ppo import PPOAgent, PPOConfig
from autoresearch_rl.policy.sdpo import compute_kl_divergence, compute_sdpo_loss
from autoresearch_rl.policy.search import ParamPolicy
from autoresearch_rl.policy.interface import ParamProposal

_HISTORY_WINDOW = 8
STATE_DIM = _HISTORY_WINDOW + 3


@dataclass
class LearnedSearchConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    sdpo_beta: float = 0.1
    sdpo_alpha: float = 0.5
    sdpo_alpha_decay: float = 0.995
    sdpo_alpha_min: float = 0.1
    snapshot_every: int = 10
    snapshot_dir: str = "artifacts/policy_snapshots"
    update_every: int = 8


@dataclass
class _Transition:
    state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float = 0.0


class LearnedParamPolicy(ParamPolicy):
    """PPO-based param policy that learns from trajectory feedback."""

    def __init__(
        self,
        param_space: dict[str, list],
        config: LearnedSearchConfig | None = None,
    ):
        self.config = config or LearnedSearchConfig()
        self._build_action_space(param_space)
        self.agent = PPOAgent(
            STATE_DIM, self.action_dim, self.config.ppo
        )
        self._buffer: list[_Transition] = []
        self._pending: _Transition | None = None
        self._update_count = 0
        self._sdpo_alpha = self.config.sdpo_alpha
        self._teacher_weights: dict | None = None
        self._load_teacher_if_exists()

    def _build_action_space(self, param_space: dict[str, list]) -> None:
        self._keys = sorted(param_space.keys())
        values = [param_space[k] for k in self._keys]
        self._combos = list(itertools.product(*values)) or [()]
        self.action_dim = max(len(self._combos), 1)

    def _load_teacher_if_exists(self) -> None:
        v = get_latest_snapshot_version(self.config.snapshot_dir)
        if v >= 0:
            self._teacher_weights = load_policy_snapshot(
                self.config.snapshot_dir, v
            )

    def next(self, *, history: list[dict]) -> ParamProposal:
        state = self._extract_state_features(history)
        action, log_prob, value = self.agent.get_action_and_value(state)
        action = action % len(self._combos)
        self._pending = _Transition(
            state=state, action=action, log_prob=log_prob, value=value,
        )
        params = {
            k: v for k, v in zip(self._keys, self._combos[action])
        }
        return ParamProposal(params=params, rationale="learned")

    def record_reward(self, reward: float) -> None:
        if self._pending is None:
            return
        self._pending.reward = reward
        self._buffer.append(self._pending)
        self._pending = None
        if len(self._buffer) >= self.config.update_every:
            self._update()

    def _extract_state_features(
        self, history: list[dict]
    ) -> np.ndarray:
        features = np.zeros(STATE_DIM, dtype=np.float64)
        scores: list[float] = []
        for entry in history[-_HISTORY_WINDOW:]:
            metrics = entry.get("metrics", {})
            val = next(iter(metrics.values()), 0.0) if metrics else 0.0
            scores.append(float(val))
        for i, s in enumerate(scores):
            features[i] = s

        streak = 0
        for entry in reversed(history):
            if entry.get("decision") == "keep":
                streak += 1
            else:
                break
        features[_HISTORY_WINDOW] = float(streak)

        recent = history[-_HISTORY_WINDOW:]
        fail_count = sum(
            1 for e in recent if e.get("status") == "failed"
        )
        features[_HISTORY_WINDOW + 1] = float(fail_count)
        features[_HISTORY_WINDOW + 2] = float(len(history)) / 100.0

        return features

    def _update(self) -> dict[str, float]:
        buf = self._buffer
        states = np.array([t.state for t in buf])
        actions = np.array([t.action for t in buf])
        old_lp = np.array([t.log_prob for t in buf])
        rewards = [t.reward for t in buf]
        values = [t.value for t in buf]

        advantages = compute_gae(
            rewards, values, next_value=0.0,
            gamma=self.config.ppo.gamma, lam=self.config.ppo.lam,
        )
        returns = compute_returns(advantages, values)

        metrics = self.agent.update(
            states, actions, old_lp,
            np.array(advantages), np.array(returns),
        )

        if self._teacher_weights is not None:
            kl = self._compute_teacher_kl(states)
            sdpo_loss = compute_sdpo_loss(
                metrics["policy_loss"], kl,
                self.config.sdpo_beta * self._sdpo_alpha,
            )
            metrics["sdpo_loss"] = sdpo_loss
            metrics["kl_divergence"] = kl

        self._update_count += 1

        if self._update_count % self.config.snapshot_every == 0:
            self._save_teacher_snapshot()

        self._sdpo_alpha = max(
            self.config.sdpo_alpha_min,
            self._sdpo_alpha * self.config.sdpo_alpha_decay,
        )

        self._buffer.clear()
        return metrics

    def _compute_teacher_kl(self, states: np.ndarray) -> float:
        teacher = PPOAgent(
            STATE_DIM, self.action_dim, self.config.ppo
        )
        teacher.set_weights(self._teacher_weights)
        kl_sum = 0.0
        for i in range(states.shape[0]):
            cur_logits = self.agent.actor.forward(states[i])
            tea_logits = teacher.actor.forward(states[i])
            from autoresearch_rl.policy.ppo import _softmax
            cur_probs = _softmax(cur_logits)
            tea_probs = _softmax(tea_logits)
            kl_sum += compute_kl_divergence(cur_probs, tea_probs)
        return kl_sum / max(1, states.shape[0])

    def _save_teacher_snapshot(self) -> None:
        save_policy_snapshot(
            self.config.snapshot_dir,
            self._update_count,
            self.agent.get_weights(),
        )
        self._teacher_weights = self.agent.get_weights()

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)
