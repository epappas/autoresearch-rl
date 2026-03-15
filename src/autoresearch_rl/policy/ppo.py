from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autoresearch_rl.policy.gae import compute_gae, compute_returns  # noqa: F401


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / (np.sum(exps) + 1e-12)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    batch_size: int = 32
    hidden_dim: int = 64
    n_layers: int = 2
    novelty_coef: float = 0.1
    novelty_k: int = 5


def compute_novelty_bonus(
    state_features: np.ndarray,
    history_features: np.ndarray,
    k: int = 5,
) -> float:
    """Compute novelty as inverse of distance to k-nearest neighbors.

    Returns 1 / (1 + mean_distance_to_k_nearest).
    Returns 1.0 for empty history (maximum novelty).
    """
    if len(history_features) == 0:
        return 1.0
    distances = np.linalg.norm(history_features - state_features, axis=1)
    k_actual = min(k, len(distances))
    nearest_k = np.partition(distances, k_actual - 1)[:k_actual]
    mean_dist = float(np.mean(nearest_k))
    return 1.0 / (1.0 + mean_dist)


@dataclass
class MLP:
    """Simple feedforward network using numpy."""

    input_dim: int
    output_dim: int
    hidden_dim: int
    n_layers: int
    rng: np.random.Generator = field(repr=False)
    _params: list[tuple[np.ndarray, np.ndarray]] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self._params = []
        dims = [self.input_dim] + [self.hidden_dim] * self.n_layers + [self.output_dim]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / fan_in)
            w = self.rng.normal(0.0, std, size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self._params.append((w, b))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activations, no activation on output."""
        h = x.astype(np.float64)
        for i, (w, b) in enumerate(self._params):
            h = h @ w + b
            if i < len(self._params) - 1:
                h = np.maximum(h, 0.0)
        return h

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(w.copy(), b.copy()) for w, b in self._params]

    def set_params(self, params: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._params = [(w.copy(), b.copy()) for w, b in params]


class PPOAgent:
    """PPO actor-critic for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        rng = np.random.default_rng(42)
        self.actor = MLP(
            state_dim, action_dim, self.config.hidden_dim, self.config.n_layers, rng
        )
        self.critic = MLP(
            state_dim, 1, self.config.hidden_dim, self.config.n_layers, rng
        )

    def get_action_and_value(
        self, state: np.ndarray
    ) -> tuple[int, float, float]:
        """Returns (action_index, log_prob, value_estimate)."""
        logits = self.actor.forward(state)
        probs = _softmax(logits)
        action = int(np.argmax(probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        value = float(self.critic.forward(state)[0])
        return action, log_prob, value

    def evaluate(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate state-action pairs. Returns (log_probs, values, entropy)."""
        n = states.shape[0]
        log_probs = np.zeros(n, dtype=np.float64)
        values = np.zeros(n, dtype=np.float64)
        entropy = np.zeros(n, dtype=np.float64)

        for i in range(n):
            logits = self.actor.forward(states[i])
            probs = _softmax(logits)
            log_probs[i] = np.log(probs[int(actions[i])] + 1e-8)
            values[i] = self.critic.forward(states[i])[0]
            entropy[i] = -np.sum(probs * np.log(probs + 1e-8))

        return log_probs, values, entropy

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """PPO clipped update. Returns loss metrics dict."""
        cfg = self.config
        n = states.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0

        for _epoch in range(cfg.epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, cfg.batch_size):
                end = min(start + cfg.batch_size, n)
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_adv = advantages[batch_idx]
                b_ret = returns[batch_idx]

                adv_std = np.std(b_adv) + 1e-8
                b_adv_norm = (b_adv - np.mean(b_adv)) / adv_std

                new_lp, vals, ent = self.evaluate(b_states, b_actions)

                ratio = np.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv_norm
                surr2 = np.clip(
                    ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon
                ) * b_adv_norm

                total_policy_loss = float(-np.mean(np.minimum(surr1, surr2)))
                total_value_loss = float(np.mean((b_ret - vals) ** 2))
                total_entropy_loss = float(-np.mean(ent))

                self._sgd_step(b_states, b_actions, b_old_lp, b_adv_norm, b_ret)

        total_loss = (
            total_policy_loss
            + cfg.value_coef * total_value_loss
            + cfg.entropy_coef * total_entropy_loss
        )
        return {
            "policy_loss": total_policy_loss,
            "value_loss": total_value_loss,
            "entropy_loss": total_entropy_loss,
            "total_loss": float(total_loss),
        }

    def _sgd_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """Update actor and critic weights via finite-difference gradient."""
        eps = 1e-4
        lr = self.config.lr

        for layer_idx, (w, b) in enumerate(self.actor._params):
            w_grad = np.zeros_like(w)
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    w[i, j] += eps
                    lp = self._actor_loss(states, actions, old_log_probs, advantages)
                    w[i, j] -= 2.0 * eps
                    lm = self._actor_loss(states, actions, old_log_probs, advantages)
                    w[i, j] += eps
                    w_grad[i, j] = (lp - lm) / (2.0 * eps)
            b_grad = np.zeros_like(b)
            for j in range(b.shape[0]):
                b[j] += eps
                lp = self._actor_loss(states, actions, old_log_probs, advantages)
                b[j] -= 2.0 * eps
                lm = self._actor_loss(states, actions, old_log_probs, advantages)
                b[j] += eps
                b_grad[j] = (lp - lm) / (2.0 * eps)
            self.actor._params[layer_idx] = (w - lr * w_grad, b - lr * b_grad)

        for layer_idx, (w, b) in enumerate(self.critic._params):
            w_grad = np.zeros_like(w)
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    w[i, j] += eps
                    lp = self._critic_loss(states, returns)
                    w[i, j] -= 2.0 * eps
                    lm = self._critic_loss(states, returns)
                    w[i, j] += eps
                    w_grad[i, j] = (lp - lm) / (2.0 * eps)
            b_grad = np.zeros_like(b)
            for j in range(b.shape[0]):
                b[j] += eps
                lp = self._critic_loss(states, returns)
                b[j] -= 2.0 * eps
                lm = self._critic_loss(states, returns)
                b[j] += eps
                b_grad[j] = (lp - lm) / (2.0 * eps)
            self.critic._params[layer_idx] = (w - lr * w_grad, b - lr * b_grad)

    def _actor_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> float:
        new_lp, _, _ = self.evaluate(states, actions)
        ratio = np.exp(new_lp - old_log_probs)
        eps = self.config.epsilon
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
        return float(-np.mean(np.minimum(surr1, surr2)))

    def _critic_loss(self, states: np.ndarray, returns: np.ndarray) -> float:
        values = np.array(
            [self.critic.forward(states[i])[0] for i in range(len(states))]
        )
        return float(np.mean((returns - values) ** 2))

    def get_weights(self) -> dict:
        """Serialize weights for checkpointing."""
        return {
            "actor": [(w.tolist(), b.tolist()) for w, b in self.actor._params],
            "critic": [(w.tolist(), b.tolist()) for w, b in self.critic._params],
        }

    def set_weights(self, weights: dict) -> None:
        """Restore weights from checkpoint."""
        actor_params = [(np.array(w), np.array(b)) for w, b in weights["actor"]]
        critic_params = [(np.array(w), np.array(b)) for w, b in weights["critic"]]
        self.actor.set_params(actor_params)
        self.critic.set_params(critic_params)
