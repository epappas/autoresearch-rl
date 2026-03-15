"""Generalized Advantage Estimation (GAE).

Paper ref: AutoResearch-RL Section 2 -- GAE for variance-reduced advantage estimation.

A_t = sum_{l=0}^{T} (gamma * lam)^l * delta_{t+l}
where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
"""
from __future__ import annotations


def compute_gae(
    rewards: list[float],
    values: list[float],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: rewards at each timestep [r_0, r_1, ..., r_{T-1}]
        values: value estimates [V(s_0), V(s_1), ..., V(s_{T-1})]
        next_value: value estimate of the terminal state V(s_T)
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: GAE advantages [A_0, A_1, ..., A_{T-1}]
    """
    n = len(rewards)
    if n == 0:
        return []

    advantages = [0.0] * n
    gae = 0.0

    for t in reversed(range(n)):
        next_val = next_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages


def compute_returns(
    advantages: list[float], values: list[float]
) -> list[float]:
    """Compute returns from advantages and values: R_t = A_t + V(s_t)."""
    return [a + v for a, v in zip(advantages, values)]
