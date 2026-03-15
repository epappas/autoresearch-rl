from __future__ import annotations

import numpy as np

from autoresearch_rl.policy.sdpo import (
    compute_adaptive_alpha,
    compute_kl_divergence,
    compute_sdpo_loss,
)


def test_kl_divergence_identical_distributions() -> None:
    p = np.array([0.25, 0.25, 0.25, 0.25])
    kl = compute_kl_divergence(p, p)
    assert abs(kl) < 1e-7


def test_kl_divergence_different_distributions() -> None:
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    kl = compute_kl_divergence(p, q)
    assert kl > 0


def test_sdpo_loss_combines_correctly() -> None:
    ppo_loss = 2.0
    kl_div = 0.5
    alpha = 0.3
    result = compute_sdpo_loss(ppo_loss, kl_div, alpha)
    assert abs(result - (2.0 + 0.3 * 0.5)) < 1e-9


def test_adaptive_alpha_below_target() -> None:
    alpha = compute_adaptive_alpha(prev_reward=0.5, target_reward=1.0)
    assert abs(alpha - 0.5) < 1e-9


def test_adaptive_alpha_above_target() -> None:
    alpha = compute_adaptive_alpha(prev_reward=2.0, target_reward=1.0)
    assert alpha == 1.0


def test_adaptive_alpha_zero_target() -> None:
    alpha = compute_adaptive_alpha(prev_reward=0.5, target_reward=0.0)
    assert alpha == 1.0


def test_adaptive_alpha_negative_target() -> None:
    alpha = compute_adaptive_alpha(prev_reward=0.5, target_reward=-1.0)
    assert alpha == 1.0
