"""Tests for GAE with hand-computed expected values."""
from __future__ import annotations

import pytest

from autoresearch_rl.policy.gae import compute_gae, compute_returns


def test_single_timestep() -> None:
    # delta_0 = r_0 + gamma * next_value - V(s_0) = 1.0 + 0.99*0.0 - 0.5 = 0.5
    # A_0 = delta_0 = 0.5
    adv = compute_gae(
        rewards=[1.0], values=[0.5], next_value=0.0, gamma=0.99, lam=0.95
    )
    assert len(adv) == 1
    assert adv[0] == pytest.approx(0.5)


def test_two_timesteps_hand_computed() -> None:
    # gamma=1.0, lam=1.0 for simpler hand calculation
    # delta_1 = r_1 + 1.0*next_value - V(s_1) = 3.0 + 1.0*0.0 - 2.0 = 1.0
    # delta_0 = r_0 + 1.0*V(s_1) - V(s_0) = 1.0 + 1.0*2.0 - 1.0 = 2.0
    # A_1 = delta_1 = 1.0
    # A_0 = delta_0 + 1.0*1.0*A_1 = 2.0 + 1.0 = 3.0
    adv = compute_gae(
        rewards=[1.0, 3.0],
        values=[1.0, 2.0],
        next_value=0.0,
        gamma=1.0,
        lam=1.0,
    )
    assert adv[0] == pytest.approx(3.0)
    assert adv[1] == pytest.approx(1.0)


def test_gamma_one_lambda_one_equals_mc_returns() -> None:
    """gamma=1, lambda=1 reduces GAE to standard MC advantage."""
    rewards = [1.0, 2.0, 3.0]
    values = [0.0, 0.0, 0.0]
    # MC returns: R_0=1+2+3=6, R_1=2+3=5, R_2=3
    # Advantages (V=0): A_t = R_t - V_t = R_t
    adv = compute_gae(
        rewards=rewards, values=values, next_value=0.0, gamma=1.0, lam=1.0
    )
    assert adv[0] == pytest.approx(6.0)
    assert adv[1] == pytest.approx(5.0)
    assert adv[2] == pytest.approx(3.0)


def test_zero_gamma() -> None:
    """gamma=0 means no lookahead: A_t = r_t - V(s_t)."""
    adv = compute_gae(
        rewards=[1.0, 2.0, 3.0],
        values=[0.5, 1.0, 1.5],
        next_value=0.0,
        gamma=0.0,
        lam=0.95,
    )
    assert adv[0] == pytest.approx(0.5)
    assert adv[1] == pytest.approx(1.0)
    assert adv[2] == pytest.approx(1.5)


def test_zero_lambda() -> None:
    """lambda=0 means 1-step TD: A_t = delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)."""
    gamma = 0.99
    adv = compute_gae(
        rewards=[1.0, 2.0],
        values=[0.5, 1.0],
        next_value=0.0,
        gamma=gamma,
        lam=0.0,
    )
    # delta_1 = 2.0 + 0.99*0.0 - 1.0 = 1.0
    # delta_0 = 1.0 + 0.99*1.0 - 0.5 = 1.49
    assert adv[1] == pytest.approx(1.0)
    assert adv[0] == pytest.approx(1.49)


def test_empty_input() -> None:
    adv = compute_gae(rewards=[], values=[], next_value=0.0)
    assert adv == []


def test_compute_returns() -> None:
    advantages = [3.0, 2.0, 1.0]
    values = [1.0, 2.0, 3.0]
    returns = compute_returns(advantages, values)
    assert returns == [4.0, 4.0, 4.0]


def test_compute_returns_empty() -> None:
    returns = compute_returns([], [])
    assert returns == []
