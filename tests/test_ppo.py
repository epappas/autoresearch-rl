from __future__ import annotations

import numpy as np
import pytest

from autoresearch_rl.policy.ppo import MLP, PPOAgent, PPOConfig, _softmax, compute_novelty_bonus


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


class TestMLP:
    def test_forward_shape(self, rng: np.random.Generator) -> None:
        mlp = MLP(input_dim=4, output_dim=3, hidden_dim=16, n_layers=2, rng=rng)
        x = rng.standard_normal(4)
        out = mlp.forward(x)
        assert out.shape == (3,)

    def test_forward_batch_shape(self, rng: np.random.Generator) -> None:
        mlp = MLP(input_dim=4, output_dim=3, hidden_dim=16, n_layers=2, rng=rng)
        x = rng.standard_normal((5, 4))
        out = mlp.forward(x)
        assert out.shape == (5, 3)

    def test_forward_deterministic(self, rng: np.random.Generator) -> None:
        mlp = MLP(input_dim=4, output_dim=2, hidden_dim=8, n_layers=1, rng=rng)
        x = rng.standard_normal(4)
        out1 = mlp.forward(x)
        out2 = mlp.forward(x)
        np.testing.assert_array_equal(out1, out2)

    def test_get_set_params_roundtrip(self, rng: np.random.Generator) -> None:
        mlp = MLP(input_dim=3, output_dim=2, hidden_dim=8, n_layers=1, rng=rng)
        x = rng.standard_normal(3)
        original_out = mlp.forward(x)
        params = mlp.get_params()
        for w, b in mlp._params:
            w[:] = 0.0
            b[:] = 0.0
        assert not np.allclose(mlp.forward(x), original_out)
        mlp.set_params(params)
        np.testing.assert_array_almost_equal(mlp.forward(x), original_out)


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        probs = _softmax(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-7

    def test_all_positive(self) -> None:
        logits = np.array([-10.0, 0.0, 10.0])
        probs = _softmax(logits)
        assert np.all(probs >= 0.0)

    def test_numerical_stability(self) -> None:
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = _softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-7

    def test_uniform_for_equal_logits(self) -> None:
        logits = np.array([5.0, 5.0, 5.0])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)


class TestPPOAgent:
    def test_get_action_and_value_types(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action, log_prob, value = agent.get_action_and_value(state)
        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert 0 <= action < 3

    def test_get_action_and_value_valid_log_prob(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        _, log_prob, _ = agent.get_action_and_value(state)
        assert log_prob <= 0.0

    def test_evaluate_shapes(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3)
        n = 8
        states = np.random.default_rng(0).standard_normal((n, 4))
        actions = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        log_probs, values, entropy = agent.evaluate(states, actions)
        assert log_probs.shape == (n,)
        assert values.shape == (n,)
        assert entropy.shape == (n,)

    def test_evaluate_entropy_non_negative(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3)
        states = np.random.default_rng(2).standard_normal((5, 4))
        actions = np.array([0, 1, 2, 0, 1])
        _, _, entropy = agent.evaluate(states, actions)
        assert np.all(entropy >= 0.0)

    def test_update_returns_loss_dict(self) -> None:
        cfg = PPOConfig(epochs=1, batch_size=4, hidden_dim=4, n_layers=1)
        agent = PPOAgent(state_dim=2, action_dim=2, config=cfg)
        n = 4
        rng = np.random.default_rng(99)
        states = rng.standard_normal((n, 2))
        actions = np.array([0, 1, 0, 1])
        old_log_probs = np.full(n, -0.7)
        advantages = rng.standard_normal(n)
        returns = rng.standard_normal(n)

        losses = agent.update(states, actions, old_log_probs, advantages, returns)
        assert isinstance(losses, dict)
        for key in ("policy_loss", "value_loss", "entropy_loss", "total_loss"):
            assert key in losses
            assert isinstance(losses[key], float)
            assert np.isfinite(losses[key])

    def test_weight_get_set_roundtrip(self) -> None:
        agent = PPOAgent(state_dim=4, action_dim=3)
        state = np.array([0.5, -0.5, 0.1, -0.1])
        original_action, original_lp, original_val = agent.get_action_and_value(state)

        weights = agent.get_weights()
        for layer_w, layer_b in weights["actor"]:
            assert isinstance(layer_w, list)
            assert isinstance(layer_b, list)

        agent2 = PPOAgent(state_dim=4, action_dim=3)
        agent2.set_weights(weights)
        restored_action, restored_lp, restored_val = agent2.get_action_and_value(state)

        assert restored_action == original_action
        assert abs(restored_lp - original_lp) < 1e-10
        assert abs(restored_val - original_val) < 1e-10

    def test_config_defaults(self) -> None:
        cfg = PPOConfig()
        assert cfg.lr == 3e-4
        assert cfg.gamma == 0.99
        assert cfg.epsilon == 0.2
        assert cfg.novelty_coef == 0.1
        assert cfg.novelty_k == 5


class TestNoveltyBonus:
    def test_empty_history_returns_one(self) -> None:
        state = np.array([1.0, 2.0, 3.0])
        history = np.empty((0, 3))
        assert compute_novelty_bonus(state, history, k=5) == 1.0

    def test_identical_state_high_value(self) -> None:
        state = np.array([1.0, 2.0, 3.0])
        history = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        bonus = compute_novelty_bonus(state, history, k=5)
        assert bonus == 1.0

    def test_distant_state_lower_value(self) -> None:
        state = np.array([0.0, 0.0, 0.0])
        history = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        bonus = compute_novelty_bonus(state, history, k=5)
        assert bonus < 0.1

    def test_k_limits_neighbors(self) -> None:
        state = np.array([0.0, 0.0])
        history = np.array([
            [0.01, 0.01],
            [100.0, 100.0],
            [200.0, 200.0],
            [300.0, 300.0],
            [400.0, 400.0],
        ])
        bonus_k1 = compute_novelty_bonus(state, history, k=1)
        bonus_k5 = compute_novelty_bonus(state, history, k=5)
        assert bonus_k1 > bonus_k5
        assert bonus_k1 > 0.9
