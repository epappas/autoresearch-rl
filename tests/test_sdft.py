from __future__ import annotations

import numpy as np

from autoresearch_rl.distillation.sdft import (
    SDFTConfig,
    apply_top_k_filter,
    compute_sdft_loss,
    should_distill,
    softmax,
)


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax(logits)
        assert abs(np.sum(result) - 1.0) < 1e-6

    def test_sums_to_one_2d(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = softmax(logits)
        for row in result:
            assert abs(np.sum(row) - 1.0) < 1e-6

    def test_higher_temperature_flatter(self) -> None:
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        dist_t1 = softmax(logits, temperature=1.0)
        dist_t2 = softmax(logits, temperature=2.0)
        assert np.max(dist_t1) > np.max(dist_t2)
        assert np.min(dist_t1) < np.min(dist_t2)

    def test_temperature_one_standard_softmax(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        result = softmax(logits, temperature=1.0)
        exp_vals = np.exp(logits - np.max(logits))
        expected = exp_vals / np.sum(exp_vals)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_all_equal_logits(self) -> None:
        logits = np.array([5.0, 5.0, 5.0, 5.0])
        result = softmax(logits)
        np.testing.assert_allclose(
            result, np.array([0.25, 0.25, 0.25, 0.25]), atol=1e-7
        )


class TestComputeSDFTLoss:
    def test_identical_distributions_near_zero(self) -> None:
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        loss = compute_sdft_loss(logits, logits, temperature=2.0)
        assert abs(loss) < 1e-6

    def test_different_distributions_positive(self) -> None:
        teacher = np.array([1.0, 5.0, 2.0, 0.5])
        student = np.array([3.0, 1.0, 0.5, 2.0])
        loss = compute_sdft_loss(teacher, student, temperature=2.0)
        assert loss > 0.0

    def test_kl_divergence_not_symmetric(self) -> None:
        a = np.array([1.0, 5.0, 2.0])
        b = np.array([3.0, 1.0, 0.5])
        loss_ab = compute_sdft_loss(a, b, temperature=2.0)
        loss_ba = compute_sdft_loss(b, a, temperature=2.0)
        assert loss_ab > 0.0
        assert loss_ba > 0.0
        assert abs(loss_ab - loss_ba) > 1e-6

    def test_hand_verified_value(self) -> None:
        teacher = np.array([1.0, 0.0])
        student = np.array([0.0, 1.0])
        loss = compute_sdft_loss(teacher, student, temperature=1.0)
        assert 0.3 < loss < 0.6

    def test_default_temperature(self) -> None:
        teacher = np.array([1.0, 5.0, 2.0])
        student = np.array([3.0, 1.0, 0.5])
        loss_default = compute_sdft_loss(teacher, student)
        loss_explicit = compute_sdft_loss(teacher, student, temperature=2.0)
        assert abs(loss_default - loss_explicit) < 1e-10


class TestApplyTopKFilter:
    def test_keeps_exactly_k_values(self) -> None:
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = apply_top_k_filter(logits, k=3)
        finite_count = np.sum(np.isfinite(result))
        assert finite_count == 3

    def test_keeps_top_values(self) -> None:
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = apply_top_k_filter(logits, k=2)
        assert result[1] == 5.0
        assert result[4] == 4.0
        assert result[0] == -np.inf
        assert result[2] == -np.inf
        assert result[3] == -np.inf

    def test_k_greater_or_equal_returns_copy(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        result = apply_top_k_filter(logits, k=3)
        np.testing.assert_array_equal(result, logits)
        assert result is not logits

    def test_k_greater_than_length_returns_copy(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        result = apply_top_k_filter(logits, k=10)
        np.testing.assert_array_equal(result, logits)
        assert result is not logits

    def test_k_one_keeps_max(self) -> None:
        logits = np.array([1.0, 5.0, 3.0])
        result = apply_top_k_filter(logits, k=1)
        assert result[1] == 5.0
        assert result[0] == -np.inf
        assert result[2] == -np.inf


class TestShouldDistill:
    def test_above_threshold(self) -> None:
        assert should_distill(0.8, 0.5) is True

    def test_below_threshold(self) -> None:
        assert should_distill(0.3, 0.5) is False

    def test_at_threshold(self) -> None:
        assert should_distill(0.5, 0.5) is True

    def test_zero_confidence(self) -> None:
        assert should_distill(0.0, 0.5) is False

    def test_zero_threshold(self) -> None:
        assert should_distill(0.0, 0.0) is True


class TestSDFTConfig:
    def test_default_values(self) -> None:
        config = SDFTConfig()
        assert config.temperature == 2.0
        assert config.top_k == 10
        assert config.confidence_threshold == 0.5

    def test_custom_values(self) -> None:
        config = SDFTConfig(temperature=3.0, top_k=20, confidence_threshold=0.8)
        assert config.temperature == 3.0
        assert config.top_k == 20
        assert config.confidence_threshold == 0.8
