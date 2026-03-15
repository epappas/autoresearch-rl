"""Tests for the power-law forecasting module."""
from __future__ import annotations

import pytest

from autoresearch_rl.forecasting import (
    fit_power_law,
    forecast_value,
    should_early_stop,
)


class TestFitPowerLaw:
    def test_known_linear_series(self):
        series = [2.0, 4.0, 6.0, 8.0, 10.0]
        a, b, c = fit_power_law(series)
        for i, y in enumerate(series):
            x = float(i + 1)
            predicted = a * (x**b) + c
            assert abs(predicted - y) < 1.0, (
                f"step {i+1}: predicted={predicted}, actual={y}"
            )

    def test_known_decreasing_series(self):
        series = [5.0, 3.5, 2.8, 2.4, 2.2, 2.1]
        a, b, c = fit_power_law(series)
        pred_first = a * (1.0**b) + c
        pred_last = a * (6.0**b) + c
        assert pred_first > pred_last

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_power_law([1.0, 2.0])

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_power_law([1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            fit_power_law([])


class TestForecastValue:
    def test_forecast_extrapolates(self):
        series = [1.0, 1.8, 2.5, 3.1, 3.6]
        predicted = forecast_value(series, 10)
        assert isinstance(predicted, float)

    def test_forecast_interpolates(self):
        series = [2.0, 4.0, 6.0, 8.0, 10.0]
        predicted = forecast_value(series, 3)
        assert abs(predicted - 6.0) < 1.0

    def test_forecast_step_zero_raises(self):
        with pytest.raises(ValueError, match="target_step"):
            forecast_value([1.0, 2.0, 3.0], 0)

    def test_forecast_negative_step_raises(self):
        with pytest.raises(ValueError, match="target_step"):
            forecast_value([1.0, 2.0, 3.0], -1)

    def test_forecast_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            forecast_value([1.0, 2.0], 5)


class TestShouldEarlyStop:
    def test_stop_when_forecast_exceeds_target(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert should_early_stop(series, target=3.0) is True

    def test_no_stop_when_forecast_below_target(self):
        series = [10.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.0]
        assert should_early_stop(series, target=20.0) is False

    def test_no_stop_too_few_points(self):
        series = [1.0, 2.0, 3.0]
        assert should_early_stop(series, target=0.5) is False

    def test_no_stop_below_min_points(self):
        series = [1.0, 2.0, 3.0, 4.0]
        assert (
            should_early_stop(series, target=0.1, min_points=5)
            is False
        )

    def test_stop_with_min_points_met(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert (
            should_early_stop(series, target=2.0, min_points=3)
            is True
        )

    def test_flat_series_no_stop_when_below_target(self):
        series = [2.0, 2.0, 2.0, 2.0, 2.0]
        assert should_early_stop(series, target=3.0) is False

    def test_flat_series_stop_when_above_target(self):
        series = [5.0, 5.0, 5.0, 5.0, 5.0]
        assert should_early_stop(series, target=3.0) is True

    def test_min_points_clamped_to_3(self):
        series = [10.0, 11.0, 12.0]
        result = should_early_stop(
            series, target=5.0, min_points=1
        )
        assert isinstance(result, bool)
