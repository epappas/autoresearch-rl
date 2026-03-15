from __future__ import annotations

import statistics

import pytest

from autoresearch_rl.telemetry.aggregation import (
    compute_episode_stats,
    compute_rolling_stats,
    compute_trend_slope,
)


# --- compute_trend_slope ---


def test_trend_slope_increasing() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    slope = compute_trend_slope(values)
    assert slope == pytest.approx(1.0)


def test_trend_slope_decreasing() -> None:
    values = [10.0, 8.0, 6.0, 4.0, 2.0]
    slope = compute_trend_slope(values)
    assert slope == pytest.approx(-2.0)


def test_trend_slope_flat() -> None:
    values = [3.0, 3.0, 3.0, 3.0]
    slope = compute_trend_slope(values)
    assert slope == pytest.approx(0.0, abs=1e-12)


def test_trend_slope_single_value() -> None:
    assert compute_trend_slope([5.0]) == 0.0


def test_trend_slope_empty() -> None:
    assert compute_trend_slope([]) == 0.0


def test_trend_slope_two_points() -> None:
    slope = compute_trend_slope([0.0, 4.0])
    assert slope == pytest.approx(4.0)


# --- compute_episode_stats ---


def test_episode_stats_known_values() -> None:
    values = [2.0, 4.0, 6.0, 8.0, 10.0]
    stats = compute_episode_stats(values)

    assert stats.mean == pytest.approx(6.0)
    assert stats.median == pytest.approx(6.0)
    assert stats.min == pytest.approx(2.0)
    assert stats.max == pytest.approx(10.0)
    assert stats.stdev == pytest.approx(statistics.stdev(values))
    assert stats.count == 5
    assert stats.trend_slope == pytest.approx(2.0)


def test_episode_stats_single_value() -> None:
    stats = compute_episode_stats([7.0])

    assert stats.mean == pytest.approx(7.0)
    assert stats.median == pytest.approx(7.0)
    assert stats.min == pytest.approx(7.0)
    assert stats.max == pytest.approx(7.0)
    assert stats.stdev == pytest.approx(0.0)
    assert stats.count == 1
    assert stats.trend_slope == pytest.approx(0.0)


def test_episode_stats_empty() -> None:
    stats = compute_episode_stats([])

    assert stats.mean == 0.0
    assert stats.median == 0.0
    assert stats.min == 0.0
    assert stats.max == 0.0
    assert stats.stdev == 0.0
    assert stats.count == 0
    assert stats.trend_slope == 0.0


def test_episode_stats_two_values() -> None:
    stats = compute_episode_stats([3.0, 9.0])

    assert stats.mean == pytest.approx(6.0)
    assert stats.median == pytest.approx(6.0)
    assert stats.min == pytest.approx(3.0)
    assert stats.max == pytest.approx(9.0)
    assert stats.stdev == pytest.approx(statistics.stdev([3.0, 9.0]))
    assert stats.count == 2
    assert stats.trend_slope == pytest.approx(6.0)


def test_episode_stats_is_frozen() -> None:
    stats = compute_episode_stats([1.0, 2.0])
    with pytest.raises(AttributeError):
        stats.mean = 99.0  # type: ignore[misc]


# --- compute_rolling_stats ---


def test_rolling_stats_window_smaller_than_list() -> None:
    values = [1.0, 2.0, 3.0, 10.0, 20.0]
    stats = compute_rolling_stats(values, window=3)

    tail = [3.0, 10.0, 20.0]
    assert stats.mean == pytest.approx(statistics.mean(tail))
    assert stats.median == pytest.approx(statistics.median(tail))
    assert stats.min == pytest.approx(3.0)
    assert stats.max == pytest.approx(20.0)
    assert stats.count == 3


def test_rolling_stats_window_equals_list() -> None:
    values = [5.0, 10.0, 15.0]
    rolling = compute_rolling_stats(values, window=3)
    full = compute_episode_stats(values)

    assert rolling.mean == pytest.approx(full.mean)
    assert rolling.count == full.count


def test_rolling_stats_window_larger_than_list() -> None:
    values = [1.0, 2.0]
    stats = compute_rolling_stats(values, window=100)

    assert stats.count == 2
    assert stats.mean == pytest.approx(1.5)


def test_rolling_stats_window_zero() -> None:
    stats = compute_rolling_stats([1.0, 2.0, 3.0], window=0)
    assert stats.count == 0


def test_rolling_stats_window_negative() -> None:
    stats = compute_rolling_stats([1.0, 2.0, 3.0], window=-1)
    assert stats.count == 0


def test_rolling_stats_empty_list() -> None:
    stats = compute_rolling_stats([], window=5)
    assert stats.count == 0
    assert stats.mean == 0.0
