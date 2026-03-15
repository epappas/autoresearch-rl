from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeStats:
    mean: float
    median: float
    min: float
    max: float
    stdev: float
    count: int
    trend_slope: float


def compute_trend_slope(values: list[float]) -> float:
    """Simple linear regression slope: y = mx + b, return m.

    Uses least squares:
        m = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    """
    n = len(values)
    if n < 2:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i, y in enumerate(values):
        x = float(i)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denom


def compute_episode_stats(values: list[float]) -> EpisodeStats:
    """Compute statistics over a list of values."""
    if not values:
        return EpisodeStats(
            mean=0.0,
            median=0.0,
            min=0.0,
            max=0.0,
            stdev=0.0,
            count=0,
            trend_slope=0.0,
        )

    n = len(values)
    m = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if n >= 2 else 0.0

    return EpisodeStats(
        mean=m,
        median=med,
        min=min(values),
        max=max(values),
        stdev=sd,
        count=n,
        trend_slope=compute_trend_slope(values),
    )


def compute_rolling_stats(values: list[float], window: int) -> EpisodeStats:
    """Compute statistics over the last `window` values."""
    if window <= 0:
        return compute_episode_stats([])

    tail = values[-window:]
    return compute_episode_stats(tail)
