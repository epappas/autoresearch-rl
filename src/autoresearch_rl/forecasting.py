"""Power-law forecasting for early-stop decisions."""
from __future__ import annotations

import math


def fit_power_law(
    series: list[float],
) -> tuple[float, float, float]:
    """Fit y = a * x^b + c to a series indexed 1..N.

    Uses log-linear regression on (y - c) with a grid search over c.
    Returns (a, b, c) coefficients.
    Raises ValueError when the series is too short or the fit fails.
    """
    if len(series) < 3:
        raise ValueError(
            f"Need at least 3 points, got {len(series)}"
        )

    points = [(float(i + 1), y) for i, y in enumerate(series)]
    result = _fit_power_law_points(points)
    if result is None:
        raise ValueError("Power-law fit failed")
    return result


def forecast_value(
    series: list[float], target_step: int
) -> float:
    """Predict value at target_step using power-law fit.

    target_step is 1-based (step 1 corresponds to series[0]).
    """
    if target_step < 1:
        raise ValueError("target_step must be >= 1")
    a, b, c = fit_power_law(series)
    return a * (float(target_step) ** b) + c


def should_early_stop(
    series: list[float],
    target: float,
    min_points: int = 5,
) -> bool:
    """Return True if the forecasted final value won't beat target.

    Assumes a minimization objective: early-stops when the forecast
    at the last step exceeds `target`.

    Returns False (don't stop) when there are fewer than min_points
    data points or when the fit fails.
    """
    if min_points < 3:
        min_points = 3
    if len(series) < min_points:
        return False
    try:
        predicted = forecast_value(series, len(series))
    except ValueError:
        return False
    return predicted > target


def _fit_power_law_points(
    points: list[tuple[float, float]],
) -> tuple[float, float, float] | None:
    """Core power-law fitting on (t, y) pairs via log-linear regression."""
    pts = [(t, y) for t, y in points if t > 0]
    if len(pts) < 3:
        return None

    ys = [y for _, y in pts]
    c_candidates = [
        min(ys) * 0.5,
        min(ys) * 0.8,
        min(ys) * 0.9,
    ]

    best: tuple[float, float, float, float] | None = None
    for c in c_candidates:
        try:
            xs = [math.log(t) for t, _ in pts]
            zs = [
                math.log(max(1e-8, y - c)) for _, y in pts
            ]
        except ValueError:
            continue

        n = len(xs)
        mean_x = sum(xs) / n
        mean_z = sum(zs) / n
        num = sum(
            (x - mean_x) * (z - mean_z)
            for x, z in zip(xs, zs)
        )
        den = sum((x - mean_x) ** 2 for x in xs)
        if den == 0:
            continue

        b = num / den
        a = math.exp(mean_z - b * mean_x)
        resid = sum(
            (a * (t**b) + c - y) ** 2 for t, y in pts
        )
        if best is None or resid < best[0]:
            best = (resid, a, b, c)

    if best is None:
        return None
    _, a, b, c = best
    return a, b, c
