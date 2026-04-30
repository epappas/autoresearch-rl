"""Unit tests for shared prompt fragments.

Currently focused on the AR_DISABLE_PROGRESS_SERIES env-var kill switch
used by the Phase A.3 trajectory-aware ablation. Production runs leave
the env var unset; the ablation's Arm B sets it to 1 to compare LLM
proposal quality with vs without progress trajectory context.
"""
from __future__ import annotations

import os

import pytest

from autoresearch_rl.policy._prompt_fragments import (
    DISABLE_PROGRESS_SERIES_ENV,
    render_progress_series,
)


HISTORY_WITH_SERIES = [
    {
        "iter": 0,
        "status": "ok",
        "params": {"lr": 1e-4},
        "metrics": {"eval_score": 0.42},
        "progress_series": [
            {"step": 0, "value": 0.10},
            {"step": 5, "value": 0.30},
            {"step": 10, "value": 0.42},
        ],
    },
    {
        "iter": 1,
        "status": "cancelled",
        "params": {"lr": 1e-2},
        "metrics": {},
        "progress_series": [
            {"step": 0, "value": 0.05},
            {"step": 3, "value": 0.04},
        ],
    },
]


def test_renders_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DISABLE_PROGRESS_SERIES_ENV, raising=False)
    out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")
    assert "PROGRESS TRAJECTORIES" in out
    assert "iter=0" in out
    assert "iter=1" in out
    assert "first=0.1" in out


def test_returns_empty_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_PROGRESS_SERIES_ENV, "1")
    out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")
    assert out == ""


def test_returns_empty_for_truthy_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    for val in ("1", "true", "yes", "on"):
        monkeypatch.setenv(DISABLE_PROGRESS_SERIES_ENV, val)
        assert render_progress_series(HISTORY_WITH_SERIES, "eval_score") == ""


def test_renders_when_env_set_to_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty string is the only falsy override that we accept as 'enabled'.

    `os.environ.get("X") == ""` is treated as 'unset' by Python's truthiness
    on the env-var read path, so the rendering proceeds. This matches the
    existing AR_PROGRESS_FILE / AR_CONTROL_FILE convention in target/progress.py
    where any non-empty value is the activation.
    """
    monkeypatch.setenv(DISABLE_PROGRESS_SERIES_ENV, "")
    out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")
    assert "PROGRESS TRAJECTORIES" in out


def test_idempotent_under_repeated_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Toggling the env var between calls flips behavior immediately.

    The kill switch reads the env var on each call so an experiment runner
    that toggles it between iterations gets the right rendering each time.
    """
    monkeypatch.delenv(DISABLE_PROGRESS_SERIES_ENV, raising=False)
    enabled_out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")
    monkeypatch.setenv(DISABLE_PROGRESS_SERIES_ENV, "1")
    disabled_out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")
    monkeypatch.delenv(DISABLE_PROGRESS_SERIES_ENV, raising=False)
    re_enabled_out = render_progress_series(HISTORY_WITH_SERIES, "eval_score")

    assert enabled_out == re_enabled_out
    assert disabled_out == ""
    assert enabled_out != disabled_out


def test_preserves_existing_no_series_behavior() -> None:
    """When history has no progress_series, render returns "" regardless of env."""
    plain_history = [{"iter": 0, "status": "ok", "params": {}, "metrics": {}}]
    os.environ.pop(DISABLE_PROGRESS_SERIES_ENV, None)
    assert render_progress_series(plain_history, "eval_score") == ""
