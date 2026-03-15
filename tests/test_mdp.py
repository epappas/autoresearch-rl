from __future__ import annotations

import pytest

from autoresearch_rl.mdp import Action, Reward, State, build_state, compute_reward


class TestState:
    def test_construction(self) -> None:
        s = State(
            code_hash="abc123",
            history=({"step": 1},),
            metrics={"loss": 0.5},
            resource_budget=100.0,
            iteration=0,
        )
        assert s.code_hash == "abc123"
        assert s.history == ({"step": 1},)
        assert s.metrics == {"loss": 0.5}
        assert s.resource_budget == 100.0
        assert s.iteration == 0

    def test_immutability(self) -> None:
        s = State(
            code_hash="abc",
            history=(),
            metrics={},
            resource_budget=10.0,
            iteration=0,
        )
        with pytest.raises(AttributeError):
            s.code_hash = "xyz"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            s.iteration = 5  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs: dict = dict(
            code_hash="h", history=(), metrics={}, resource_budget=1.0, iteration=0
        )
        assert State(**kwargs) == State(**kwargs)


class TestAction:
    def test_params_only(self) -> None:
        a = Action(params={"lr": 0.01})
        assert a.params == {"lr": 0.01}
        assert a.diff is None
        assert a.rationale == ""

    def test_diff_only(self) -> None:
        a = Action(diff="--- a\n+++ b\n")
        assert a.diff == "--- a\n+++ b\n"
        assert a.params is None

    def test_both(self) -> None:
        a = Action(params={"lr": 0.01}, diff="patch", rationale="test both")
        assert a.params == {"lr": 0.01}
        assert a.diff == "patch"
        assert a.rationale == "test both"

    def test_immutability(self) -> None:
        a = Action(params={"lr": 0.01})
        with pytest.raises(AttributeError):
            a.rationale = "new"  # type: ignore[misc]


class TestReward:
    def test_value_and_components(self) -> None:
        r = Reward(value=0.5, components={"score_delta": 0.5})
        assert r.value == 0.5
        assert r.components == {"score_delta": 0.5}

    def test_default_components(self) -> None:
        r = Reward(value=1.0)
        assert r.components == {}

    def test_immutability(self) -> None:
        r = Reward(value=0.0)
        with pytest.raises(AttributeError):
            r.value = 1.0  # type: ignore[misc]


class TestBuildState:
    def test_converts_list_to_tuple(self) -> None:
        history_list = [{"a": 1}, {"b": 2}]
        s = build_state(
            code_hash="h",
            history=history_list,
            metrics={"m": 1.0},
            resource_budget=50.0,
            iteration=3,
        )
        assert isinstance(s.history, tuple)
        assert s.history == ({"a": 1}, {"b": 2})
        assert s.iteration == 3

    def test_copies_metrics(self) -> None:
        original = {"loss": 0.3}
        s = build_state(
            code_hash="h",
            history=[],
            metrics=original,
            resource_budget=10.0,
            iteration=0,
        )
        original["loss"] = 999.0
        assert s.metrics["loss"] == 0.3


class TestComputeReward:
    def test_improvement(self) -> None:
        r = compute_reward(prev_score=1.0, curr_score=0.5, status="kept")
        assert r.value == pytest.approx(0.5)
        assert r.components["score_delta"] == pytest.approx(0.5)
        assert "status_penalty" not in r.components

    def test_regression(self) -> None:
        r = compute_reward(prev_score=0.5, curr_score=1.0, status="discarded")
        assert r.value == pytest.approx(-0.5)
        assert r.components["score_delta"] == pytest.approx(-0.5)
        assert "status_penalty" not in r.components

    def test_failure_penalty(self) -> None:
        r = compute_reward(prev_score=1.0, curr_score=1.0, status="failed")
        assert r.value == pytest.approx(-0.8)
        assert r.components["score_delta"] == pytest.approx(0.0)
        assert r.components["status_penalty"] == pytest.approx(-0.8)

    def test_timeout_penalty(self) -> None:
        r = compute_reward(prev_score=1.0, curr_score=0.8, status="timeout")
        assert r.value == pytest.approx(0.2 - 0.8)
        assert r.components["status_penalty"] == pytest.approx(-0.8)

    def test_rejected_penalty(self) -> None:
        r = compute_reward(prev_score=1.0, curr_score=1.0, status="rejected")
        assert r.value == pytest.approx(-0.8)

    def test_custom_fail_penalty(self) -> None:
        r = compute_reward(prev_score=1.0, curr_score=1.0, status="failed", fail_penalty=2.0)
        assert r.value == pytest.approx(-2.0)
        assert r.components["status_penalty"] == pytest.approx(-2.0)
