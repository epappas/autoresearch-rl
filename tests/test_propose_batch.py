"""propose_batch contract: native impls + LLM batch parsing."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from autoresearch_rl.policy.interface import ParamProposal, propose_batch
from autoresearch_rl.policy.llm_search import LLMParamPolicy, _parse_batch_response
from autoresearch_rl.policy.search import GridPolicy, RandomPolicy, StaticPolicy


# ---------------------------------------------------------------- defaults


def test_helper_falls_back_to_propose_loop_when_no_native() -> None:
    """A policy without propose_batch is iterated k times by the helper."""

    class OnlyPropose:
        def __init__(self) -> None:
            self.calls = 0

        def propose(self, state: dict) -> ParamProposal:
            self.calls += 1
            return ParamProposal(params={"i": self.calls})

    p = OnlyPropose()
    out = propose_batch(p, {}, 3)
    assert len(out) == 3
    assert [pp.params["i"] for pp in out] == [1, 2, 3]


def test_helper_returns_empty_for_zero_or_negative() -> None:
    p = StaticPolicy()
    assert propose_batch(p, {}, 0) == []
    assert propose_batch(p, {}, -1) == []


# ---------------------------------------------------------------- native impls


def test_random_propose_batch_is_seeded_reproducible() -> None:
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2], "bs": [16, 32, 64]}
    p1 = RandomPolicy(space, seed=11)
    p2 = RandomPolicy(space, seed=11)
    a = [pp.params for pp in p1.propose_batch({}, 5)]
    b = [pp.params for pp in p2.propose_batch({}, 5)]
    assert a == b


def test_random_serial_and_batch_match_for_same_seed() -> None:
    """Running 4 serial proposals == running propose_batch(4)."""
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2]}
    p_serial = RandomPolicy(space, seed=42)
    p_batch = RandomPolicy(space, seed=42)
    serial = [p_serial.propose({}).params for _ in range(4)]
    batch = [pp.params for pp in p_batch.propose_batch({}, 4)]
    assert serial == batch


def test_grid_propose_batch_advances_through_cells() -> None:
    p = GridPolicy({"a": [1, 2], "b": [10, 20]})
    out = p.propose_batch({}, 4)
    cells = [(pp.params["a"], pp.params["b"]) for pp in out]
    assert sorted(cells) == [(1, 10), (1, 20), (2, 10), (2, 20)]


def test_static_propose_batch_returns_empties() -> None:
    p = StaticPolicy()
    out = p.propose_batch({}, 3)
    assert len(out) == 3
    assert all(pp.params == {} for pp in out)


# ---------------------------------------------------------------- LLM batch parser


def test_parse_batch_response_happy_path() -> None:
    space = {"lr": [1e-5, 1e-4, 1e-3]}
    raw = json.dumps([
        {"lr": 1e-5}, {"lr": 1e-4}, {"lr": 1e-3},
    ])
    out = _parse_batch_response(raw, space, 3)
    assert [pp.params["lr"] for pp in out] == [1e-5, 1e-4, 1e-3]
    assert all(pp.rationale == "llm-batch" for pp in out)


def test_parse_batch_response_strips_json_fence() -> None:
    space = {"lr": [1e-5, 1e-3]}
    raw = "```json\n[{\"lr\": 1e-5}, {\"lr\": 1e-3}]\n```"
    out = _parse_batch_response(raw, space, 2)
    assert len(out) == 2


def test_parse_batch_response_rejects_wrong_count() -> None:
    space = {"lr": [1e-5, 1e-3]}
    raw = json.dumps([{"lr": 1e-5}])
    with pytest.raises(ValueError, match="Expected 2"):
        _parse_batch_response(raw, space, 2)


def test_parse_batch_response_rejects_disallowed_value() -> None:
    space = {"lr": [1e-5, 1e-3]}
    raw = json.dumps([{"lr": 1e-5}, {"lr": 9.99}])
    with pytest.raises(ValueError, match="not in allowed"):
        _parse_batch_response(raw, space, 2)


# ---------------------------------------------------------------- LLM policy batch


def test_llm_batch_returns_distinct_lrs(monkeypatch: pytest.MonkeyPatch) -> None:
    space = {"lr": [1e-5, 1e-4, 1e-3, 1e-2]}
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    canned = json.dumps([
        {"lr": 1e-5}, {"lr": 1e-4}, {"lr": 1e-3}, {"lr": 1e-2},
    ])
    with patch(
        "autoresearch_rl.policy.llm_search._call_chat_api_messages",
        return_value=canned,
    ):
        policy = LLMParamPolicy(
            space, api_url="http://stub", model="stub", metric="loss",
        )
        out = policy.propose_batch({}, 4)
    assert len(out) == 4
    lrs = [pp.params["lr"] for pp in out]
    assert len(set(lrs)) == 4


def test_llm_batch_falls_back_to_random_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = {"lr": [1e-5, 1e-4, 1e-3]}
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    bogus = "not even json"
    with patch(
        "autoresearch_rl.policy.llm_search._call_chat_api_messages",
        return_value=bogus,
    ):
        policy = LLMParamPolicy(
            space, api_url="http://stub", model="stub", seed=7,
        )
        out = policy.propose_batch({}, 4)
    assert len(out) == 4
    # All should still be valid space members.
    for pp in out:
        assert pp.params["lr"] in space["lr"]
    # And they should be flagged as fallback.
    assert all(pp.rationale == "llm-fallback-random" for pp in out)


def test_llm_batch_falls_back_to_random_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = {"lr": [1e-5, 1e-3]}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    policy = LLMParamPolicy(
        space, api_url="http://stub", model="stub", seed=11,
    )
    out = policy.propose_batch({}, 3)
    assert len(out) == 3
    assert all(pp.rationale == "llm-fallback-random" for pp in out)


def test_llm_batch_k_one_uses_single_propose(monkeypatch: pytest.MonkeyPatch) -> None:
    """propose_batch(state, 1) takes the single-proposal path, not batch."""
    space = {"lr": [1e-5, 1e-3]}
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    single = json.dumps({"lr": 1e-3})
    with patch(
        "autoresearch_rl.policy.llm_search._call_chat_api_messages",
        return_value=single,
    ) as mock_call:
        policy = LLMParamPolicy(space, api_url="http://stub", model="stub")
        out = policy.propose_batch({}, 1)
    assert len(out) == 1
    assert out[0].params["lr"] == 1e-3
    assert mock_call.call_count == 1
