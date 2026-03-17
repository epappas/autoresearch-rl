from __future__ import annotations

import json
import random
from unittest.mock import MagicMock, patch

import pytest

from autoresearch_rl.policy.llm_search import (
    LLMParamPolicy,
    _coerce_value,
    _format_prompt,
    _parse_response,
    _random_fallback,
)


SPACE = {"learning_rate": [0.00001, 0.00002, 0.00003], "epochs": [1, 2, 3]}


# --- _format_prompt ---


def test_format_prompt_with_program():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min", program="Train a small LM on wikitext.")
    assert prompt.startswith("Task specification:")
    assert "Train a small LM on wikitext." in prompt
    assert "Objective:" in prompt
    # program comes before objective
    assert prompt.index("Task specification:") < prompt.index("Objective:")


def test_format_prompt_empty_program_omits_section():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min", program="")
    assert "Task specification:" not in prompt
    prompt2 = _format_prompt(SPACE, [], "val_bpb", "min")
    assert "Task specification:" not in prompt2


def test_format_prompt_with_history():
    history = [
        {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": 1.5}, "status": "ok"},
        {"params": {"learning_rate": 0.00002, "epochs": 2}, "metrics": {"val_bpb": 1.3}, "status": "ok"},
    ]
    prompt = _format_prompt(SPACE, history, "val_bpb", "min")
    assert "minimize" in prompt.lower()
    assert "learning_rate" in prompt
    assert "1.5" in prompt
    assert "1.3" in prompt
    assert "last 2 of 2" in prompt


def test_format_prompt_empty_history():
    prompt = _format_prompt(SPACE, [], "val_bpb", "min")
    assert "No experiment history" in prompt
    assert "learning_rate" in prompt


def test_format_prompt_truncation():
    history = [
        {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": float(i)}, "status": "ok"}
        for i in range(100)
    ]
    prompt = _format_prompt(SPACE, history, "val_bpb", "min")
    assert "last 50 of 100" in prompt


def test_format_prompt_max_direction():
    prompt = _format_prompt(SPACE, [], "accuracy", "max")
    assert "maximize" in prompt.lower()


# --- _parse_response ---


def test_parse_valid_json():
    raw = '{"learning_rate": 0.00002, "epochs": 2}'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00002, "epochs": 2}


def test_parse_markdown_fences():
    raw = '```json\n{"learning_rate": 0.00001, "epochs": 3}\n```'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00001, "epochs": 3}


def test_parse_markdown_fences_no_lang():
    raw = '```\n{"learning_rate": 0.00003, "epochs": 1}\n```'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00003, "epochs": 1}


def test_parse_string_coercion():
    raw = '{"learning_rate": "0.00002", "epochs": "2"}'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00002, "epochs": 2}


def test_parse_missing_key():
    raw = '{"learning_rate": 0.00002}'
    with pytest.raises(ValueError, match="Missing key"):
        _parse_response(raw, SPACE)


def test_parse_invalid_value():
    raw = '{"learning_rate": 0.99, "epochs": 2}'
    with pytest.raises(ValueError, match="not in allowed"):
        _parse_response(raw, SPACE)


def test_parse_no_json():
    with pytest.raises(ValueError, match="No JSON object"):
        _parse_response("I think you should try these params", SPACE)


def test_parse_surrounding_text():
    raw = 'Here is my suggestion:\n{"learning_rate": 0.00001, "epochs": 1}\nGood luck!'
    result = _parse_response(raw, SPACE)
    assert result == {"learning_rate": 0.00001, "epochs": 1}


# --- _coerce_value ---


def test_coerce_direct_match():
    assert _coerce_value(0.00002, [0.00001, 0.00002]) == 0.00002


def test_coerce_string_to_float():
    assert _coerce_value("0.00002", [0.00001, 0.00002]) == 0.00002


def test_coerce_string_to_int():
    assert _coerce_value("2", [1, 2, 3]) == 2


def test_coerce_float_to_int():
    assert _coerce_value(2.0, [1, 2, 3]) == 2


def test_coerce_bool():
    assert _coerce_value("true", [True, False]) is True
    assert _coerce_value("false", [True, False]) is False


def test_coerce_no_match():
    assert _coerce_value(999, [1, 2, 3]) is None


def test_coerce_string_values():
    assert _coerce_value("adam", ["adam", "sgd"]) == "adam"


# --- _random_fallback ---


def test_random_fallback_reproducible():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    p1 = _random_fallback(SPACE, rng1)
    p2 = _random_fallback(SPACE, rng2)
    assert p1.params == p2.params
    assert p1.rationale == "llm-fallback-random"


def test_random_fallback_values_in_space():
    rng = random.Random(0)
    for _ in range(20):
        p = _random_fallback(SPACE, rng)
        for k, v in p.params.items():
            assert v in SPACE[k]


# --- LLMParamPolicy.next ---


def _mock_urlopen_response(content: str):
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestLLMParamPolicyNext:

    def _make_policy(self, **kwargs):
        defaults = {
            "api_url": "http://localhost:8000/v1",
            "model": "test-model",
            "api_key_env": "TEST_LLM_KEY",
            "seed": 42,
        }
        defaults.update(kwargs)
        return LLMParamPolicy(SPACE, **defaults)

    def test_success(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": []})

        assert proposal.params == {"learning_rate": 0.00002, "epochs": 2}
        assert proposal.rationale == "llm"

    def test_api_error_falls_back(self):
        policy = self._make_policy()
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.side_effect = Exception("API timeout")
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"
        for k, v in proposal.params.items():
            assert v in SPACE[k]

    def test_parse_error_falls_back(self):
        policy = self._make_policy()
        response = "I cannot help with that"
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"

    def test_missing_api_key_falls_back(self):
        policy = self._make_policy()
        with patch.dict("os.environ", {}, clear=True):
            proposal = policy.propose({"history": []})

        assert proposal.rationale == "llm-fallback-random"

    def test_api_call_sends_correct_payload(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00001, "epochs": 1}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": []})

            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "http://localhost:8000/v1/chat/completions"
            assert req.get_header("Authorization") == "Bearer sk-test"
            assert req.get_header("Content-type") == "application/json"
            body = json.loads(req.data)
            assert body["model"] == "test-model"
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"

    def test_with_history(self):
        policy = self._make_policy()
        history = [
            {"params": {"learning_rate": 0.00001, "epochs": 1}, "metrics": {"val_bpb": 1.5}, "status": "ok"},
        ]
        response = '{"learning_rate": 0.00003, "epochs": 3}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            proposal = policy.propose({"history": history})

        assert proposal.params == {"learning_rate": 0.00003, "epochs": 3}

    def test_program_flows_to_api_prompt(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": [], "program": "Minimize perplexity on wikitext."})

            req = mock_urlopen.call_args[0][0]
            body = json.loads(req.data)
            user_msg = body["messages"][1]["content"]
            assert "Task specification:" in user_msg
            assert "Minimize perplexity on wikitext." in user_msg

    def test_no_program_omits_section_in_api(self):
        policy = self._make_policy()
        response = '{"learning_rate": 0.00002, "epochs": 2}'
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch("autoresearch_rl.policy.llm_search.urllib.request.urlopen")
            as mock_urlopen,
        ):
            mock_urlopen.return_value = _mock_urlopen_response(response)
            policy.propose({"history": []})

            req = mock_urlopen.call_args[0][0]
            body = json.loads(req.data)
            user_msg = body["messages"][1]["content"]
            assert "Task specification:" not in user_msg

    def test_fallback_randomness_advances(self):
        """Two consecutive fallbacks should return different params (most of the time)."""
        policy = self._make_policy()
        results = []
        with patch.dict("os.environ", {}, clear=True):
            for _ in range(10):
                results.append(policy.propose({"history": []}).params)
        # With 10 draws from a small space, we should see at least 2 distinct combos
        unique = {tuple(sorted(r.items())) for r in results}
        assert len(unique) >= 2
