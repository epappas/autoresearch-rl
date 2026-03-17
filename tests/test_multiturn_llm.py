"""Tests for multi-turn conversation in LLMParamPolicy and LLMDiffPolicy."""
from __future__ import annotations

from unittest.mock import patch

from autoresearch_rl.policy.interface import DiffProposal, ParamProposal
from autoresearch_rl.policy.llm_diff import LLMDiffPolicy, _MAX_CONVERSATION_PAIRS
from autoresearch_rl.policy.llm_search import LLMParamPolicy

SAMPLE_SOURCE = """\
import torch
LEARNING_RATE = 0.0026
EPOCHS = 10
"""

SAMPLE_DIFF = """\
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
 import torch
-LEARNING_RATE = 0.0026
+LEARNING_RATE = 0.0020
 EPOCHS = 10
"""

SAMPLE_PARAMS_JSON = '{"lr": [0.001, 0.01]}'


# ---- helpers ----------------------------------------------------------------


def _make_param_policy(**kw) -> LLMParamPolicy:
    defaults = dict(
        space={"lr": [0.001, 0.01]},
        api_url="http://localhost:8000/v1",
        model="test-model",
        api_key_env="TEST_KEY",
        seed=42,
    )
    defaults.update(kw)
    return LLMParamPolicy(**defaults)


def _make_diff_policy(**kw) -> LLMDiffPolicy:
    defaults = dict(
        mutable_file="/tmp/test_train.py",
        api_url="http://localhost:8000/v1",
        model="test-model",
        api_key_env="TEST_KEY",
        seed=42,
    )
    defaults.update(kw)
    return LLMDiffPolicy(**defaults)


_PARAM_API = "autoresearch_rl.policy.llm_search._call_chat_api_messages"
_DIFF_API = "autoresearch_rl.policy.llm_diff._call_chat_api_messages"

_GOOD_PARAM_RESPONSE = '{"lr": 0.001}'


# ---- LLMParamPolicy ---------------------------------------------------------


class TestLLMParamPolicyMultiTurn:

    def test_conversation_starts_empty(self):
        policy = _make_param_policy()
        assert policy._conversation == []

    def test_successful_propose_adds_pair(self):
        policy = _make_param_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE),
        ):
            policy.propose({"history": [], "program": ""})

        assert len(policy._conversation) == 2
        assert policy._conversation[0]["role"] == "user"
        assert policy._conversation[1]["role"] == "assistant"
        assert policy._conversation[1]["content"] == _GOOD_PARAM_RESPONSE

    def test_second_call_sends_prior_conversation_as_context(self):
        policy = _make_param_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE) as mock_api,
        ):
            policy.propose({"history": [], "program": ""})
            policy.propose({"history": [], "program": ""})

        # Second call's messages list must include the first pair
        second_call_messages = mock_api.call_args_list[1][0][3]
        roles = [m["role"] for m in second_call_messages]
        assert roles.count("user") >= 2
        assert roles.count("assistant") >= 1

    def test_conversation_grows_across_iterations(self):
        policy = _make_param_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE),
        ):
            for _ in range(3):
                policy.propose({"history": [], "program": ""})

        assert len(policy._conversation) == 6  # 3 user + 3 assistant

    def test_conversation_trimmed_at_max_pairs(self):
        policy = _make_param_policy()
        limit = _MAX_CONVERSATION_PAIRS + 2
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE),
        ):
            for _ in range(limit):
                policy.propose({"history": [], "program": ""})

        assert len(policy._conversation) == _MAX_CONVERSATION_PAIRS * 2

    def test_fallback_leaves_conversation_unchanged(self):
        policy = _make_param_policy()
        # First succeed to populate conversation
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE),
        ):
            policy.propose({"history": [], "program": ""})

        before = len(policy._conversation)

        # Then fail (no api key)
        with patch.dict("os.environ", {}, clear=True):
            policy.propose({"history": [], "program": ""})

        assert len(policy._conversation) == before

    def test_api_error_fallback_leaves_conversation_unchanged(self):
        policy = _make_param_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, side_effect=Exception("network error")),
        ):
            result = policy.propose({"history": [], "program": ""})

        assert isinstance(result, ParamProposal)
        assert result.rationale == "llm-fallback-random"
        assert policy._conversation == []

    def test_second_propose_message_includes_system_as_first(self):
        policy = _make_param_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_PARAM_API, return_value=_GOOD_PARAM_RESPONSE) as mock_api,
        ):
            policy.propose({"history": [], "program": ""})
            policy.propose({"history": [], "program": ""})

        second_messages = mock_api.call_args_list[1][0][3]
        assert second_messages[0]["role"] == "system"


# ---- LLMDiffPolicy ----------------------------------------------------------


class TestLLMDiffPolicyMultiTurn:

    def test_conversation_starts_empty(self):
        policy = _make_diff_policy()
        assert policy._conversation == []

    def test_successful_propose_adds_pair(self):
        policy = _make_diff_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value=SAMPLE_DIFF),
        ):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert proposal.rationale == "llm-diff"
        assert len(policy._conversation) == 2
        assert policy._conversation[0]["role"] == "user"
        assert policy._conversation[1]["role"] == "assistant"

    def test_second_propose_includes_prior_pair(self):
        policy = _make_diff_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value=SAMPLE_DIFF) as mock_api,
        ):
            policy.propose({"history": [], "source": SAMPLE_SOURCE})
            policy.propose({"history": [], "source": SAMPLE_SOURCE})

        second_messages = mock_api.call_args_list[1][0][3]
        roles = [m["role"] for m in second_messages]
        assert roles.count("user") >= 2
        assert roles.count("assistant") >= 1

    def test_correction_retry_on_invalid_diff(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))

        responses = iter(["not a diff at all", SAMPLE_DIFF])

        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, side_effect=lambda *a, **kw: next(responses)) as mock_api,
        ):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert proposal.rationale == "llm-diff"
        assert mock_api.call_count == 2

        # Only the successful pair is committed to conversation
        assert len(policy._conversation) == 2

    def test_correction_message_appended_between_retries(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))

        responses = iter(["bad", SAMPLE_DIFF])

        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, side_effect=lambda *a, **kw: next(responses)) as mock_api,
        ):
            policy.propose({"history": [], "source": SAMPLE_SOURCE})

        # Second call's messages must contain a correction user message
        second_messages = mock_api.call_args_list[1][0][3]
        user_messages = [m["content"] for m in second_messages if m["role"] == "user"]
        correction_found = any("invalid" in m.lower() or "invalid" in m for m in user_messages)
        assert correction_found, f"No correction message found: {user_messages}"

    def test_all_retries_exhausted_falls_back_to_greedy(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))

        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value="not a diff"),
        ):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert isinstance(proposal, DiffProposal)
        assert proposal.rationale != "llm-diff"
        # Conversation must NOT be polluted with failed attempts
        assert policy._conversation == []

    def test_api_error_on_all_retries_falls_back(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))

        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, side_effect=Exception("network error")),
        ):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert isinstance(proposal, DiffProposal)
        assert proposal.rationale != "llm-diff"
        assert policy._conversation == []

    def test_reset_conversation_clears_state(self):
        policy = _make_diff_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value=SAMPLE_DIFF),
        ):
            policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert len(policy._conversation) == 2
        policy.reset_conversation()
        assert policy._conversation == []

    def test_no_source_falls_back_immediately(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))
        with patch.dict("os.environ", {"TEST_KEY": "sk-test"}):
            proposal = policy.propose({"history": [], "source": ""})

        assert proposal.rationale != "llm-diff"
        assert policy._conversation == []

    def test_no_api_key_falls_back_immediately(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = _make_diff_policy(mutable_file=str(src))
        with patch.dict("os.environ", {}, clear=True):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert proposal.rationale != "llm-diff"
        assert policy._conversation == []

    def test_conversation_trimmed_at_max_pairs(self):
        policy = _make_diff_policy()
        limit = _MAX_CONVERSATION_PAIRS + 2
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value=SAMPLE_DIFF),
        ):
            for _ in range(limit):
                policy.propose({"history": [], "source": SAMPLE_SOURCE})

        assert len(policy._conversation) == _MAX_CONVERSATION_PAIRS * 2

    def test_system_message_is_first_in_api_call(self):
        policy = _make_diff_policy()
        with (
            patch.dict("os.environ", {"TEST_KEY": "sk-test"}),
            patch(_DIFF_API, return_value=SAMPLE_DIFF) as mock_api,
        ):
            policy.propose({"history": [], "source": SAMPLE_SOURCE})

        messages = mock_api.call_args[0][3]
        assert messages[0]["role"] == "system"
