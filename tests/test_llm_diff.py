from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autoresearch_rl.policy.interface import DiffProposal
from autoresearch_rl.policy.llm_diff import (
    LLMDiffPolicy,
    _format_diff_prompt,
    _parse_diff_response,
)


SAMPLE_SOURCE = """\
import torch
LEARNING_RATE = 0.0026
EPOCHS = 10

def train():
    pass
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


# --- _format_diff_prompt ---


def test_format_prompt_includes_source():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=[], metric="val_bpb", direction="min",
    )
    assert "Current source (train.py):" in prompt
    assert "LEARNING_RATE = 0.0026" in prompt
    assert "```python" in prompt


def test_format_prompt_includes_program():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=[], metric="val_bpb", direction="min",
        program="Train a small LM on wikitext.",
    )
    assert "Task specification:" in prompt
    assert "Train a small LM on wikitext." in prompt


def test_format_prompt_empty_program_omits_section():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=[], metric="val_bpb", direction="min",
    )
    assert "Task specification:" not in prompt


def test_format_prompt_includes_history():
    history = [
        {"iter": 0, "metrics": {"val_bpb": 1.5}, "status": "ok", "decision": "keep"},
        {"iter": 1, "metrics": {"val_bpb": 1.3}, "status": "ok", "decision": "keep"},
    ]
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=history, metric="val_bpb", direction="min",
    )
    assert "1.5" in prompt
    assert "1.3" in prompt


def test_format_prompt_includes_errors_from_history():
    history = [
        {
            "iter": 0, "metrics": {}, "status": "failed", "decision": "discard",
            "stderr_tail": "CUDA OOM",
        },
    ]
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=history, metric="val_bpb", direction="min",
    )
    assert "CUDA OOM" in prompt


def test_format_prompt_includes_objective_direction():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=[], metric="accuracy", direction="max",
    )
    assert "maximize" in prompt.lower()


def test_format_prompt_diff_instructions():
    prompt = _format_diff_prompt(
        source=SAMPLE_SOURCE, filename="train.py",
        history=[], metric="val_bpb", direction="min",
    )
    assert "--- a/train.py" in prompt
    assert "+++ b/train.py" in prompt


# --- _parse_diff_response ---


def test_parse_valid_diff():
    diff = _parse_diff_response(SAMPLE_DIFF, "train.py")
    assert "--- a/train.py" in diff
    assert "+++ b/train.py" in diff
    assert "@@" in diff


def test_parse_diff_in_markdown_fences():
    raw = "```diff\n" + SAMPLE_DIFF + "\n```"
    diff = _parse_diff_response(raw, "train.py")
    assert "--- a/train.py" in diff
    assert "+LEARNING_RATE = 0.0020" in diff


def test_parse_diff_with_surrounding_text():
    raw = "Here's the change:\n" + SAMPLE_DIFF + "\nThis should help."
    diff = _parse_diff_response(raw, "train.py")
    assert "--- a/train.py" in diff


def test_parse_diff_with_git_diff_header():
    raw = "diff --git a/train.py b/train.py\n" + SAMPLE_DIFF
    diff = _parse_diff_response(raw, "train.py")
    assert "diff --git" in diff


def test_parse_no_diff_raises():
    with pytest.raises(ValueError, match="No unified diff"):
        _parse_diff_response("I think you should try something", "train.py")


def test_parse_incomplete_diff_raises():
    raw = "--- a/train.py\n+++ b/train.py\nsome content"
    with pytest.raises(ValueError, match="missing required sections"):
        _parse_diff_response(raw, "train.py")


# --- LLMDiffPolicy ---


def _mock_urlopen_response(content: str):
    body = json.dumps({
        "choices": [{"message": {"content": content}}]
    }).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


_PATCH = "autoresearch_rl.policy.llm_diff._call_chat_api_messages"


class TestLLMDiffPolicy:

    def _make_policy(self, **kwargs):
        defaults = {
            "mutable_file": "/tmp/test_train.py",
            "api_url": "http://localhost:8000/v1",
            "model": "test-model",
            "api_key_env": "TEST_LLM_KEY",
            "seed": 42,
        }
        defaults.update(kwargs)
        return LLMDiffPolicy(**defaults)

    def test_success(self):
        policy = self._make_policy()
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch(_PATCH, return_value=SAMPLE_DIFF),
        ):
            proposal = policy.propose({
                "history": [],
                "source": SAMPLE_SOURCE,
            })

        assert proposal.diff.strip() != ""
        assert proposal.rationale == "llm-diff"
        assert "--- a/" in proposal.diff

    def test_missing_api_key_falls_back(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = self._make_policy(mutable_file=str(src))
        with patch.dict("os.environ", {}, clear=True):
            proposal = policy.propose({"history": [], "source": SAMPLE_SOURCE})

        # GreedyLLMPolicy returns its own rationale names
        assert proposal.rationale != "llm-diff"
        assert isinstance(proposal, DiffProposal)

    def test_api_error_falls_back(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = self._make_policy(mutable_file=str(src))
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch(_PATCH, side_effect=Exception("timeout")),
        ):
            proposal = policy.propose({
                "history": [],
                "source": SAMPLE_SOURCE,
            })

        assert proposal.rationale != "llm-diff"
        assert isinstance(proposal, DiffProposal)

    def test_no_source_falls_back(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = self._make_policy(mutable_file=str(src))
        with patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}):
            proposal = policy.propose({"history": [], "source": ""})

        assert proposal.rationale != "llm-diff"
        assert isinstance(proposal, DiffProposal)

    def test_parse_error_falls_back(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(SAMPLE_SOURCE)
        policy = self._make_policy(mutable_file=str(src))
        with (
            patch.dict("os.environ", {"TEST_LLM_KEY": "sk-test"}),
            patch(_PATCH, return_value="I cannot help with that"),
        ):
            proposal = policy.propose({
                "history": [],
                "source": SAMPLE_SOURCE,
            })

        assert proposal.rationale != "llm-diff"
        assert isinstance(proposal, DiffProposal)
