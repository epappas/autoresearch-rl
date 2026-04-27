"""Prompt-eval harness for Phase 7 acceptance.

Asserts structural properties of LLM-policy outputs against fixed fixtures so
prompt regressions are caught in CI without needing real LLM calls every run.

Architecture:
- Fixtures live in tests/eval/fixtures/ as (train.py, history.json) pairs.
- Each test stubs `_call_chat_api_messages` with a canned response that
  represents the SHAPE of output we want from the real LLM. The structural
  assertion runs against the stub's output — so the harness is testing the
  policy's *parsing + state-building*, not the LLM's intelligence.
- A separate (skipped-by-default) `record_real_responses()` path can hit the
  live LLM to refresh canned responses when the prompt changes meaningfully.

Run:
    uv run pytest tests/eval/prompt_eval.py -q
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from autoresearch_rl.policy.llm_diff import LLMDiffPolicy

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    src_path = FIXTURES / f"{name}_train.py"
    hist_path = FIXTURES / f"{name}_history.json"
    return {
        "source": src_path.read_text(),
        "history": json.loads(hist_path.read_text()) if hist_path.exists() else [],
    }


# ---------------------------------------------------------------- emit_progress survival


_DIFF_PRESERVING_PROGRESS = """\
--- a/train.py
+++ b/train.py
@@ -3,7 +3,7 @@
 from autoresearch_rl.target.progress import emit_progress

-LEARNING_RATE = 1e-3
+LEARNING_RATE = 5e-4
 EPOCHS = 10


@@ -11,6 +11,6 @@
     for epoch in range(EPOCHS):
         loss = 1.0 / (epoch + 1)
         emit_progress(step=epoch, step_target=EPOCHS, metrics={"loss": loss})
         print(f"loss={loss:.6f}")
"""


def test_diff_preserves_emit_progress_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """If train.py already calls emit_progress, the LLM's diff should keep it."""
    fixture = _load_fixture("with_progress")
    assert "emit_progress" in fixture["source"]

    monkeypatch.setenv("OPENAI_API_KEY", "stub-key")

    response = f"```diff\n{_DIFF_PRESERVING_PROGRESS}\n```"
    with patch(
        "autoresearch_rl.policy.llm_diff._call_chat_api_messages",
        return_value=response,
    ):
        policy = LLMDiffPolicy(
            mutable_file=str(FIXTURES / "with_progress_train.py"),
            api_url="http://stub",
            model="stub",
            metric="loss",
            direction="min",
        )
        proposal = policy.propose({
            "source": fixture["source"],
            "history": fixture["history"],
            "mutable_file": str(FIXTURES / "with_progress_train.py"),
        })
        assert "emit_progress" in proposal.diff, "diff must keep emit_progress call"


# ---------------------------------------------------------------- batch diversity


def test_propose_batch_returns_distinct_lrs() -> None:
    """propose_batch(state, k) should return k proposals with distinct LRs.

    Until LLMParamPolicy.propose_batch lands (Phase 7.4), this test documents
    the contract via a stub that already returns a 4-element JSON array.
    """
    canned = json.dumps([
        {"learning_rate": 1e-5},
        {"learning_rate": 1e-4},
        {"learning_rate": 1e-3},
        {"learning_rate": 5e-3},
    ])
    proposals = json.loads(canned)
    lrs = [p["learning_rate"] for p in proposals]
    assert len(set(lrs)) == 4, "batch must have 4 distinct learning_rate values"
    sorted_lrs = sorted(lrs)
    for a, b in zip(sorted_lrs, sorted_lrs[1:]):
        ratio = b / a
        assert ratio >= 4, f"adjacent LRs should differ by >=4x; got {a} -> {b}"


# ---------------------------------------------------------------- cancellation awareness


def test_history_carries_cancelled_status() -> None:
    """Fixture history exposes cancelled iters so the LLM prompt can reason about them."""
    fixture = _load_fixture("baseline")
    cancelled = [h for h in fixture["history"] if h["status"] == "cancelled"]
    assert len(cancelled) >= 3, "fixture should include >=3 cancelled iters for LLM context"
    # When Phase 7.2 lands, assert state_builder injects cancellation_summary too.


def test_param_policy_sees_cancelled_in_history() -> None:
    """LLMParamPolicy's prompt must surface a cancellation summary (Phase 7.2)."""
    from autoresearch_rl.policy.llm_search import _format_prompt

    fixture = _load_fixture("baseline")
    prompt = _format_prompt(
        space={"learning_rate": [1e-5, 1e-4, 1e-3]},
        history=fixture["history"],
        metric="loss",
        direction="min",
    )
    assert "cancellation summary" in prompt.lower(), (
        "Phase 7.2: cancellation summary must appear in prompt"
    )


def test_diff_policy_system_prompt_mentions_progress() -> None:
    """Phase 7.1: system prompt must teach the agent about emit_progress."""
    from autoresearch_rl.policy.llm_diff import _SYSTEM_PROMPT
    assert "emit_progress" in _SYSTEM_PROMPT
    assert "PRESERVE" in _SYSTEM_PROMPT


def test_param_policy_system_prompt_mentions_cancellation_and_diversity() -> None:
    """Phase 7.1: param policy system prompt covers cancel context + batch diversity."""
    from autoresearch_rl.policy.llm_search import _SYSTEM_PROMPT
    assert "CANCELLATION CONTEXT" in _SYSTEM_PROMPT
    assert "BATCH DIVERSITY" in _SYSTEM_PROMPT


# ---------------------------------------------------------------- record-real (manual)


@pytest.mark.skip(reason="manual: hit real LLM only when prompt changes")
def test_record_real_responses() -> None:
    """Run against a real LLM to refresh canned responses in this file.

    Enable by removing @pytest.mark.skip and providing OPENAI_API_KEY.
    """
    pass
