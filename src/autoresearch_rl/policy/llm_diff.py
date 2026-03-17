"""LLM-powered diff policy for code modification proposals.

Sends the current training script source, experiment history, and task
description to an LLM via a persistent multi-turn conversation. The LLM
builds cumulative reasoning across iterations. On validation failure the
policy sends a correction request and retries before falling back to
GreedyLLMPolicy.
"""
from __future__ import annotations

import logging
import os
import re

from autoresearch_rl.policy.interface import DiffProposal
from autoresearch_rl.policy.llm_context import (
    extract_recent_errors,
    extract_recent_logs,
    format_history_section,
)
from autoresearch_rl.policy.llm_search import _call_chat_api_messages
from autoresearch_rl.sandbox.validator import validate_diff

logger = logging.getLogger(__name__)

_MAX_CONVERSATION_PAIRS = 10
_MAX_CORRECTION_RETRIES = 2

_SYSTEM_PROMPT = (
    "You are a code optimization assistant. "
    "Given a training script, experiment history, and task description, "
    "propose a code modification as a unified diff. "
    "Respond with ONLY a valid unified diff (starting with --- a/ and +++ b/). "
    "Make targeted, minimal changes to improve the objective metric."
)


def _format_diff_prompt(
    source: str,
    filename: str,
    history: list[dict],
    metric: str,
    direction: str,
    program: str = "",
) -> str:
    lines: list[str] = []
    if program:
        lines.append("Task specification:")
        lines.append(program)
        lines.append("")

    lines.append(f"Objective: {direction}imize '{metric}'")
    lines.append("")

    lines.append(f"Current source ({filename}):")
    lines.append("```python")
    lines.append(source)
    lines.append("```")
    lines.append("")

    history_section = format_history_section(history, metric)
    lines.append(history_section)
    lines.append("")

    recent_errors = extract_recent_errors(history)
    if recent_errors:
        lines.append("Recent errors:")
        for err in recent_errors:
            lines.append(f"  - {err}")
        lines.append("")

    recent_logs = extract_recent_logs(history)
    if recent_logs:
        lines.append("Recent training logs:")
        for log_entry in recent_logs:
            lines.append(f"  {log_entry}")
        lines.append("")

    lines.append(
        f"Respond with ONLY a unified diff. "
        f"Use '--- a/{filename}' and '+++ b/{filename}' as file paths. "
        f"Make targeted, minimal changes to improve {metric}."
    )
    return "\n".join(lines)


def _parse_diff_response(raw: str, filename: str) -> str:
    """Extract unified diff from LLM response."""
    text = raw.strip()

    # Strip markdown fences
    match = re.search(r"```(?:diff)?\s*\n?(.*)", text, re.DOTALL)
    if match:
        inner = match.group(1)
        inner = re.sub(r"\s*```\s*$", "", inner)
        text = inner.strip()

    # Find diff start
    diff_start: int | None = None
    text_lines = text.splitlines()
    for i, line in enumerate(text_lines):
        if line.startswith("---") or line.startswith("diff --git"):
            diff_start = i
            break

    if diff_start is None:
        raise ValueError(f"No unified diff found in response: {raw[:200]}")

    diff_lines = text_lines[diff_start:]
    diff_text = "\n".join(diff_lines) + "\n"

    has_minus = any(ln.startswith("---") for ln in diff_lines)
    has_plus = any(ln.startswith("+++") for ln in diff_lines)
    has_hunk = any(ln.startswith("@@") for ln in diff_lines)

    if not (has_minus and has_plus and has_hunk):
        raise ValueError("Diff missing required sections (---, +++, or @@)")

    return diff_text


class LLMDiffPolicy:
    """Calls an OpenAI-compatible chat API to propose code diffs.

    Maintains a multi-turn conversation across iterations so the LLM
    accumulates context about what changes worked. On validation failure,
    sends a correction request and retries up to _MAX_CORRECTION_RETRIES
    times before falling back to GreedyLLMPolicy.
    """

    def __init__(
        self,
        *,
        mutable_file: str,
        api_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 60,
        metric: str = "val_bpb",
        direction: str = "min",
        seed: int = 7,
    ):
        self._mutable_file = mutable_file
        self._api_url = api_url
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._metric = metric
        self._direction = direction
        self._filename = os.path.basename(mutable_file)
        self._conversation: list[dict] = []

    def propose(self, state: dict) -> DiffProposal:
        history: list[dict] = state.get("history", [])
        program: str = state.get("program", "")
        source: str = state.get("source", "")

        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            logger.warning(
                "LLM diff policy: %s not set, falling back to greedy",
                self._api_key_env,
            )
            return self._greedy_fallback()

        if not source:
            logger.warning("LLM diff policy: no source in state, falling back to greedy")
            return self._greedy_fallback()

        user_msg = _format_diff_prompt(
            source=source,
            filename=self._filename,
            history=history,
            metric=self._metric,
            direction=self._direction,
            program=program,
        )

        # Build local messages for this attempt (includes conversation + new user msg).
        # The local list may grow with correction messages on retry; _conversation
        # only stores clean successful (user, assistant) pairs.
        messages: list[dict] = list(self._trimmed_conversation())
        messages.append({"role": "user", "content": user_msg})

        raw = ""
        for attempt in range(_MAX_CORRECTION_RETRIES + 1):
            try:
                full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + messages
                raw = _call_chat_api_messages(
                    self._api_url, self._model, api_key,
                    full_messages, self._timeout_s, max_tokens=4096,
                )
                diff = _parse_diff_response(raw, self._filename)
                result = validate_diff(diff)
                if not result.ok:
                    raise ValueError(f"diff validation failed: {result.reason}")

                # Success: commit to conversation (clean pair only)
                self._conversation.append({"role": "user", "content": user_msg})
                self._conversation.append({"role": "assistant", "content": raw})
                self._trim_conversation()
                return DiffProposal(diff=diff, rationale="llm-diff")

            except Exception as exc:
                logger.debug(
                    "LLM diff attempt %d/%d failed: %s",
                    attempt + 1, _MAX_CORRECTION_RETRIES + 1, exc,
                )
                if attempt < _MAX_CORRECTION_RETRIES:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"That response was invalid: {exc}. "
                            f"Please provide a correct unified diff for {self._filename}."
                        ),
                    })

        logger.warning(
            "LLM diff policy failed after %d attempts, falling back to greedy",
            _MAX_CORRECTION_RETRIES + 1,
        )
        return self._greedy_fallback()

    def reset_conversation(self) -> None:
        """Clear conversation history (use when starting a new experiment)."""
        self._conversation.clear()

    def _trimmed_conversation(self) -> list[dict]:
        limit = _MAX_CONVERSATION_PAIRS * 2
        return self._conversation[-limit:]

    def _trim_conversation(self) -> None:
        limit = _MAX_CONVERSATION_PAIRS * 2
        if len(self._conversation) > limit:
            self._conversation = self._conversation[-limit:]

    def _greedy_fallback(self) -> DiffProposal:
        """Delegate to GreedyLLMPolicy as a fallback."""
        from autoresearch_rl.policy.baselines import GreedyLLMPolicy

        greedy = GreedyLLMPolicy()
        state = {"mutable_file": self._mutable_file, "workdir": "."}
        try:
            return greedy.propose(state)
        except Exception:
            logger.warning("Greedy fallback also failed", exc_info=True)
            return DiffProposal(diff="", rationale="llm-diff-fallback-empty")
