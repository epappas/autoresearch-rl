from __future__ import annotations

import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any

from autoresearch_rl.policy.interface import ParamProposal

logger = logging.getLogger(__name__)

_MAX_HISTORY = 50
_MAX_CONVERSATION_PAIRS = 10
_SYSTEM_PROMPT = (
    "You are a hyperparameter optimization assistant. "
    "Given a search space and experiment history, propose the next set of "
    "hyperparameters to try. Respond with ONLY a JSON object mapping "
    "parameter names to values from the allowed choices."
)


def _format_prompt(
    space: dict[str, list[Any]],
    history: list[dict],
    metric: str,
    direction: str,
    program: str = "",
    source: str = "",
    recent_errors: list[str] | None = None,
    recent_logs: list[str] | None = None,
) -> str:
    lines: list[str] = []
    if program:
        lines.append("Task specification:")
        lines.append(program)
        lines.append("")
    lines.append(f"Objective: {direction}imize '{metric}'")
    lines.append("")

    if source:
        lines.append("Training script:")
        lines.append("```python")
        lines.append(source)
        lines.append("```")
        lines.append("")

    lines.append("Search space:")
    for name, values in space.items():
        lines.append(f"  {name}: {values}")
    lines.append("")

    recent = history[-_MAX_HISTORY:]
    if recent:
        lines.append(f"Experiment history (last {len(recent)} of {len(history)}):")
        for entry in recent:
            params = entry.get("params", {})
            metrics = entry.get("metrics", {})
            status = entry.get("status", "unknown")
            val = metrics.get(metric, "N/A")
            lines.append(f"  params={params} -> {metric}={val} (status={status})")
    else:
        lines.append("No experiment history yet. Propose a good starting configuration.")
    lines.append("")

    if recent_errors:
        lines.append("Recent errors:")
        for err in recent_errors:
            lines.append(f"  - {err}")
        lines.append("")

    if recent_logs:
        lines.append("Recent training logs:")
        for log_entry in recent_logs:
            lines.append(f"  {log_entry}")
        lines.append("")

    lines.append(
        "Respond with ONLY a JSON object. "
        "Keys must match the parameter names above. "
        "Values must be from the allowed choices."
    )
    return "\n".join(lines)


def _call_chat_api_messages(
    url: str,
    model: str,
    api_key: str,
    messages: list[dict],
    timeout: int,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> str:
    import time as _time

    endpoint = url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503) and attempt < max_retries:
                base = min(10 * (2 ** attempt), 90)
                jitter = random.uniform(0, base * 0.3)
                wait = base + jitter
                logger.warning(
                    "LLM API error %d, retry %d/%d in %.0fs",
                    e.code, attempt + 1, max_retries, wait,
                )
                _time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt < max_retries:
                wait = min(10 * (2 ** attempt), 90)
                logger.warning(
                    "LLM API network error, retry %d/%d in %.0fs: %s",
                    attempt + 1, max_retries, wait, e,
                )
                _time.sleep(wait)
                continue
            raise

    raise RuntimeError("max retries exhausted")


def _call_chat_api(
    url: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    timeout: int,
) -> str:
    return _call_chat_api_messages(
        url, model, api_key,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout,
    )


def _coerce_value(raw: Any, allowed: list[Any]) -> Any | None:
    """Try to match raw value against allowed values with type coercion."""
    # Direct match first
    if raw in allowed:
        return raw

    # Try numeric coercion
    for candidate in allowed:
        if isinstance(candidate, float):
            try:
                if float(raw) == candidate:
                    return candidate
            except (ValueError, TypeError):
                continue
        elif isinstance(candidate, int) and not isinstance(candidate, bool):
            try:
                if int(float(raw)) == candidate:
                    return candidate
            except (ValueError, TypeError):
                continue
        elif isinstance(candidate, str):
            if str(raw) == candidate:
                return candidate
        elif isinstance(candidate, bool):
            if str(raw).lower() in ("true", "1"):
                if candidate is True:
                    return candidate
            elif str(raw).lower() in ("false", "0"):
                if candidate is False:
                    return candidate

    return None


def _parse_response(raw: str, space: dict[str, list[Any]]) -> dict[str, Any]:
    """Extract JSON from LLM response and validate against space."""
    # Strip markdown fences (handles truncated fences too)
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
    if match:
        inner = match.group(1)
        # Remove closing fence if present
        inner = re.sub(r"\s*```\s*$", "", inner)
        text = inner.strip()

    # Extract JSON: find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict, got {type(parsed).__name__}")

    result: dict[str, Any] = {}
    for key, allowed in space.items():
        if key not in parsed:
            raise ValueError(f"Missing key '{key}' in LLM response")
        coerced = _coerce_value(parsed[key], allowed)
        if coerced is None:
            raise ValueError(
                f"Value {parsed[key]!r} for '{key}' not in allowed: {allowed}"
            )
        result[key] = coerced

    return result


def _random_fallback(
    space: dict[str, list[Any]], rng: random.Random
) -> ParamProposal:
    params = {k: rng.choice(v) for k, v in space.items() if v}
    return ParamProposal(params=params, rationale="llm-fallback-random")


class LLMParamPolicy:
    """Calls an OpenAI-compatible chat API to propose hyperparameters.

    Maintains conversation history across iterations so the LLM builds
    cumulative reasoning about what hyperparameters work and why.
    """

    def __init__(
        self,
        space: dict[str, list[Any]],
        *,
        api_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 30,
        metric: str = "val_bpb",
        direction: str = "min",
        seed: int = 7,
    ):
        self._space = {k: list(v) for k, v in space.items()}
        self._api_url = api_url
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._metric = metric
        self._direction = direction
        self._rng = random.Random(seed)
        self._conversation: list[dict] = []

    def propose(self, state: dict) -> ParamProposal:
        history: list[dict] = state.get("history", [])
        program: str = state.get("program", "")
        source: str = state.get("source", "")
        recent_errors: list[str] | None = state.get("recent_errors")
        recent_logs: list[str] | None = state.get("recent_logs")
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            logger.warning("LLM policy: %s not set, falling back to random", self._api_key_env)
            return _random_fallback(self._space, self._rng)

        try:
            user_msg = _format_prompt(
                self._space,
                history,
                self._metric,
                self._direction,
                program=program,
                source=source,
                recent_errors=recent_errors,
                recent_logs=recent_logs,
            )
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            messages.extend(self._trimmed_conversation())
            messages.append({"role": "user", "content": user_msg})

            raw = _call_chat_api_messages(
                self._api_url, self._model, api_key, messages, self._timeout_s,
            )
            params = _parse_response(raw, self._space)

            self._conversation.append({"role": "user", "content": user_msg})
            self._conversation.append({"role": "assistant", "content": raw})
            self._trim_conversation()

            return ParamProposal(params=params, rationale="llm")
        except Exception:
            logger.warning("LLM policy failed, falling back to random", exc_info=True)
            return _random_fallback(self._space, self._rng)

    def _trimmed_conversation(self) -> list[dict]:
        limit = _MAX_CONVERSATION_PAIRS * 2
        return self._conversation[-limit:]

    def _trim_conversation(self) -> None:
        limit = _MAX_CONVERSATION_PAIRS * 2
        if len(self._conversation) > limit:
            self._conversation = self._conversation[-limit:]
