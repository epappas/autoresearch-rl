"""Tests for sandbox.validator.validate_required_calls (Phase 7.3)."""
from __future__ import annotations

from autoresearch_rl.sandbox.validator import validate_required_calls


PRE_WITH_PROGRESS = """\
from autoresearch_rl.target.progress import emit_progress

LR = 1e-3

def train():
    for i in range(10):
        emit_progress(step=i, step_target=10, metrics={"loss": 1.0/(i+1)})
"""


def test_passes_when_call_preserved() -> None:
    post = PRE_WITH_PROGRESS.replace("LR = 1e-3", "LR = 5e-4")
    result = validate_required_calls(PRE_WITH_PROGRESS, post, ["emit_progress"])
    assert result.ok is True


def test_rejects_when_call_stripped() -> None:
    post = """\
from autoresearch_rl.target.progress import emit_progress

LR = 1e-3

def train():
    for i in range(10):
        pass
"""
    result = validate_required_calls(PRE_WITH_PROGRESS, post, ["emit_progress"])
    assert result.ok is False
    assert "emit_progress" in result.reason
    assert "required" in result.reason.lower() or "preserved" in result.reason.lower()


def test_passes_when_attribute_call_preserved() -> None:
    """progress.emit_progress(...) (attribute style) also counts."""
    pre = """\
import autoresearch_rl.target.progress as progress

def train():
    progress.emit_progress(step=1, step_target=1, metrics={})
"""
    post = pre.replace("step=1", "step=2")
    result = validate_required_calls(pre, post, ["emit_progress"])
    assert result.ok is True


def test_no_required_passes_anything() -> None:
    result = validate_required_calls(PRE_WITH_PROGRESS, "x=1", [])
    assert result.ok is True


def test_rejects_post_syntax_error() -> None:
    bad = "def train(:\n    pass\n"
    result = validate_required_calls(PRE_WITH_PROGRESS, bad, ["emit_progress"])
    assert result.ok is False
    assert "syntax" in result.reason.lower()


def test_passes_when_pre_had_none() -> None:
    """If neither side has the call, it's not a regression — pass."""
    pre = "x = 1\n"
    post = "x = 2\n"
    result = validate_required_calls(pre, post, ["emit_progress"])
    assert result.ok is True


def test_multiple_required_calls_partial_strip() -> None:
    pre = """\
def train():
    emit_progress(step=1, step_target=1, metrics={})
    save_checkpoint("path")
"""
    post = """\
def train():
    emit_progress(step=1, step_target=1, metrics={})
"""
    result = validate_required_calls(pre, post, ["emit_progress", "save_checkpoint"])
    assert result.ok is False
    assert "save_checkpoint" in result.reason
