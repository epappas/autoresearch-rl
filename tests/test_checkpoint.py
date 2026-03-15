from __future__ import annotations

import json
import os
import tempfile
import time

from autoresearch_rl.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint


def _make_checkpoint(**overrides) -> LoopCheckpoint:
    defaults = {
        "episode_id": "ep-001",
        "iteration": 3,
        "best_score": 0.42,
        "best_value": 0.42,
        "no_improve_streak": 1,
        "history": [{"iter": 0, "status": "ok"}],
        "recent_statuses": ["ok", "ok", "failed"],
        "policy_state": {"type": "static"},
        "elapsed_s": 12.5,
        "timestamp": time.time(),
    }
    defaults.update(overrides)
    return LoopCheckpoint(**defaults)


def test_save_load_roundtrip():
    ckpt = _make_checkpoint()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        save_checkpoint(path, ckpt)
        loaded = load_checkpoint(path)

    assert loaded is not None
    assert loaded.episode_id == ckpt.episode_id
    assert loaded.iteration == ckpt.iteration
    assert loaded.best_score == ckpt.best_score
    assert loaded.best_value == ckpt.best_value
    assert loaded.no_improve_streak == ckpt.no_improve_streak
    assert loaded.history == ckpt.history
    assert loaded.recent_statuses == ckpt.recent_statuses
    assert loaded.policy_state == ckpt.policy_state
    assert loaded.elapsed_s == ckpt.elapsed_s
    assert loaded.timestamp == ckpt.timestamp


def test_load_missing_file_returns_none():
    result = load_checkpoint("/tmp/nonexistent_ckpt_abc123.json")
    assert result is None


def test_atomic_write_no_partial():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        for i in range(20):
            save_checkpoint(
                path, _make_checkpoint(iteration=i, elapsed_s=float(i))
            )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["iteration"] == i
            assert data["elapsed_s"] == float(i)


def test_save_creates_parent_directories():
    with tempfile.TemporaryDirectory() as td:
        nested = os.path.join(td, "a", "b", "c", "ckpt.json")
        save_checkpoint(nested, _make_checkpoint())
        loaded = load_checkpoint(nested)
        assert loaded is not None
        assert loaded.episode_id == "ep-001"


def test_overwrite_existing_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        save_checkpoint(path, _make_checkpoint(iteration=0, best_score=1.0))
        save_checkpoint(path, _make_checkpoint(iteration=5, best_score=0.3))
        loaded = load_checkpoint(path)
        assert loaded is not None
        assert loaded.iteration == 5
        assert loaded.best_score == 0.3


def test_best_value_none_roundtrip():
    ckpt = _make_checkpoint(best_value=None)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        save_checkpoint(path, ckpt)
        loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.best_value is None


def test_empty_history_roundtrip():
    ckpt = _make_checkpoint(history=[], recent_statuses=[])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        save_checkpoint(path, ckpt)
        loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.history == []
    assert loaded.recent_statuses == []


def test_no_temp_files_left_after_save():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.json")
        save_checkpoint(path, _make_checkpoint())
        files = os.listdir(td)
        assert files == ["ckpt.json"]
