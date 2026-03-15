from __future__ import annotations

import os
import tempfile

from autoresearch_rl.checkpoint import (
    get_latest_snapshot_version,
    load_policy_snapshot,
    save_policy_snapshot,
)


def test_save_load_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        weights = {"lr": 0.001, "beta": 0.5, "layers": [64, 32]}
        path = save_policy_snapshot(tmp, 3, weights)
        assert os.path.isfile(path)
        assert path.endswith("policy_v0003.json")

        loaded = load_policy_snapshot(tmp, 3)
        assert loaded == weights


def test_load_missing_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = load_policy_snapshot(tmp, 99)
        assert result is None


def test_get_latest_snapshot_version_multiple() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        save_policy_snapshot(tmp, 1, {"a": 1})
        save_policy_snapshot(tmp, 5, {"a": 5})
        save_policy_snapshot(tmp, 3, {"a": 3})

        assert get_latest_snapshot_version(tmp) == 5


def test_get_latest_snapshot_version_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert get_latest_snapshot_version(tmp) == -1


def test_get_latest_snapshot_version_nonexistent_dir() -> None:
    assert get_latest_snapshot_version("/tmp/nonexistent_snapshot_dir_xyz") == -1
