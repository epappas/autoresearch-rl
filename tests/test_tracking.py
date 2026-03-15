from __future__ import annotations

import json
from pathlib import Path

from autoresearch_rl.tracking import LocalFileTracker


def test_creates_directory_structure(tmp_path: Path) -> None:
    LocalFileTracker(str(tmp_path), "run-001")
    assert (tmp_path / "run-001").is_dir()
    assert (tmp_path / "run-001" / "artifacts").is_dir()


def test_log_params_writes_valid_json(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-002")
    tracker.log_params({"lr": 0.001, "epochs": 10})
    params_path = tmp_path / "run-002" / "params.json"
    assert params_path.exists()
    data = json.loads(params_path.read_text(encoding="utf-8"))
    assert data["lr"] == 0.001
    assert data["epochs"] == 10


def test_log_params_merges_successive_calls(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-003")
    tracker.log_params({"lr": 0.001})
    tracker.log_params({"batch_size": 32})
    data = json.loads(
        (tmp_path / "run-003" / "params.json").read_text(encoding="utf-8")
    )
    assert data == {"lr": 0.001, "batch_size": 32}


def test_log_params_overwrites_existing_key(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-004")
    tracker.log_params({"lr": 0.001})
    tracker.log_params({"lr": 0.01})
    data = json.loads(
        (tmp_path / "run-004" / "params.json").read_text(encoding="utf-8")
    )
    assert data["lr"] == 0.01


def test_log_metrics_appends_jsonl(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-005")
    tracker.log_metrics({"loss": 0.5, "acc": 0.8}, step=0)
    tracker.log_metrics({"loss": 0.3, "acc": 0.9}, step=1)
    metrics_path = tmp_path / "run-005" / "metrics.jsonl"
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 0
    assert first["loss"] == 0.5
    assert first["acc"] == 0.8
    assert "ts" in first
    second = json.loads(lines[1])
    assert second["step"] == 1
    assert second["loss"] == 0.3


def test_log_artifact_copies_file(tmp_path: Path) -> None:
    src_file = tmp_path / "model.bin"
    src_file.write_bytes(b"fake-model-weights")

    tracker = LocalFileTracker(str(tmp_path), "run-006")
    tracker.log_artifact(str(src_file), "best_model.bin")

    dest = tmp_path / "run-006" / "artifacts" / "best_model.bin"
    assert dest.exists()
    assert dest.read_bytes() == b"fake-model-weights"


def test_log_artifact_raises_on_missing_source(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-007")
    try:
        tracker.log_artifact("/nonexistent/file.txt", "missing.txt")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_set_status_updates_status_file(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-008")
    tracker.set_status("running")
    status_path = tmp_path / "run-008" / "status.json"
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["status"] == "running"
    assert "ts" in data

    tracker.set_status("completed")
    data = json.loads(status_path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"


def test_run_dir_property(tmp_path: Path) -> None:
    tracker = LocalFileTracker(str(tmp_path), "run-009")
    assert tracker.run_dir == tmp_path / "run-009"
