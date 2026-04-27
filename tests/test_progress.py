from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from autoresearch_rl.target.progress import (
    CANCEL_EXIT_CODE,
    ProgressReport,
    emit_progress,
)
from autoresearch_rl.target.progress_reader import ProgressReader


# ---------------------------------------------------------------- emit_progress


def test_emit_writes_jsonl_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    progress = tmp_path / "p.jsonl"
    monkeypatch.setenv("AR_PROGRESS_FILE", str(progress))
    monkeypatch.setenv("AR_ITER", "7")

    cont = emit_progress(step=3, step_target=10, metrics={"loss": 0.5})

    assert cont is True
    lines = progress.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["iter"] == 7
    assert data["step"] == 3
    assert data["step_target"] == 10
    assert data["metrics"] == {"loss": 0.5}
    assert data["should_continue"] is True


def test_emit_noop_without_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AR_PROGRESS_FILE", raising=False)
    cont = emit_progress(step=1, step_target=10, metrics={"loss": 1.0})
    assert cont is True
    # no file created anywhere


def test_emit_appends_multiple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    progress = tmp_path / "p.jsonl"
    monkeypatch.setenv("AR_PROGRESS_FILE", str(progress))
    for i in range(5):
        emit_progress(step=i, step_target=5, metrics={"loss": 1.0 / (i + 1)})
    lines = progress.read_text().splitlines()
    assert len(lines) == 5
    losses = [json.loads(line)["metrics"]["loss"] for line in lines]
    assert losses == [1.0, 0.5, pytest.approx(1 / 3), 0.25, 0.2]


def test_cancel_signal_triggers_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "p.jsonl"
    control = tmp_path / "c.json"
    control.write_text(json.dumps({"action": "cancel", "reason": "test"}))
    monkeypatch.setenv("AR_PROGRESS_FILE", str(progress))
    monkeypatch.setenv("AR_CONTROL_FILE", str(control))

    with pytest.raises(SystemExit) as exc:
        emit_progress(step=1, step_target=10, metrics={"loss": 0.5})
    assert exc.value.code == CANCEL_EXIT_CODE


def test_cancel_no_exit_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "p.jsonl"
    control = tmp_path / "c.json"
    control.write_text(json.dumps({"action": "cancel"}))
    monkeypatch.setenv("AR_PROGRESS_FILE", str(progress))
    monkeypatch.setenv("AR_CONTROL_FILE", str(control))

    cont = emit_progress(
        step=1, step_target=10, metrics={"loss": 0.5}, exit_on_cancel=False,
    )
    assert cont is False


def test_emit_survives_unwritable_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AR_PROGRESS_FILE", "/nonexistent_dir/p.jsonl")
    # Should not raise — trial filesystem failures are non-fatal.
    assert emit_progress(step=1, step_target=10, metrics={}) is True


# ---------------------------------------------------------------- ProgressReader


def test_reader_drains_appended_lines(tmp_path: Path) -> None:
    progress = tmp_path / "p.jsonl"
    reader = ProgressReader(str(progress), poll_interval_s=0.05)
    reader.start()
    try:
        progress.write_text(
            ProgressReport(
                iter=0, step=1, step_target=10, elapsed_s=1.0, metrics={"loss": 0.5},
            ).to_json_line() + "\n"
            + ProgressReport(
                iter=0, step=2, step_target=10, elapsed_s=2.0, metrics={"loss": 0.4},
            ).to_json_line() + "\n"
        )
        time.sleep(0.2)
        reports = reader.drain()
    finally:
        reader.stop()
    assert len(reports) == 2
    assert reports[0].step == 1
    assert reports[1].metrics == {"loss": 0.4}


def test_reader_skips_malformed_lines(tmp_path: Path) -> None:
    progress = tmp_path / "p.jsonl"
    reader = ProgressReader(str(progress), poll_interval_s=0.05)
    reader.start()
    try:
        progress.write_text(
            "not valid json\n"
            + ProgressReport(
                iter=0, step=1, step_target=10, elapsed_s=1.0, metrics={"loss": 0.5},
            ).to_json_line() + "\n"
            + "{partial}\n"
        )
        time.sleep(0.2)
        reports = reader.drain()
    finally:
        reader.stop()
    assert len(reports) == 1
    assert reports[0].step == 1


def test_reader_latest_returns_most_recent(tmp_path: Path) -> None:
    progress = tmp_path / "p.jsonl"
    reader = ProgressReader(str(progress), poll_interval_s=0.05)
    reader.start()
    try:
        for i in range(3):
            with open(progress, "a") as f:
                f.write(
                    ProgressReport(
                        iter=0, step=i, step_target=10, elapsed_s=i,
                        metrics={"loss": 1.0 / (i + 1)},
                    ).to_json_line() + "\n"
                )
            time.sleep(0.1)
        time.sleep(0.1)
        latest = reader.latest()
        all_reports = reader.drain()
    finally:
        reader.stop()
    assert latest is not None
    assert latest.step == 2
    assert len(all_reports) == 3


# ---------------------------------------------------------------- end-to-end via subprocess


def test_emit_in_subprocess(tmp_path: Path) -> None:
    """Simulates a real trial: spawn a Python subprocess with AR_PROGRESS_FILE set."""
    progress = tmp_path / "p.jsonl"
    script = tmp_path / "trial.py"
    script.write_text(textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from autoresearch_rl.target.progress import emit_progress
        for i in range(4):
            emit_progress(step=i, step_target=4, metrics={"loss": 1.0 / (i + 1)})
        sys.exit(0)
    """) % str(Path(__file__).resolve().parents[1] / "src"))

    env = {"AR_PROGRESS_FILE": str(progress), "AR_ITER": "3", "PATH": ""}
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    lines = progress.read_text().splitlines()
    assert len(lines) == 4
    iters = {json.loads(line)["iter"] for line in lines}
    assert iters == {3}
