"""Tests for agent-facing CLI commands: status and run-one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from autoresearch_rl.checkpoint import LoopCheckpoint, save_checkpoint
from autoresearch_rl.cli import app

runner = CliRunner()

TRAIN_CMD = [sys.executable, "-c", "print('val_bpb=0.5')"]

VALID_DIFF = """\
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
 import torch
-LR = 0.001
+LR = 0.002
 EPOCHS = 10
"""

TRAIN_SOURCE = "import torch\nLR = 0.001\nEPOCHS = 10\n"


def _write_config(tmp_path: Path, extra: dict | None = None) -> str:
    cfg: dict = {
        "name": "test",
        "target": {
            "type": "command",
            "train_cmd": TRAIN_CMD,
        },
        "controller": {
            "max_wall_time_s": 60,
            "checkpoint_path": str(tmp_path / "checkpoint.json"),
        },
        "comparability": {"strict": False},
        "telemetry": {
            "trace_path": str(tmp_path / "traces/events.jsonl"),
            "ledger_path": str(tmp_path / "results.tsv"),
            "artifacts_dir": str(tmp_path / "artifacts/runs"),
            "versions_dir": str(tmp_path / "artifacts/versions"),
        },
    }
    if extra:
        _deep_merge(cfg, extra)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


def _deep_merge(base: dict, update: dict) -> None:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _make_checkpoint(tmp_path: Path) -> str:
    ckpt_path = str(tmp_path / "checkpoint.json")
    ckpt = LoopCheckpoint(
        episode_id="test-episode",
        iteration=4,
        best_score=0.42,
        best_value=0.42,
        no_improve_streak=2,
        history=[
            {"iter": i, "status": "ok", "decision": "keep" if i == 0 else "discard",
             "metrics": {"val_bpb": 0.5 - i * 0.01}, "params": {}}
            for i in range(5)
        ],
        recent_statuses=["ok", "ok", "failed"],
        policy_state={},
        elapsed_s=300.0,
        timestamp=1000.0,
    )
    save_checkpoint(ckpt_path, ckpt)
    return ckpt_path


# ---- status -----------------------------------------------------------------


class TestStatusCommand:

    def test_no_checkpoint_returns_empty_state(self, tmp_path):
        # Config with no checkpoint_path
        cfg_path = str(tmp_path / "config.yaml")
        cfg = {
            "target": {"type": "command", "train_cmd": TRAIN_CMD},
            "comparability": {"strict": False},
        }
        Path(cfg_path).write_text(yaml.dump(cfg))

        result = runner.invoke(app, ["status", "--config", cfg_path])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["best_value"] is None
        assert out["iterations_done"] == 0
        assert out["recent_history"] == []

    def test_with_checkpoint_returns_state(self, tmp_path):
        ckpt_path = _make_checkpoint(tmp_path)
        cfg_path = str(tmp_path / "config.yaml")
        cfg = {
            "target": {"type": "command", "train_cmd": TRAIN_CMD},
            "controller": {"checkpoint_path": ckpt_path},
            "comparability": {"strict": False},
        }
        Path(cfg_path).write_text(yaml.dump(cfg))

        result = runner.invoke(app, ["status", "--config", cfg_path])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["best_value"] == pytest.approx(0.42)
        assert out["iterations_done"] == 5
        assert out["no_improve_streak"] == 2
        assert len(out["recent_history"]) == 5

    def test_last_n_limits_history(self, tmp_path):
        ckpt_path = _make_checkpoint(tmp_path)
        cfg_path = str(tmp_path / "config.yaml")
        cfg = {
            "target": {"type": "command", "train_cmd": TRAIN_CMD},
            "controller": {"checkpoint_path": ckpt_path},
            "comparability": {"strict": False},
        }
        Path(cfg_path).write_text(yaml.dump(cfg))

        result = runner.invoke(app, ["status", "--config", cfg_path, "--last", "3"])

        assert result.exit_code == 0
        out = json.loads(result.output)
        assert len(out["recent_history"]) == 3

    def test_missing_checkpoint_file_returns_empty(self, tmp_path):
        cfg_path = str(tmp_path / "config.yaml")
        cfg = {
            "target": {"type": "command", "train_cmd": TRAIN_CMD},
            "controller": {"checkpoint_path": str(tmp_path / "nonexistent.json")},
            "comparability": {"strict": False},
        }
        Path(cfg_path).write_text(yaml.dump(cfg))

        result = runner.invoke(app, ["status", "--config", cfg_path])

        assert result.exit_code == 0
        out = json.loads(result.output)
        assert out["best_value"] is None


# ---- run-one ----------------------------------------------------------------


class TestRunOneCommand:

    def test_with_params_runs_one_iteration(self, tmp_path):
        cfg_path = _write_config(tmp_path)

        result = runner.invoke(app, [
            "run-one", "--config", cfg_path,
            "--params", '{"lr": 0.001}',
        ])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["iterations"] == 1

    def test_with_diff_runs_one_iteration(self, tmp_path):
        src = tmp_path / "train.py"
        src.write_text(TRAIN_SOURCE)
        diff_file = tmp_path / "change.patch"
        diff_file.write_text(VALID_DIFF)

        cfg_path = _write_config(tmp_path, {
            "policy": {"mutable_file": str(src)},
        })

        result = runner.invoke(app, [
            "run-one", "--config", cfg_path,
            "--diff", str(diff_file),
        ])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["iterations"] == 1

    def test_no_proposal_uses_configured_policy(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "policy": {"type": "static"},
        })

        result = runner.invoke(app, ["run-one", "--config", cfg_path])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert out["iterations"] == 1

    def test_params_and_diff_are_mutually_exclusive(self, tmp_path):
        diff_file = tmp_path / "change.patch"
        diff_file.write_text(VALID_DIFF)
        cfg_path = _write_config(tmp_path)

        result = runner.invoke(app, [
            "run-one", "--config", cfg_path,
            "--params", '{"lr": 0.001}',
            "--diff", str(diff_file),
        ])

        assert result.exit_code != 0

    def test_diff_without_mutable_file_raises(self, tmp_path):
        diff_file = tmp_path / "change.patch"
        diff_file.write_text(VALID_DIFF)
        # Config with no mutable_file
        cfg_path = _write_config(tmp_path)

        result = runner.invoke(app, [
            "run-one", "--config", cfg_path,
            "--diff", str(diff_file),
        ])

        assert result.exit_code != 0

    def test_run_one_result_is_valid_json(self, tmp_path):
        cfg_path = _write_config(tmp_path)

        result = runner.invoke(app, [
            "run-one", "--config", cfg_path,
            "--params", "{}",
        ])

        assert result.exit_code == 0, result.output
        out = json.loads(result.output)
        assert "iterations" in out
        assert "best_value" in out
        assert "best_score" in out
