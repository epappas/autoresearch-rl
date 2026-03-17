from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from autoresearch_rl.cli import app


def test_cli_run_with_program_path(tmp_path: Path):
    program_file = tmp_path / "task.md"
    program_file.write_text("Train a small LM to minimize val_bpb.", encoding="utf-8")
    cfg = {
        "name": "cli-program",
        "program_path": str(program_file),
        "objective": {"metric": "val_bpb", "direction": "min"},
        "target": {
            "type": "command",
            "workdir": ".",
            "timeout_s": 10,
            "train_cmd": ["python3", "examples/minimal-trainable-target/train.py"],
            "eval_cmd": ["python3", "examples/minimal-trainable-target/train.py"],
        },
        "policy": {"type": "static", "params": {}},
        "controller": {"max_wall_time_s": 2, "no_improve_limit": 1},
        "comparability": {"budget_mode": "fixed_wallclock", "expected_budget_s": 2, "strict": False},
        "telemetry": {
            "trace_path": str(tmp_path / "events.jsonl"),
            "ledger_path": str(tmp_path / "results.tsv"),
            "artifacts_dir": str(tmp_path / "runs"),
            "versions_dir": str(tmp_path / "versions"),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    captured_program: list[str] = []
    original_run = __import__(
        "autoresearch_rl.controller.continuous", fromlist=["run_continuous"]
    ).run_continuous

    def spy_run_continuous(**kwargs):
        captured_program.append(kwargs.get("program", ""))
        return original_run(**kwargs)

    with patch("autoresearch_rl.cli.run_continuous", side_effect=spy_run_continuous):
        runner = CliRunner()
        result = runner.invoke(app, ["--config", str(cfg_path)])

    assert result.exit_code == 0
    assert len(captured_program) == 1
    assert "Train a small LM" in captured_program[0]


def test_cli_run_smoke(tmp_path: Path):
    cfg = {
        "name": "cli-smoke",
        "objective": {"metric": "val_bpb", "direction": "min"},
        "target": {
            "type": "command",
            "workdir": ".",
            "timeout_s": 10,
            "train_cmd": ["python3", "examples/minimal-trainable-target/train.py"],
            "eval_cmd": ["python3", "examples/minimal-trainable-target/train.py"],
        },
        "policy": {"type": "static", "params": {}},
        "controller": {"max_wall_time_s": 2, "no_improve_limit": 1},
        "comparability": {"budget_mode": "fixed_wallclock", "expected_budget_s": 2, "strict": False},
        "telemetry": {
            "trace_path": str(tmp_path / "events.jsonl"),
            "ledger_path": str(tmp_path / "results.tsv"),
            "artifacts_dir": str(tmp_path / "runs"),
            "versions_dir": str(tmp_path / "versions"),
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "iterations" in result.stdout
