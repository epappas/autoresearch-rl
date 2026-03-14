from pathlib import Path

import yaml
from typer.testing import CliRunner

from autoresearch_rl.cli import app


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
