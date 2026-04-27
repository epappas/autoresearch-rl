"""End-to-end CommandTarget + emit_progress flow."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from autoresearch_rl.target.command import CommandTarget


def _trial_script(tmp_path: Path, body: str) -> Path:
    src_root = Path(__file__).resolve().parents[1] / "src"
    script = tmp_path / "trial.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(src_root)!r})
        from autoresearch_rl.target.progress import emit_progress
        {body}
    """))
    return script


def test_command_target_collects_progress(tmp_path: Path) -> None:
    script = _trial_script(tmp_path, body=textwrap.indent(textwrap.dedent("""
        for i in range(3):
            emit_progress(step=i, step_target=3, metrics={"loss": 1.0 / (i + 1)})
        print("loss=0.333333")
    """), "        ").lstrip())

    target = CommandTarget(
        train_cmd=[sys.executable, str(script)],
        eval_cmd=None,
        workdir=str(tmp_path),
        timeout_s=30,
    )
    run_dir = tmp_path / "run-0000"
    outcome = target.run(run_dir=str(run_dir), params={})

    assert outcome.status == "ok", outcome.stderr
    progress_file = run_dir / "progress.jsonl"
    assert progress_file.exists()
    lines = progress_file.read_text().splitlines()
    assert len(lines) == 3
    losses = [json.loads(line)["metrics"]["loss"] for line in lines]
    assert losses[0] == 1.0
    assert losses[1] == 0.5
    assert losses[2] == 1 / 3


def test_progress_metrics_backfill_when_stdout_silent(tmp_path: Path) -> None:
    """If trial only emits via emit_progress (no stdout metrics), the latest report
    should backfill outcome.metrics so the engine has a value to score on."""
    script = _trial_script(tmp_path, body=textwrap.indent(textwrap.dedent("""
        emit_progress(step=1, step_target=1, metrics={"loss": 0.42})
    """), "        ").lstrip())

    target = CommandTarget(
        train_cmd=[sys.executable, str(script)],
        eval_cmd=None,
        workdir=str(tmp_path),
        timeout_s=30,
    )
    run_dir = tmp_path / "run-0000"
    outcome = target.run(run_dir=str(run_dir), params={})
    assert outcome.status == "ok"
    assert outcome.metrics.get("loss") == 0.42
