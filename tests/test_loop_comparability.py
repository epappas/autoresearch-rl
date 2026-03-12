import pytest

from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy


def test_run_loop_blocks_non_comparable_budget(tmp_path):
    with pytest.raises(ValueError):
        run_loop(
            max_iterations=1,
            trace_path=str(tmp_path / "t.jsonl"),
            artifacts_dir=str(tmp_path / "a"),
            ledger_path=str(tmp_path / "results.tsv"),
            trial_timeout_s=10,
            comparability_policy=ComparabilityPolicy(expected_budget_s=300, strict=True),
        )
