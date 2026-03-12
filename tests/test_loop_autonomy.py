from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy


def test_loop_stops_on_no_improve_limit(tmp_path):
    r = run_loop(
        max_iterations=50,
        mutable_file="examples/autoresearch-like/train.py",
        frozen_file="examples/autoresearch-like/prepare.py",
        program_path="examples/autoresearch-like/program.md",
        trial_timeout_s=2,
        trial_command=["python3", "-c", "print('val_bpb=1.2')"],
        comparability_policy=ComparabilityPolicy(expected_budget_s=2, strict=True),
        no_improve_limit=2,
        continuous=False,
        ledger_path=str(tmp_path / "r.tsv"),
        trace_path=str(tmp_path / "e.jsonl"),
        artifacts_dir=str(tmp_path / "a"),
    )
    assert r.iterations < 50
