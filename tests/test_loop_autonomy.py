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


def test_loop_stops_on_failure_rate_limit(tmp_path):
    r = run_loop(
        max_iterations=50,
        mutable_file="examples/autoresearch-like/train.py",
        frozen_file="examples/autoresearch-like/prepare.py",
        program_path="examples/autoresearch-like/program.md",
        trial_timeout_s=2,
        trial_command=["python3", "-c", "import sys; sys.exit(1)"],
        comparability_policy=ComparabilityPolicy(expected_budget_s=2, strict=True),
        failure_rate_limit=0.5,
        failure_window=2,
        continuous=False,
        ledger_path=str(tmp_path / "r2.tsv"),
        trace_path=str(tmp_path / "e2.jsonl"),
        artifacts_dir=str(tmp_path / "a2"),
    )
    assert r.iterations < 50


def test_loop_stops_on_max_wall_time(tmp_path):
    r = run_loop(
        max_iterations=999,
        mutable_file="examples/autoresearch-like/train.py",
        frozen_file="examples/autoresearch-like/prepare.py",
        program_path="examples/autoresearch-like/program.md",
        trial_timeout_s=1,
        trial_command=["python3", "-c", "print('val_bpb=1.2')"],
        comparability_policy=ComparabilityPolicy(expected_budget_s=1, strict=True),
        continuous=True,
        max_wall_time_s=1,
        ledger_path=str(tmp_path / "r3.tsv"),
        trace_path=str(tmp_path / "e3.jsonl"),
        artifacts_dir=str(tmp_path / "a3"),
    )
    assert r.iterations < 999
