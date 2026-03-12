from autoresearch_rl.controller.loop import run_loop


def test_loop_runs():
    r = run_loop(
        max_iterations=1,
        mutable_file="examples/autoresearch-style-contract/train.py",
        frozen_file="examples/autoresearch-style-contract/prepare.py",
        program_path="examples/autoresearch-style-contract/program.md",
        contract_strict=True,
    )
    assert r.iterations == 1
