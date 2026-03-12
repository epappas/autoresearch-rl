from autoresearch_rl.controller.contract import ContractConfig, validate_diff_against_contract


def _contract() -> ContractConfig:
    return ContractConfig(
        frozen_file="prepare.py",
        mutable_file="train.py",
        program_file="program.md",
        strict=True,
    )


def test_allows_mutable_file_diff():
    diff = (
        "diff --git a/train.py b/train.py\n"
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    ok, reason = validate_diff_against_contract(diff, _contract())
    assert ok
    assert reason == ""


def test_blocks_frozen_file_diff():
    diff = (
        "diff --git a/prepare.py b/prepare.py\n"
        "--- a/prepare.py\n"
        "+++ b/prepare.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    ok, reason = validate_diff_against_contract(diff, _contract())
    assert not ok
    assert reason.startswith("frozen_file_mutation_blocked")


def test_blocks_out_of_scope_file_diff():
    diff = (
        "diff --git a/other.py b/other.py\n"
        "--- a/other.py\n"
        "+++ b/other.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    ok, reason = validate_diff_against_contract(diff, _contract())
    assert not ok
    assert reason.startswith("out_of_scope_mutation_blocked")
