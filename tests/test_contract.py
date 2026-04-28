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


def test_allows_diff_when_contract_uses_workdir_prefixed_paths():
    """Regression: examples set policy.mutable_file to a workdir-prefixed
    path like 'examples/foo/train.py' but LLM-generated diffs use the
    basename ('train.py'). The validator must compare on basename, not
    on the literal config string. This bug silently broke every llm_diff
    and hybrid example until it was caught by smoke-testing them."""
    contract = ContractConfig(
        frozen_file="examples/foo/prepare.py",
        mutable_file="examples/foo/train.py",
        program_file="examples/foo/program.md",
        strict=True,
    )
    diff = (
        "diff --git a/train.py b/train.py\n"
        "--- a/train.py\n"
        "+++ b/train.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    ok, reason = validate_diff_against_contract(diff, contract)
    assert ok, f"expected pass; got {reason}"


def test_blocks_frozen_when_contract_uses_workdir_prefixed_paths():
    """Same regression coverage but for the frozen file."""
    contract = ContractConfig(
        frozen_file="examples/foo/prepare.py",
        mutable_file="examples/foo/train.py",
        program_file="examples/foo/program.md",
        strict=True,
    )
    diff = (
        "diff --git a/prepare.py b/prepare.py\n"
        "--- a/prepare.py\n"
        "+++ b/prepare.py\n"
        "@@ -1 +1 @@\n"
        "-x=1\n"
        "+x=2\n"
    )
    ok, reason = validate_diff_against_contract(diff, contract)
    assert not ok
    assert reason.startswith("frozen_file_mutation_blocked")
