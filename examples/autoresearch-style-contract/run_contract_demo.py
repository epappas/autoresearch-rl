from __future__ import annotations

from autoresearch_rl.controller.contract import ContractConfig, validate_diff_against_contract


def main() -> None:
    contract = ContractConfig(
        frozen_file="examples/autoresearch-style-contract/prepare.py",
        mutable_file="examples/autoresearch-style-contract/train.py",
        program_file="examples/autoresearch-style-contract/program.md",
        strict=True,
    )

    good_diff = (
        "diff --git a/examples/autoresearch-style-contract/train.py "
        "b/examples/autoresearch-style-contract/train.py\n"
        "--- a/examples/autoresearch-style-contract/train.py\n"
        "+++ b/examples/autoresearch-style-contract/train.py\n"
        "@@ -1 +1 @@\n"
        "-LEARNING_RATE = 0.0026\n"
        "+LEARNING_RATE = 0.0028\n"
    )

    bad_diff = (
        "diff --git a/examples/autoresearch-style-contract/prepare.py "
        "b/examples/autoresearch-style-contract/prepare.py\n"
        "--- a/examples/autoresearch-style-contract/prepare.py\n"
        "+++ b/examples/autoresearch-style-contract/prepare.py\n"
        "@@ -1 +1 @@\n"
        "-TIME_BUDGET_S = 30\n"
        "+TIME_BUDGET_S = 60\n"
    )

    ok1, reason1 = validate_diff_against_contract(good_diff, contract)
    ok2, reason2 = validate_diff_against_contract(bad_diff, contract)

    print({"good_diff_allowed": ok1, "good_reason": reason1})
    print({"bad_diff_allowed": ok2, "bad_reason": reason2})


if __name__ == "__main__":
    main()
