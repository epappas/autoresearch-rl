from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from autoresearch_rl.sandbox.diff_utils import extract_touched_files_from_diff


@dataclass(frozen=True)
class ContractConfig:
    frozen_file: str
    mutable_file: str
    program_file: str
    strict: bool = True


def validate_contract_files_exist(contract: ContractConfig, root: str = ".") -> tuple[bool, str]:
    base = Path(root)
    required = [contract.frozen_file, contract.mutable_file, contract.program_file]
    for rel in required:
        if not (base / rel).exists():
            return False, f"contract_file_missing:{rel}"
    return True, ""


def validate_diff_against_contract(diff: str, contract: ContractConfig) -> tuple[bool, str]:
    """Reject diffs that touch files outside the mutable scope.

    Path comparison is on basename: LLM-generated diffs use basenames
    (the LLM only sees filenames, never workdir prefixes), but the
    contract config stores workdir-relative paths
    (e.g. examples/foo/train.py). Comparing literally rejects every
    well-formed diff. We normalize to basename on both sides.
    """
    import os.path

    touched = extract_touched_files_from_diff(diff)
    if not touched:
        return True, ""

    frozen_base = os.path.basename(contract.frozen_file)
    program_base = os.path.basename(contract.program_file)
    mutable_base = os.path.basename(contract.mutable_file)

    for path in touched:
        path_base = os.path.basename(path)
        if path_base == frozen_base:
            return False, f"frozen_file_mutation_blocked:{path}"
        if path_base == program_base:
            return False, f"program_file_mutation_blocked:{path}"
        if path_base != mutable_base:
            return False, f"out_of_scope_mutation_blocked:{path}"

    return True, ""
