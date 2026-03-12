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
    touched = extract_touched_files_from_diff(diff)
    if not touched:
        return True, ""

    for path in touched:
        if path == contract.frozen_file:
            return False, f"frozen_file_mutation_blocked:{path}"
        if path == contract.program_file:
            return False, f"program_file_mutation_blocked:{path}"
        if path != contract.mutable_file:
            return False, f"out_of_scope_mutation_blocked:{path}"

    return True, ""
