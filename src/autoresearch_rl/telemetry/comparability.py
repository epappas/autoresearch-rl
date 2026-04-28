from __future__ import annotations

import functools
import hashlib
import platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparabilityPolicy:
    budget_mode: str = "fixed_wallclock"
    expected_budget_s: int = 300
    expected_hardware_fingerprint: str | None = None
    strict: bool = True


@functools.lru_cache(maxsize=1)
def hardware_fingerprint() -> str:
    parts = [
        platform.system(),
        platform.release(),
        platform.machine(),
        platform.python_version(),
    ]

    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
        if cp.returncode == 0 and cp.stdout.strip():
            parts.append(cp.stdout.strip().replace("\n", ","))
    except Exception:
        pass

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def check_comparability(policy: ComparabilityPolicy, run_budget_s: int, run_hardware_fingerprint: str) -> tuple[bool, str]:
    if policy.budget_mode not in ("fixed_wallclock", "parallel_wallclock"):
        return False, "unsupported_budget_mode"

    # parallel_wallclock writes per-trial budget into the ledger; the
    # loop-level budget is meaningless under parallelism, so skip the
    # equality check.
    if policy.budget_mode == "fixed_wallclock":
        if run_budget_s != policy.expected_budget_s:
            return False, f"budget_mismatch:{run_budget_s}!={policy.expected_budget_s}"

    if policy.expected_hardware_fingerprint and run_hardware_fingerprint != policy.expected_hardware_fingerprint:
        return False, "hardware_fingerprint_mismatch"

    return True, ""
