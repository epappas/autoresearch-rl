from __future__ import annotations

from dataclasses import dataclass


FORBIDDEN_TOKENS = [
    "import socket",
    "requests.",
    "subprocess.Popen(",
    "os.system(",
]


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


def validate_diff(diff: str) -> ValidationResult:
    if not diff.strip():
        return ValidationResult(ok=False, reason="empty diff")
    for token in FORBIDDEN_TOKENS:
        if token in diff:
            return ValidationResult(ok=False, reason=f"forbidden token: {token}")
    return ValidationResult(ok=True)
