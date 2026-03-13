from __future__ import annotations

from dataclasses import dataclass

from autoresearch_rl.sandbox.ast_policy import validate_python_source

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

    # quick token guard for any diff format
    for token in FORBIDDEN_TOKENS:
        if token in diff:
            return ValidationResult(ok=False, reason=f"forbidden token: {token}")

    # best-effort AST validation for added Python lines
    added_lines = []
    for line in diff.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:].lstrip())
    if added_lines:
        src = "\n".join(added_lines)
        ast_result = validate_python_source(src)
        if not ast_result.ok:
            return ValidationResult(ok=False, reason=ast_result.reason)

    return ValidationResult(ok=True)
