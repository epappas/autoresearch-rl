from __future__ import annotations

import ast
from dataclasses import dataclass

from autoresearch_rl.sandbox.ast_policy import _dotted_name, validate_python_source

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


def validate_required_calls(
    pre_source: str,
    post_source: str,
    required: list[str],
) -> ValidationResult:
    """Reject diffs that strip load-bearing function calls.

    For each name in `required`, if the pre-patch source had at least one call
    and the post-patch source has none, reject with a correction message that
    the LLM can read on retry.
    """
    if not required:
        return ValidationResult(ok=True)
    try:
        pre_tree = ast.parse(pre_source)
        post_tree = ast.parse(post_source)
    except SyntaxError as exc:
        return ValidationResult(ok=False, reason=f"post-patch syntax error: {exc.msg}")

    pre_counts = _count_calls(pre_tree, required)
    post_counts = _count_calls(post_tree, required)

    stripped = [
        name for name in required
        if pre_counts.get(name, 0) > 0 and post_counts.get(name, 0) == 0
    ]
    if stripped:
        joined = ", ".join(f"{n}(...)" for n in stripped)
        return ValidationResult(
            ok=False,
            reason=(
                f"diff removes all calls to {joined}; these are required by "
                f"policy.required_calls and must be preserved"
            ),
        )
    return ValidationResult(ok=True)


def _count_calls(tree: ast.AST, names: list[str]) -> dict[str, int]:
    """Count Call nodes whose function name (or attr) matches one of `names`."""
    counts: dict[str, int] = {n: 0 for n in names}
    name_set = set(names)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ""
        if isinstance(node.func, ast.Name):
            target = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target = node.func.attr
            # Also try the dotted form (e.g. mod.emit_progress)
            dotted = _dotted_name(node.func)
            if dotted in name_set:
                counts[dotted] += 1
                continue
        if target in name_set:
            counts[target] += 1
    return counts
