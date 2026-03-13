from __future__ import annotations

import ast
from dataclasses import dataclass

FORBIDDEN_IMPORTS = {"socket", "requests", "httpx", "urllib", "subprocess"}
FORBIDDEN_CALLS = {
    "os.system",
    "subprocess.Popen",
    "subprocess.run",
}


@dataclass
class AstPolicyResult:
    ok: bool
    reason: str = ""


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def validate_python_source(src: str) -> AstPolicyResult:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return AstPolicyResult(ok=False, reason=f"syntax error: {e.msg}")

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                root = alias.name.split('.')[0]
                if root in FORBIDDEN_IMPORTS:
                    return AstPolicyResult(ok=False, reason=f"forbidden import: {root}")
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                root = n.module.split('.')[0]
                if root in FORBIDDEN_IMPORTS:
                    return AstPolicyResult(ok=False, reason=f"forbidden import-from: {root}")
        elif isinstance(n, ast.Call):
            dotted = _dotted_name(n.func)
            if dotted in FORBIDDEN_CALLS:
                return AstPolicyResult(ok=False, reason=f"forbidden call: {dotted}")

    return AstPolicyResult(ok=True)
