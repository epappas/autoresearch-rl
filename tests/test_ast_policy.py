from autoresearch_rl.sandbox.ast_policy import validate_python_source


def test_blocks_forbidden_import():
    r = validate_python_source("import socket\n")
    assert not r.ok


def test_allows_safe_code():
    r = validate_python_source("x = 1\nprint(x)\n")
    assert r.ok
