from autoresearch_rl.sandbox.runner import run_trial


def test_rejects_forbidden_diff():
    r = run_trial("import socket", timeout_s=1)
    assert r.status == "rejected"
