from autoresearch_rl.telemetry.comparability import ComparabilityPolicy, check_comparability


def test_comparability_budget_match():
    p = ComparabilityPolicy(expected_budget_s=300, strict=True)
    ok, reason = check_comparability(policy=p, run_budget_s=300, run_hardware_fingerprint="fp1")
    assert ok
    assert reason == ""


def test_comparability_budget_mismatch():
    p = ComparabilityPolicy(expected_budget_s=300, strict=True)
    ok, reason = check_comparability(policy=p, run_budget_s=120, run_hardware_fingerprint="fp1")
    assert not ok
    assert reason.startswith("budget_mismatch")


def test_comparability_hardware_mismatch():
    p = ComparabilityPolicy(expected_budget_s=300, expected_hardware_fingerprint="fpX", strict=True)
    ok, reason = check_comparability(policy=p, run_budget_s=300, run_hardware_fingerprint="fpY")
    assert not ok
    assert reason == "hardware_fingerprint_mismatch"
