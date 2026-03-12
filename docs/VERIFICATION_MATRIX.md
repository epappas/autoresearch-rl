# Verification Matrix

This matrix defines the core guarantees and how to verify them.

| Capability | Verification command | Expected outcome |
|---|---|---|
| Contract rules (allow mutable / block frozen+out-of-scope) | `pytest -q tests/test_contract.py` | all pass |
| Strict comparability blocks budget mismatch | `PYTHONPATH=src python3 - <<'PY'\nfrom autoresearch_rl.controller.loop import run_loop\nfrom autoresearch_rl.telemetry.comparability import ComparabilityPolicy\ntry:\n run_loop(max_iterations=1, trial_timeout_s=10, comparability_policy=ComparabilityPolicy(expected_budget_s=300, strict=True), mutable_file='examples/autoresearch-like/train.py', frozen_file='examples/autoresearch-like/prepare.py', program_path='examples/autoresearch-like/program.md')\n print('unexpected')\nexcept ValueError as e:\n print('blocked', str(e))\nPY` | Output starts with `blocked Non-comparable run blocked: budget_mismatch` |
| Comparable runs append ledger row with metadata | `PYTHONPATH=src python3 - <<'PY'\nfrom autoresearch_rl.controller.loop import run_loop\nfrom autoresearch_rl.telemetry.comparability import ComparabilityPolicy\nrun_loop(max_iterations=1, trial_timeout_s=1, ledger_path='artifacts/verify/results.tsv', comparability_policy=ComparabilityPolicy(expected_budget_s=1, strict=True), mutable_file='examples/autoresearch-like/train.py', frozen_file='examples/autoresearch-like/prepare.py', program_path='examples/autoresearch-like/program.md')\nprint('ok')\nPY && sed -n '1,2p' artifacts/verify/results.tsv` | TSV contains columns: `budget_mode`, `budget_s`, `hardware_fingerprint`, `comparable`, `non_comparable_reason` |
| CI contract smoke remains valid | `pytest -q tests/test_scaffold.py tests/test_contract.py tests/test_comparability.py tests/test_loop_comparability.py` | all pass |
