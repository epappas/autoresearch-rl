#!/usr/bin/env bash
set -euo pipefail

# 1) Contract rules
pytest -q tests/test_contract.py

# 2) Strict comparability blocks budget mismatch
PYTHONPATH=src python3 - <<'PY'
from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy
try:
    run_loop(
        max_iterations=1,
        trial_timeout_s=10,
        comparability_policy=ComparabilityPolicy(expected_budget_s=300, strict=True),
        mutable_file='examples/autoresearch-like/train.py',
        frozen_file='examples/autoresearch-like/prepare.py',
        program_path='examples/autoresearch-like/program.md',
    )
    raise SystemExit('expected comparability block, got success')
except ValueError as e:
    msg = str(e)
    assert 'budget_mismatch' in msg, msg
PY

# 3) Comparable run appends ledger metadata
rm -f artifacts/verify/results.tsv
PYTHONPATH=src python3 - <<'PY'
from autoresearch_rl.controller.loop import run_loop
from autoresearch_rl.telemetry.comparability import ComparabilityPolicy
run_loop(
    max_iterations=1,
    trial_timeout_s=1,
    ledger_path='artifacts/verify/results.tsv',
    comparability_policy=ComparabilityPolicy(expected_budget_s=1, strict=True),
    mutable_file='examples/autoresearch-like/train.py',
    frozen_file='examples/autoresearch-like/prepare.py',
    program_path='examples/autoresearch-like/program.md',
)
PY

python3 - <<'PY'
import csv
from pathlib import Path
p = Path('artifacts/verify/results.tsv')
assert p.exists(), 'missing artifacts/verify/results.tsv'
rows = list(csv.reader(p.open(encoding='utf-8'), delimiter='\t'))
assert len(rows) >= 2, f'expected data row, got {len(rows)} rows'
header = rows[0]
required = {'budget_mode', 'budget_s', 'hardware_fingerprint', 'comparable', 'non_comparable_reason'}
missing = required.difference(header)
assert not missing, f'missing columns: {sorted(missing)}'
PY

echo "verification matrix checks passed"
