#!/usr/bin/env bash
# Convenience wrapper that runs the showcase end-to-end and prints
# a one-page summary of artifacts produced.
#
# Run from the repo root:
#   bash examples/parallel-cancel-showcase/run.sh
#
# No GPU required. Uses CPU-only synthetic trials.

set -euo pipefail

CONFIG="examples/parallel-cancel-showcase/config.yaml"
ART="examples/parallel-cancel-showcase/artifacts"

# Clean previous run so wall-time numbers are honest.
rm -rf "$ART"

echo "== Running parallel + cancel showcase (max_concurrency=4) =="
time uv run autoresearch-rl run "$CONFIG"

echo ""
echo "== Ledger summary =="
echo "Total rows:       $(awk 'NR>1' "$ART/results.tsv" | wc -l)"
echo "Status breakdown:"
awk -F'\t' 'NR>1 {print "  ", $5}' "$ART/results.tsv" | sort | uniq -c

echo ""
echo "== Timeline events =="
python3 -c "
import json
data = json.load(open('$ART/traces/timeline.json'))
from collections import Counter
counts = Counter(e['name'] for e in data)
print(f'  Total spans: {len(data)}')
for name, n in sorted(counts.items()):
    print(f'  {name}: {n}')
"

echo ""
echo "== Cancellations (per-trial control files) =="
find "$ART/runs" -name "control.json" -print0 \
    | xargs -0 -I{} sh -c 'echo "  $(dirname {} | xargs basename): $(cat {})"' \
    || true

echo ""
echo "Artifacts under $ART/"
echo "Open the timeline in chrome://tracing or ui.perfetto.dev to see"
echo "per-trial spans and intra-iteration cancellations on a timeline."
