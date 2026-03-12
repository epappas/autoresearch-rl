# Autoresearch Adapter (KISS)

This example uses the real community repo `karpathy/autoresearch` as the base.

Single command:

```bash
uv run python examples/autoresearch-adapter/run.py
```

What it does:
1. clones/updates `karpathy/autoresearch` into `artifacts/autoresearch-adapter/workdir`
2. validates three-file contract (`prepare.py`, `train.py`, `program.md`)
3. runs one bounded `uv run train.py` attempt
4. writes canonical ledger row to `artifacts/autoresearch-adapter/results.tsv`

Notes:
- Requires `uv` and suitable GPU/runtime for full training path.
- If `uv` is missing, it records a crash row with reason `uv_not_available`.

Optional env vars:
- `AUTORESEARCH_REF` (default: `master`)
- `AUTORESEARCH_TIMEOUT_S` (default: `300`)
- `AUTORESEARCH_LEDGER` (default: `artifacts/autoresearch-adapter/results.tsv`)
