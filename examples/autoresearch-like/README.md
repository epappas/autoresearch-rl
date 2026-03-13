# autoresearch-like (continuous CLI)

This example runs via the **continuous CLI** using a command target.

## Run
```bash
uv run autoresearch-rl --config examples/autoresearch-like/example.yaml
```

## How it works
- Parameters are injected via env vars:
  - `AR_PARAMS_JSON`
  - `AR_PARAM_<NAME>`
- The script prints:
  - `loss=...`
  - `val_bpb=...`

## What to expect
- Run completes in ~30s (bounded by TIME_BUDGET_S).
- Artifacts:
  - `artifacts/autoresearch-like/results.tsv`
  - `artifacts/autoresearch-like/runs/`
  - `artifacts/autoresearch-like/versions/`
