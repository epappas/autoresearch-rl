# deberta-prompt-injection (continuous CLI)

This example is wired for the continuous CLI. You must install requirements
and point the target commands to your environment.

## Run
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
uv run autoresearch-rl --config examples/deberta-prompt-injection/example.yaml
```

## How it works
- Params injected via env vars:
  - `AR_PARAMS_JSON`
  - `AR_PARAM_<NAME>`
- Script prints metrics including `val_bpb`.

## What to expect
- This run is **slow** (HF fine-tuning).
- Artifacts:
  - `artifacts/deberta/results.tsv`
  - `artifacts/deberta/runs/`
  - `artifacts/deberta/versions/`

## Notes
Update `example.yaml` for your hardware / dataset paths.
