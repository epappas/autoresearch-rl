# deberta-prompt-injection

Fine-tune DeBERTa for prompt-injection classification.

## Prerequisites

```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
```

## Run (local, LLM-guided -- default)

```bash
export CHUTES_API_KEY="your-key"
bash examples/deberta-prompt-injection/run.sh
```

## Run (local, grid search)

```bash
bash examples/deberta-prompt-injection/run.sh --override policy.type=grid
```

## Deploy (Basilica)

```bash
export BASILICA_API_TOKEN="your-basilica-token"

# LLM-guided (default)
python3 examples/deberta-prompt-injection/deploy.py

# Grid search on Basilica
python3 examples/deberta-prompt-injection/deploy.py --policy grid
```

## How it works
- Params injected via env vars:
  - `AR_PARAMS_JSON`
  - `AR_PARAM_<NAME>`
- Script prints metrics including `val_bpb`.
- `val_bpb = 1.0 - f1` (lower is better).
- `program.md` provides task context to the LLM policy.

## What to expect
- Local runs are slow (HF fine-tuning on CPU or small GPU).
- Basilica deploys use 1x A100/H100/L40S with `setup_cmd` to install deps and fetch files.
- Artifacts written to `artifacts/deberta/` and `traces/deberta/`.

## Notes
Use `--override` to tweak parameters without editing the config:
```bash
bash examples/deberta-prompt-injection/run.sh \
  --override controller.max_wall_time_s=3600 \
  --override target.timeout_s=1800
```
