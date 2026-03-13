# Minimal Trainable Target

A tiny deterministic target project for the **continuous CLI**.

## Run (continuous CLI)
```bash
uv run autoresearch-rl --config configs/example.yaml
```

## Run manually
```bash
python examples/minimal-trainable-target/train.py \
  --learning-rate 2.8e-3 \
  --use-qk-norm \
  --grad-clip 0.8
```

## How it works
- The CLI injects params via env vars:
  - `AR_PARAMS_JSON`
  - `AR_PARAM_<NAME>`
- The script prints metrics:
  - `loss=...`
  - `val_bpb=...`

## What to expect
- The run completes quickly (seconds).
- Artifacts are written to:
  - `traces/events.jsonl`
  - `artifacts/results.tsv`
  - `artifacts/runs/`
  - `artifacts/versions/`

## Purpose
Use this project for:
- deterministic metric output
- quick end-to-end validation
- no external dependencies or GPUs
