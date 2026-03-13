# DeBERTa Prompt-Injection Example

A realistic target project for the scaffold using a Hugging Face DeBERTa classifier.

- Base model: `protectai/deberta-v3-base-prompt-injection-v2`
- Task: binary prompt-injection detection
- Dataset: small local JSONL (for reproducible smoke tests)

## Install deps (uv)

```bash
uv sync --extra dev
uv pip install -r examples/deberta-prompt-injection/requirements.txt
```

## Install deps (pip fallback)

```bash
pip install -e .[dev]
pip install -r examples/deberta-prompt-injection/requirements.txt
```

## Run training directly

```bash
uv run python examples/deberta-prompt-injection/train.py
```

## Run through scaffold trial runner

```bash
uv run python examples/deberta-prompt-injection/run.py
```


## Output metrics
The script prints:
- `loss=...`
- `val_bpb=...` (defined as `1 - f1` for scaffold compatibility)
- `f1=...`
- `accuracy=...`

and writes `artifacts/deberta-example/metrics.json`.

## Notes
- This is a minimal demonstration target. For real use, swap in larger curated train/val sets.
- You can point `--model-name` to another checkpoint if desired.
