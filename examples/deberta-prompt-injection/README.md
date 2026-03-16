# deberta-prompt-injection (continuous CLI)

Fine-tune DeBERTa for prompt-injection classification. Four config variants are provided.

## Configs

| File | Target | Policy |
|------|--------|--------|
| `example.yaml` | local command | grid |
| `example-llm.yaml` | local command | LLM-guided |
| `basilica.yaml` | Basilica GPU | grid |
| `basilica-llm.yaml` | Basilica GPU | LLM-guided |

## Run (local, grid search)
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
uv run autoresearch-rl --config examples/deberta-prompt-injection/example.yaml
```

## Run (local, LLM-guided)
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
export CHUTES_API_KEY="your-key"
uv run autoresearch-rl --config examples/deberta-prompt-injection/example-llm.yaml
```

## Run (Basilica, grid search)
```bash
export BASILICA_API_TOKEN="your-basilica-token"
uv run autoresearch-rl --config examples/deberta-prompt-injection/basilica.yaml
```

## Run (Basilica, LLM-guided)
```bash
export BASILICA_API_TOKEN="your-basilica-token"
export CHUTES_API_KEY="your-key"
uv run autoresearch-rl --config examples/deberta-prompt-injection/basilica-llm.yaml
```

## How it works
- Params injected via env vars:
  - `AR_PARAMS_JSON`
  - `AR_PARAM_<NAME>`
- Script prints metrics including `val_bpb`.
- `val_bpb = 1.0 - f1` (lower is better).

## What to expect
- Local runs are slow (HF fine-tuning on CPU or small GPU).
- Basilica variants use 1x A100/H100/L40S with `setup_cmd` to install deps and fetch files.
- Artifacts written to `artifacts/deberta*/` and `traces/deberta*/`.

## Notes
Update the relevant YAML for your hardware or dataset paths.
