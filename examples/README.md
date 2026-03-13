# Examples

This folder contains ready-to-run examples for the **continuous CLI**.

## Minimal trainable target
```bash
uv run autoresearch-rl --config configs/example.yaml
```

## Autoresearch-like
```bash
uv run autoresearch-rl --config examples/autoresearch-like/example.yaml
```

## DeBERTa prompt injection
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
uv run autoresearch-rl --config examples/deberta-prompt-injection/example.yaml
```
