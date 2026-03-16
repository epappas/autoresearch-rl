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

## Basilica GRPO
```bash
export BASILICA_API_TOKEN="your-basilica-token"
export HF_TOKEN="your-huggingface-token"
uv run autoresearch-rl --config examples/basilica-grpo/example.yaml
```

## DeBERTa prompt injection

### Local (grid search)
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
uv run autoresearch-rl --config examples/deberta-prompt-injection/example.yaml
```

### Local (LLM-guided search)
```bash
pip install -r examples/deberta-prompt-injection/requirements.txt
export CHUTES_API_KEY="your-key"
uv run autoresearch-rl --config examples/deberta-prompt-injection/example-llm.yaml
```

### Basilica (grid search)
```bash
export BASILICA_API_TOKEN="your-basilica-token"
uv run autoresearch-rl --config examples/deberta-prompt-injection/basilica.yaml
```

### Basilica (LLM-guided search)
```bash
export BASILICA_API_TOKEN="your-basilica-token"
export CHUTES_API_KEY="your-key"
uv run autoresearch-rl --config examples/deberta-prompt-injection/basilica-llm.yaml
```
