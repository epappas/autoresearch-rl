# Autoresearch-style Contract Example

KISS entrypoint:

```bash
uv run python examples/autoresearch-style-contract/run.py
```

What it shows:
- diff on `train.py` => allowed
- diff on `prepare.py` => blocked

Optional full loop:

```bash
uv run python scripts/run_once.py --config examples/autoresearch-style-contract/example.yaml
```
