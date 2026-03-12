# Autoresearch-like Example

This is a simple in-repo example that follows Karpathy's file pattern:
- `prepare.py` (frozen)
- `train.py` (mutable)
- `program.md` (policy)

Run:

```bash
uv run python examples/autoresearch-like/run.py
```

Or run controller loop with this example config:

```bash
uv run python scripts/run_once.py --config examples/autoresearch-like/example.yaml
```

Output:
- executes one bounded training run (`train.py`)
- validates contract + comparability
- appends `results.tsv` in this folder
