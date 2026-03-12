# Autoresearch-style Three-File Contract Demo

This example mirrors the **structure** popularized by `karpathy/autoresearch`:

- `prepare.py` (frozen environment/eval constants)
- `train.py` (only mutable target)
- `program.md` (policy/instructions)

It demonstrates how the scaffold enforces the contract in strict mode.

## Run contract validator demo

```bash
uv run python examples/autoresearch-style-contract/run_contract_demo.py
```

Expected:
- diff touching `train.py` is allowed
- diff touching `prepare.py` is blocked

## Run through scaffold

```bash
uv run python scripts/run_once.py --config examples/autoresearch-style-contract/example.yaml
```

## Why this matters

This pattern keeps experiment comparability and reproducibility strong:
- benchmark harness remains immutable,
- mutation scope is narrow and reviewable,
- policy changes are explicit and versioned.
