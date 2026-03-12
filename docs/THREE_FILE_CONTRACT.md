# Three-File Contract (AutoResearch style)

This repository adopts a strict control contract inspired by `karpathy/autoresearch`.

## Contract
1. **Frozen environment file** (`prepare.py`-equivalent)
   - Holds fixed benchmark/data/eval/runtime constants.
   - Must not be mutated by autonomous runs.

2. **Mutable target file** (`train.py`-equivalent)
   - Single RL-editable artifact for experiments.
   - All candidate diffs should be scoped to this file by default.

3. **Program file** (`program.md`-equivalent)
   - Human-authored policy/instructions for the autonomous loop.
   - Treated as the "research org code" and versioned explicitly.

## Why
- Maximizes comparability and reproducibility.
- Keeps experiments reviewable.
- Prevents hidden drift in evaluation harness.

## Recommended defaults in this scaffold
- frozen: `prepare.py` (or target-specific equivalent)
- mutable: `train.py`
- program: `programs/default.md`
