# Examples

This folder contains concrete target projects that the scaffold can optimize.

## Included

- `minimal-trainable-target/`
  - Tiny synthetic training project with a real `train.py`.
  - Emits `loss=...` and `val_bpb=...` so the scaffold metric parser works out-of-the-box.

## Why this exists

The scaffold repository is primarily a **control-plane** (proposal → trial → scoring).
These examples provide a **data-plane target** so you can run realistic loops against actual files.
