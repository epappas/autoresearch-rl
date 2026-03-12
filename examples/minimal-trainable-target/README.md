# Minimal Trainable Target

A tiny deterministic target project for `autoresearch-rl-scaffold`.

## Run manually

```bash
python examples/minimal-trainable-target/train.py \
  --learning-rate 2.8e-3 \
  --use-qk-norm \
  --grad-clip 0.8
```

Output format matches scaffold parser expectations:

- `loss=...`
- `val_bpb=...`

## Purpose

Use this project when you want:
- a real `train.py` file the scaffold can conceptually mutate,
- deterministic metric output for reproducible loop testing,
- no external dependencies or GPUs.
