# Minimal Trainable Target

## Objective
Minimize `val_bpb` by proposing code diffs to `train.py`. No external dependencies — runs in
milliseconds. Ideal for validating the end-to-end autoresearch-rl loop.

## Mutable File
`train.py` — the training script. The RL loop proposes unified diffs to this file each iteration.

## Frozen File
`prepare.py` — evaluation oracle defining the canonical objective landscape. Must not be modified.

## Metrics
- `val_bpb` (primary, lower is better)
- `loss` (secondary)

## Objective Landscape
The synthetic benchmark has a known optimum at:
- `learning_rate ≈ 2.6e-3`
- `grad_clip ≈ 0.85`
- `use_qk_norm = True`

The penalty coefficients in `synthetic_metrics` control how steep the landscape is around this
optimum. Modifying them changes the difficulty of the search.

## Guidance
- Do not change the `print(f"loss=...")` and `print(f"val_bpb=...")` output lines.
- Each iteration is near-instant — the wall-time budget allows many iterations.
- Propose one change at a time: parameter default, penalty coefficient, or logic fix.
