# AutoResearch-Like Training Program

## Objective
Minimize `val_bpb` by proposing code diffs to `train.py` — modifying the training logic directly,
not just hyperparameters.

## Mutable File
`train.py` — the training loop. The RL loop proposes unified diffs to this file each iteration.
Accepted diffs are permanently applied; rejected diffs are rolled back.

## Frozen File
`prepare.py` — data preparation infrastructure. Must not be modified.

## Metrics
- `val_bpb` (primary, lower is better)
- `loss` (secondary)
- `training_seconds` (budget tracking)
- `num_steps` (progress tracking)

## Known Issues in Current `train.py`
- Line 63: `use_qk_norm = True` is a dangling global that has no effect — it shadows the local
  variable inside `main()` but is never read. This is a bug worth fixing.
- The penalty landscape is hand-crafted around `learning_rate ≈ 2.6e-3`, `grad_clip ≈ 0.85`,
  `use_qk_norm = True`. The constants `130.0`, `0.05`, `0.3` are arbitrary and may not be optimal.
- `TIME_BUDGET_S = 1.5` controls training depth. Increasing it trades speed for accuracy.

## Guidance
- Propose minimal, surgical diffs — one focused change per proposal.
- Avoid changing the `print()` output format; the controller parses `val_bpb=...` from stdout.
- The sweet spot for `val_bpb` is around `1.05`. Can you get below it?
- Each iteration completes in ~1.5 seconds — aim for 10–20 iterations per session.
