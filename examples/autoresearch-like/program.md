# AutoResearch-Like Training Program

## Objective
Minimize val_bpb through hyperparameter optimization of the training script.

## Mutable File
`train.py` - the training script that can be modified by the RL loop.

## Frozen File
`prepare.py` - data preparation infrastructure, must not be modified.

## Metrics
- `val_bpb` (primary, lower is better)
- `loss` (secondary)
- `training_seconds` (budget tracking)
- `num_steps` (progress tracking)
