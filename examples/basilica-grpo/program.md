# GRPO Fine-Tuning: Qwen2.5-0.5B on GSM8K

## Objective
Maximize GSM8K pass@1 by optimizing GRPO training hyperparameters for Qwen2.5-0.5B-Instruct.
Reported as `val_bpb = 1 - pass@1` (lower is better).

## Model
`Qwen/Qwen2.5-0.5B-Instruct` — a 500M parameter instruct model capable of multi-step reasoning.

## Dataset
GSM8K (Grade School Math): 7.5K training / 1.3K test problems requiring multi-step arithmetic.

## Mutable File
`train.py` — the GRPO training loop. The LLM proposes hyperparameter combinations across
iterations, building cumulative reasoning from the full experiment history.

## Frozen File
`prepare.py` — dataset loading and prompt formatting utilities. Must not be modified.

## Tunable Hyperparameters
| Parameter       | Typical Range   | Notes                                       |
|-----------------|-----------------|---------------------------------------------|
| learning_rate   | 1e-6 to 2e-5    | GRPO is sensitive; 5e-6 is a safe start     |
| batch_size      | 4 to 8          | Constrained by VRAM on 24GB GPU             |
| max_steps       | 20 to 100       | More steps = better signal but slower       |
| num_generations | 4 to 8          | GRPO rollout width; higher = better signal  |
| temperature     | 0.7 to 1.2      | Rollout temperature; affects diversity      |

## Guidance
- Baseline pass@1 before training is typically 30–45% for this model size.
- GRPO is sensitive to reward signal quality. The exact-match reward may under-reward partially
  correct multi-step reasoning.
- A learning rate of 5e-6 with `num_generations=4` is a reliable starting point.
- If val_bpb stops improving, try increasing `num_generations` or `max_steps` before raising lr.
- Each Basilica iteration takes ~15–20 minutes. The 4-hour wall time allows ~12 iterations.
