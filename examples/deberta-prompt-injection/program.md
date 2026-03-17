# DeBERTa Prompt-Injection Defense

## Task
Fine-tune DeBERTa-v3-base for binary classification of prompt-injection attacks.
The model distinguishes safe user inputs from adversarial prompt injections.

## Objective
Minimize `val_bpb = 1 - f1` (lower is better). A perfect classifier yields val_bpb = 0.

## Dataset
- Small JSONL corpus (~hundreds of examples) with `text` and `label` fields.
- Binary labels: 0 = safe, 1 = injection.
- Class imbalance is moderate; F1 is the primary metric for this reason.

## Mutable File
`train.py` — the fine-tuning script. The hybrid policy proposes hyperparameter changes (first N
iterations) then code diffs (when stalled).

## Frozen File
`prepare.py` — data loading and tokenization infrastructure. Must not be modified.

## Constraints
- Single-GPU or CPU training; each iteration should complete in under 10 minutes.
- Model: `protectai/deberta-v3-base-prompt-injection-v2` (pre-trained checkpoint).
- Max sequence length: 256 tokens.

## Hyperparameter Guidance
| Parameter      | Typical Range   | Notes                                  |
|----------------|-----------------|----------------------------------------|
| learning_rate  | 1e-5 to 5e-5    | Transformer fine-tuning sweet spot     |
| epochs         | 1 to 3          | Small dataset overfits quickly past 3  |
| weight_decay   | 0.0 to 0.1      | 0.01 is a safe default                 |
| batch_size     | 8 or 16         | 8 is safer on limited memory           |
| grad_clip      | 0.5 or 1.0      | Prevents gradient explosions           |

Start with: `learning_rate=2e-5`, `epochs=2`, `weight_decay=0.01`, `batch_size=8`, `grad_clip=1.0`.

## Code Diff Guidance
After the initial hyperparameter exploration phase, the policy switches to proposing code diffs.
Useful areas to explore:
- `compute_metrics`: alternative metrics beyond F1 (e.g., precision-recall tradeoff).
- `TrainingArguments`: warmup steps, scheduler type, gradient accumulation.
- Class weighting via `compute_class_weight` to handle imbalance.
- The `max_length=256` truncation strategy — padding/truncation direction.
