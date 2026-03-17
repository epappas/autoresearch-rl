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

## Constraints
- Single-GPU or CPU training; each iteration should complete in under 10 minutes.
- Model: `protectai/deberta-v3-base-prompt-injection-v2` (pre-trained checkpoint).
- Max sequence length: 256 tokens.

## Tunable Hyperparameters
| Parameter      | Typical Range              | Notes                                     |
|----------------|----------------------------|-------------------------------------------|
| learning_rate  | 1e-5 to 5e-5               | Transformer fine-tuning sweet spot        |
| epochs         | 1 to 3                     | Small dataset overfits quickly beyond 3   |
| weight_decay   | 0.0 to 0.1                 | Regularization; 0.01 is a safe default    |
| batch_size     | 4 or 8                     | Limited by memory; 8 is usually better    |
| grad_clip      | 0.5 or 1.0                 | Prevents gradient explosions              |

## Guidance
- Start with learning_rate=2e-5, epochs=2, weight_decay=0.01, batch_size=8, grad_clip=1.0.
- If val_bpb > 0.3 after first iteration, try increasing epochs or lowering learning_rate.
- Avoid epochs > 3 on this dataset size; diminishing returns and overfitting risk.
