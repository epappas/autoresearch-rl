# GRPO Post-Training: Qwen2.5-0.5B on GSM8K

## Objective
Maximize GSM8K pass@1 accuracy by optimizing GRPO training for Qwen2.5-0.5B-Instruct.
Reported as `eval_score` (higher is better). Baseline pass@1 is typically 30-45%.

## Model
`Qwen/Qwen2.5-0.5B-Instruct` -- a 500M parameter instruct model capable of multi-step reasoning.

## Dataset
GSM8K (Grade School Math): 7.5K training / 1.3K test problems requiring multi-step arithmetic.

## Mutable File
`train.py` -- the GRPO training loop. You may modify training configuration, reward function
design, evaluation strategy, optimizer settings, and any other aspect of the training code.

## Frozen File
`prepare.py` -- dataset loading and prompt formatting utilities. Must not be modified.

## Tunable Hyperparameters (param mode)
| Parameter       | Choices         | Notes                                       |
|-----------------|-----------------|---------------------------------------------|
| learning_rate   | 3e-6, 5e-6, 1e-5 | GRPO is sensitive; 5e-6 is a safe start   |
| batch_size      | 2, 4            | Constrained by VRAM; 2 is safest           |
| max_steps       | 30, 50, 80      | More steps = better signal but slower       |
| num_generations | 4, 8            | GRPO rollout width; higher = better signal  |
| temperature     | 0.8, 1.0        | Rollout temperature; affects diversity      |

## Code Diff Guidance (diff mode)
When proposing code changes to train.py, consider these high-value modifications:

### Reward Function Engineering
- The current reward is binary (exact match = 1.0, else 0.0). This is sparse.
- Consider partial credit: award 0.5 if the reasoning steps are correct but the final answer has
  a minor arithmetic error.
- Consider format rewards: award 0.1-0.2 for responses that show step-by-step reasoning
  (containing "Step 1", numbered steps, or intermediate calculations).
- Consider length penalties to discourage overly verbose or truncated responses.

### Training Optimization
- Add gradient accumulation (`gradient_accumulation_steps`) to effectively increase batch size
  without increasing VRAM usage.
- Add warmup steps (`warmup_steps` or `warmup_ratio` in GRPOConfig).
- Experiment with `max_completion_length` -- shorter completions train faster but may truncate
  reasoning chains.
- Add learning rate scheduling (cosine, linear decay).

### Evaluation Improvements
- The current eval uses greedy decoding. Consider majority voting (sample N responses,
  take the most common answer) for more stable accuracy measurement.
- Improve `extract_answer` to handle more output formats (e.g., "The answer is 42",
  "Therefore, 42", boxed answers like `\boxed{42}`).

### Architecture/Optimizer Tweaks
- Add LoRA/QLoRA for parameter-efficient fine-tuning (requires `peft`).
- Experiment with different KL penalty coefficients if available in the GRPOConfig.

## Known Constraints
- Single GPU (A100 or H100), 24-80GB VRAM
- Each iteration should complete within 30 minutes
- Do not modify prepare.py or change the model to a different architecture
- Keep the eval metric as `eval_score` (pass@1 accuracy on GSM8K test set)
