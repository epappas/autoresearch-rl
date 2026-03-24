# Security Judge: LoRA + GRPO for Structured Prompt Injection Detection

## Objective
Train Qwen2.5-0.5B-Instruct to act as a security judge that outputs structured verdicts:
```json
{"decision": "pass|block|warning", "security_score": 0.0-1.0}
```

Maximize `eval_score` (composite reward: 0.3 valid JSON + 0.4 correct decision + 0.3 calibrated score).

## Model
`Qwen/Qwen2.5-0.5B-Instruct` with LoRA adapters (parameter-efficient, trains only ~0.5-2% of weights).

## Dataset
19,186 samples from 26 security benchmark datasets (deepset/prompt-injections + llmtrace collection).
Binary labels: 0 = safe content, 1 = prompt injection / jailbreak attempt.

## Mutable File
`train.py` -- the LoRA + GRPO training loop. You may modify:
- The reward function (component weights, new reward signals)
- LoRA configuration (rank, alpha, target modules, dropout)
- GRPO parameters (clip epsilon, KL coefficient)
- Optimizer settings (weight decay, scheduler)
- Generation parameters (max tokens, sampling strategy)

## Frozen File
`prepare.py` -- data loading, prompt formatting, verdict parsing, evaluation.

## Tunable Hyperparameters (param mode)
| Parameter       | Choices         | Notes                                       |
|-----------------|-----------------|---------------------------------------------|
| learning_rate   | 5e-5, 1e-4, 3e-4 | LoRA tolerates higher LR than full fine-tuning |
| max_steps       | 30, 50, 80      | Training steps per iteration                |
| num_generations | 2, 3            | GRPO rollout width                          |
| temperature     | 0.7, 0.9        | Rollout sampling temperature                |
| lora_rank       | 4, 8, 16        | Higher rank = more capacity, more memory    |

## Code Diff Guidance (diff mode)

### Reward Engineering
The current reward is: 0.3 (valid JSON) + 0.4 (correct decision) + 0.3 (score calibration).
Consider improvements:
- **Format penalty**: subtract 0.1 if the model produces any text outside the JSON object
- **Confidence calibration**: reward security_score closer to 0.0 for safe and 1.0 for attacks
  (sharper predictions)
- **Warning detection**: give partial credit (0.2) when the model says "warning" on a
  borderline sample that the ground truth marks as either pass or block
- **Length penalty**: the response should be exactly one JSON line; penalize multi-line output

### Training Improvements
- Add gradient accumulation to increase effective batch size
- Implement learning rate warmup (first 10% of steps)
- Try targeting more LoRA modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- Add LoRA to MLP layers: ["gate_proj", "up_proj", "down_proj"]

### Generation Strategy
- Reduce max_new_tokens from 64 to 32 (the expected output is ~50 characters)
- Try lower temperature (0.5) for more focused judge outputs
- Add repetition_penalty to prevent degenerate outputs

## Known Constraints
- Single GPU (A100), 24-80GB VRAM
- LoRA adapters only -- do not modify the base model weights
- Each iteration should complete within 30 minutes
- Output must be parseable as JSON with "decision" and "security_score" keys
