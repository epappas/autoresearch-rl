# security-judge: LoRA + GRPO for LLM-as-Security-Judge

Train Qwen2.5-0.5B-Instruct to act as a structured security judge that outputs
`{"decision": "pass|block|warning", "security_score": 0.0-1.0}` for prompt injection
detection. Uses LoRA for parameter-efficient training and GRPO with a multi-component
reward (JSON compliance + decision accuracy + score calibration).

## Why this is novel

No published recipe exists for GRPO-training a small LLM as a structured security judge.
The reward function is non-trivial (multi-component, not binary), the output format is
structured JSON (not free text), and the task requires the model to calibrate confidence
scores alongside binary decisions. The hybrid policy's diff mode can evolve the reward
function autonomously when param search stalls.

## Prerequisites

```bash
export BASILICA_API_TOKEN="..."
export CHUTES_API_KEY="..."
```

## Run (Basilica)

```bash
python3 examples/security-judge/deploy.py
```

## Pipeline

```
prepare.py  -->  /app/data/{train,eval}.jsonl  -->  train.py  -->  [metrics]
(runs once)       (formatted judge prompts)         (each iter)    (keep/discard)
```

`prepare.py` converts raw security data into judge-formatted prompts with expected
structured verdicts. `train.py` applies LoRA, runs GRPO training, evaluates, and
prints metrics. No import between them.

## How it works

1. `hybrid` policy: LLM-guided param search (LoRA rank, LR, steps, generations, temp),
   switches to code diffs when param search stalls.
2. Each iteration deploys a Basilica A100 container with LoRA + GRPO training.
3. Multi-component reward: 0.3 (valid JSON) + 0.4 (correct decision) + 0.3 (calibrated score).
4. The LLM in diff mode can propose reward function improvements, LoRA target changes,
   or generation strategy modifications to `train.py`.

## Files

| File | Role |
|------|------|
| `train.py` | Mutable -- LoRA + GRPO training, reward function, generation |
| `prepare.py` | Frozen -- data formatting, verdict parsing, evaluation |
| `program.md` | Task guidance for the LLM (param + diff mode) |

## Metrics

| Metric | Description |
|--------|-------------|
| `eval_score` | Composite reward (primary, used for keep/discard) |
| `decision_accuracy` | Fraction of correct pass/block/warning decisions |
| `json_compliance` | Fraction of outputs that parse as valid structured verdicts |
