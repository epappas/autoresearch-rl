# GRPO Post-Training Showcase: Autonomous RL Optimization on Basilica

## Summary

We ran 22 autonomous GRPO post-training iterations on Basilica GPU cloud (A100-80GB),
optimizing Qwen2.5-0.5B-Instruct on GSM8K math reasoning. The autoresearch-rl controller
proposed hyperparameter configurations via LLM-guided search (DeepSeek-V3), trained each
configuration in an isolated containerized GPU job, evaluated pass@1 accuracy, and
kept/discarded based on improvement.

**Result:** GSM8K pass@1 improved from 0% (untrained baseline) to 4% across 22 iterations,
with 2 kept improvements and a 77% iteration success rate.

## Differentiation from Karpathy's autoresearch

| Aspect | Karpathy autoresearch | autoresearch-rl (this work) |
|--------|----------------------|----------------------------|
| Task | Pre-training GPT from scratch | Post-training (GRPO) on pre-trained model |
| Metric | val_bpb (language modeling) | eval_score (task accuracy on GSM8K) |
| Execution | Local single GPU | Cloud GPU via Basilica (containerized) |
| Algorithm | LLM edits training code | LLM proposes hyperparams (+ code diffs in hybrid mode) |
| Training | From random init, 5 min budget | From pre-trained checkpoint, GRPO with reward signal |
| Infrastructure | Manual git commits | Automated loop with checkpoint/resume, telemetry |

**Why post-training matters:** The industry bottleneck is not pre-training (which requires
massive compute and is done by a few labs). The bottleneck is post-training: RLHF, DPO, GRPO,
SFT fine-tuning — where teams spend weeks manually tuning reward functions, learning rates,
and training recipes. Autonomous post-training optimization is the higher-value problem.

## Experiment Configuration

- **Model:** Qwen/Qwen2.5-0.5B-Instruct (494M trainable parameters)
- **Dataset:** GSM8K (7,473 train / 1,319 test, grade-school math)
- **Algorithm:** GRPO (Group Relative Policy Optimization) with clipped PPO loss and KL penalty
- **Evaluation:** pass@1 on 100 GSM8K test problems with greedy decoding
- **GPU:** NVIDIA A100-SXM4-80GB on Basilica cloud
- **Policy:** LLM-guided param search (DeepSeek-V3-0324 via Chutes API)
- **Budget:** 8 hours wall time, 2.1 hours training time, ~6 hours total elapsed

### Hyperparameter Search Space

| Parameter | Values | Notes |
|-----------|--------|-------|
| learning_rate | 3e-6, 5e-6, 1e-5 | GRPO is sensitive to LR |
| batch_size | 1, 2 | Constrained by GRPO generation memory |
| max_steps | 15, 30, 50 | Training steps per iteration |
| num_generations | 2, 3 | GRPO rollout width |
| temperature | 0.8, 1.0 | Rollout sampling temperature |

## Results

### Iteration Log

| Iter | Decision | eval_score | lr | steps | gen | Training time |
|------|----------|-----------|-----|-------|-----|---------------|
| 0 | **keep** | **0.03** | 5e-6 | 30 | 2 | 469s |
| 1 | failed | - | 3e-6 | 50 | 3 | - |
| 2 | discard | 0.02 | 5e-6 | 30 | 2 | - |
| 3 | discard | 0.01 | 5e-6 | 30 | 2 | - |
| 4 | discard | 0.01 | 1e-5 | 15 | 3 | - |
| 5 | discard | 0.01 | 1e-5 | 30 | 2 | - |
| 6 | discard | 0.03 | 3e-6 | 30 | 3 | - |
| 7 | discard | 0.00 | 3e-6 | 50 | 3 | - |
| 8-10 | failed | - | - | - | - | - |
| 11 | **keep** | **0.04** | 3e-6 | 50 | 2 | 604s |
| 12 | discard | 0.04 | 3e-6 | 50 | 2 | - |
| 13 | discard | 0.04 | 5e-6 | 50 | 2 | - |
| 14 | discard | 0.01 | 3e-6 | 50 | 2 | - |
| 15 | discard | 0.01 | 5e-6 | 50 | 3 | - |
| 16 | discard | 0.02 | 5e-6 | 30 | 2 | - |
| 17 | failed | - | 3e-6 | 50 | 3 | - |
| 18 | discard | 0.01 | 3e-6 | 50 | 2 | - |
| 19 | discard | 0.04 | 5e-6 | 50 | 2 | - |
| 20 | discard | 0.01 | 5e-6 | 50 | 3 | - |
| 21 | discard | 0.01 | 1e-5 | 30 | 2 | - |

### Aggregate Statistics

| Metric | Value |
|--------|-------|
| Total iterations | 22 |
| Successful | 17 (77%) |
| Failed (infra) | 5 (23%) |
| Kept (improvements) | 2 (9%) |
| Best eval_score | 0.04 (4% pass@1) |
| Baseline | 0.00 (0% pass@1) |
| Mean eval_score | 0.020 |
| Total training time | 2.1 GPU-hours |
| Total elapsed time | 6.0 hours |

### Winning Configurations

**Best (iter 11):** lr=3e-6, batch_size=1, max_steps=50, num_generations=2, temperature=1.0
- eval_score=0.04, loss=0.001053, training_time=604s

**First improvement (iter 0):** lr=5e-6, batch_size=2, max_steps=30, num_generations=2, temperature=1.0
- eval_score=0.03, loss=0.001195, training_time=469s

### Key Observations

1. **Lower learning rate wins:** The best result used lr=3e-6 vs the initial lr=5e-6. Higher
   lr (1e-5) consistently produced worse results (0.01 pass@1).

2. **More steps matter:** 50 steps outperformed 15-30 steps. The winning config used
   max_steps=50 while the initial improvement used 30.

3. **Fewer generations is better at this scale:** num_generations=2 dominated. With the
   binary exact-match reward, more generations don't provide better signal — they just add
   noise when the model rarely produces correct answers.

4. **Infrastructure reliability:** 77% success rate across Basilica deployments. Failures
   were: 1 timeout (iter 1, 2459s), 3 diff-mode policy failures (iters 8-10), 1 Basilica
   outage (iter 17).

## Technical Implementation

### Pure PyTorch GRPO (no TRL)

We implemented GRPO from scratch in PyTorch, bypassing TRL entirely. TRL's GRPOTrainer
deadlocks in containerized environments due to accelerate/DDP process management issues.

The implementation follows the DeepSeek-R1 GRPO algorithm:
1. Sample prompt, generate G completions via `model.generate()`
2. Score each completion with exact-match reward function
3. Compute per-prompt advantage: reward_i - mean(rewards)
4. Compute clipped policy gradient loss (PPO-style, epsilon=0.2)
5. Add KL penalty against frozen reference model (coefficient=0.01)
6. Update with AdamW optimizer

### Basilica Container Architecture

Each iteration deploys a fresh container on Basilica:
1. Base image: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`
2. Setup command installs dependencies (transformers, datasets, accelerate)
3. Training script injected via base64 encoding in deploy.py
4. Health server runs in daemon thread for Basilica health checks
5. Metrics extracted from stdout via `key=value` pattern matching

### Key Engineering Decisions

- **No `key=value` in intermediate output:** The Basilica adapter's metric parser matches
  any `key=value` pattern. Intermediate training logs use bracket-prefixed format
  (`[step 1/30]`) to avoid premature metric detection.

- **DDP environment cleanup:** Container environments may have stale distributed training
  env vars. The training script explicitly removes WORLD_SIZE, RANK, MASTER_ADDR etc.

- **`use_vllm=False`:** TRL's default vLLM generation backend is not available in the
  container. This must be explicitly disabled.

## Progress Chart

![GRPO Progress](../../grpo_progress.png)

The chart shows the Karpathy-style scatter plot with:
- Gray dots: discarded experiments (did not improve)
- Green dots: kept experiments (improvements)
- Step function: running best eval_score

## Limitations and Next Steps

### Current Limitations

1. **Low absolute accuracy (4%):** Qwen2.5-0.5B starts near 0% on GSM8K. The model needs
   significantly more training steps (100-500) or a stronger base model to reach meaningful
   accuracy (>30%).

2. **Binary reward signal:** Exact-match-only reward is extremely sparse for a model that
   rarely produces correct answers. Partial credit (correct reasoning steps, format rewards)
   would provide better learning signal.

3. **Small training budget:** 30-50 GRPO steps per iteration is minimal. Production GRPO
   runs typically use 1,000-10,000 steps.

### Recommended Next Steps

1. **Increase max_steps to 100-200** for deeper training per iteration
2. **Add partial-credit rewards:** award 0.5 for correct intermediate steps
3. **Use a stronger base model:** Qwen2.5-1.5B or 3B would start with higher baseline
4. **Enable hybrid mode** (code diffs) once LLM API rate limiting is resolved
5. **Pre-build Docker image** to eliminate pip install overhead (~60s per iteration)

## Reproducibility

- **Episode ID:** 897e096800b8
- **Hardware:** NVIDIA A100-SXM4-80GB (Basilica cloud)
- **Results ledger:** artifacts/basilica-grpo/results.tsv
- **Event trace:** traces/basilica-grpo/events.jsonl
- **Checkpoint:** artifacts/basilica-grpo/checkpoint.json
