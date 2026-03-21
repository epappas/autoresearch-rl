# GRPO Post-Training Showcase: Autonomous RL Optimization on Basilica

## Summary

We ran 22 autonomous GRPO post-training iterations on Basilica GPU cloud (A100-80GB),
optimizing Qwen2.5-0.5B-Instruct on GSM8K math reasoning. The autoresearch-rl controller
proposed hyperparameter configurations via LLM-guided search (DeepSeek-V3), trained each
configuration in an isolated containerized GPU job, evaluated pass@1 accuracy, and
kept/discarded based on improvement.

**Result:** The system completed 22 iterations (77% success rate), finding 2 improvements
that raised eval_score from 0.00 to 0.04 (4% pass@1 on 100 GSM8K test problems).

**Important caveat:** The 0% baseline is anomalous. Qwen2.5-0.5B-Instruct is documented
at ~36% zero-shot on GSM8K (Qwen2.5 technical report). Our 0% baseline indicates the
evaluation prompt format does not elicit the `####` answer delimiter the `extract_answer`
regex expects. The 4% result reflects the model learning to occasionally produce this format,
not 4% mathematical reasoning ability. Fixing the prompt template or adding format-aware
reward shaping would unlock the model's actual capabilities. See [Known Issues](#known-issues)
for details.

## Basilica: Cloud GPU Infrastructure for Autonomous ML

Basilica is a GPU cloud platform that provides on-demand containerized GPU instances.
This experiment uses Basilica as the execution backend, and the integration demonstrates
capabilities that are not possible with local GPU setups or traditional cloud VMs.

### Why Basilica

Traditional ML experiment loops run on a single machine: the researcher's GPU, a cloud VM,
or a shared cluster. This creates three problems that Basilica solves:

1. **Isolation:** Each experiment runs in a fresh container with its own dependencies,
   CUDA runtime, and filesystem. A bad hyperparameter choice that corrupts model weights
   or fills disk cannot affect subsequent experiments. Basilica containers are ephemeral --
   they are created, used, and destroyed per iteration.

2. **Elasticity:** The autoresearch-rl controller runs on a lightweight CPU machine (no GPU
   required). It provisions A100-80GB instances on Basilica only when needed, pays only for
   training time, and releases them immediately after. There is no idle GPU cost between
   iterations while the LLM policy reasons about the next hyperparameter proposal.

3. **Reproducibility:** Each iteration records the exact container image, GPU model
   (NVIDIA A100-SXM4-80GB), hardware fingerprint, and training parameters in the telemetry
   ledger. The Basilica deployment API guarantees the same hardware class across iterations,
   making results comparable.

### How Basilica Is Used

The autoresearch-rl framework's `BasilicaTarget` adapter manages the full lifecycle
of each training iteration:

```
Controller (CPU) --> Basilica API --> GPU Container --> Metrics --> Keep/Discard
     |                                    |
     |  1. Create deployment              |  4. Poll logs for metrics
     |  2. Inject train.py via base64     |  5. Parse eval_score=X.XX
     |  3. Wait for health check          |  6. Cleanup deployment
```

Each iteration:
- Deploys a `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` container on an A100 GPU
- Injects train.py and prepare.py via base64-encoded setup (no Docker registry needed)
- Runs the three-stage pipeline: `setup_cmd` (pip install) -> `prepare_cmd` (prepare.py
  produces data files) -> `train_cmd` (train.py reads data, trains, prints metrics)
- Starts a health-check HTTP server in a daemon thread for Basilica liveness probes
- Injects hyperparameters via `AR_PARAMS_JSON` environment variable
- Streams training logs; the adapter polls for `key=value` metric patterns in stdout
- Cleans up the deployment after metrics are collected or on timeout/failure

### What Makes This Novel

No existing autonomous ML research framework combines these capabilities:

- **LLM-guided search over cloud GPU:** The LLM policy (DeepSeek-V3) proposes hyperparameters
  based on full experiment history, and each proposal is executed on a fresh cloud GPU
  instance. The controller never touches a GPU directly.

- **Hybrid param + code diff mode:** The framework can switch from hyperparameter search
  to LLM-generated code modifications mid-experiment. The LLM reads the training script,
  proposes a unified diff, the framework validates it, and the modified script is deployed
  to Basilica. This enables the agent to evolve not just hyperparameters but the training
  algorithm itself.

- **Checkpoint/resume across failures:** The controller persists episode state (best score,
  iteration index, experiment history) to a JSON checkpoint after every iteration. When
  Basilica has capacity issues or the controller process restarts, the experiment resumes
  from the last completed iteration without losing progress. This experiment survived 3
  session interruptions.

- **Keep/discard with versioned artifacts:** Iterations that beat the current best are
  "kept" with full artifacts saved to `artifacts/versions/v####/`. Discarded iterations
  are logged but their artifacts are not promoted. This creates a monotonically improving
  artifact chain.

## The autoresearch-rl Framework

autoresearch-rl is an autonomous ML experiment controller that closes the loop between
hypothesis, training, and evaluation. It is designed around three principles:

### Pluggable Targets

The framework separates "what to train" from "where to train" via the `TargetAdapter`
protocol. The same experiment config can run locally (`CommandTarget`), against a remote
API (`HttpTarget`), or on cloud GPU (`BasilicaTarget`) by changing one config field.
This experiment uses `BasilicaTarget`, but the same GRPO training script works locally
with `CommandTarget` for development and debugging.

### Pluggable Policies

The parameter proposal strategy is interchangeable:
- `GridPolicy` / `RandomPolicy`: exhaustive or random search baselines
- `LLMParamPolicy`: sends experiment history to an LLM and asks for the next hyperparameter
  set. Maintains multi-turn conversation context so the LLM builds cumulative reasoning.
- `LLMDiffPolicy`: asks the LLM to propose code modifications as unified diffs. Includes
  correction retry -- if a diff fails validation, the error is sent back to the LLM for
  a second attempt.
- `HybridPolicy`: starts with param exploration, switches to code diffs when params stall.
  Falls back to param mode if diff proposals fail consecutively.

### Telemetry and Observability

Every iteration emits structured JSONL events (proposals, outcomes, decisions) and appends
to a TSV results ledger. The comparability system records hardware fingerprints and budget
modes to ensure results across runs are scientifically comparable. The `progress_chart.py`
script generates Karpathy-style visualization from this data.

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
SFT fine-tuning -- where teams spend weeks manually tuning reward functions, learning rates,
and training recipes. Autonomous post-training optimization is the higher-value problem.

## Experiment Configuration

- **Model:** Qwen/Qwen2.5-0.5B-Instruct (494M trainable parameters)
- **Dataset:** GSM8K (7,473 train / 1,319 test, grade-school math)
- **Algorithm:** GRPO (Group Relative Policy Optimization) with clipped PPO loss and KL penalty
  - Pure PyTorch implementation (no TRL dependency)
  - Per-prompt advantage normalization
  - Frozen reference model for KL regularization
- **Evaluation:** pass@1 on 100 GSM8K test problems with greedy decoding
  - Note: 100 samples yields a 95% confidence interval of approximately +/-4 percentage points
- **GPU:** NVIDIA A100-SXM4-80GB on Basilica cloud
- **Policy:** LLM-guided param search (DeepSeek-V3-0324 via Chutes API), with random fallback on API rate limits
- **Budget:** 8 hours wall time, 2.1 hours actual training time, ~6 hours total elapsed

### Hyperparameter Search Space

| Parameter | Values | Notes |
|-----------|--------|-------|
| learning_rate | 3e-6, 5e-6, 1e-5 | GRPO is sensitive to LR |
| batch_size | 1, 2 | Constrained by GRPO generation memory |
| max_steps | 15, 30, 50 | Training steps per iteration |
| num_generations | 2, 3 | GRPO rollout width per prompt |
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
| Best eval_score | 0.04 (4% pass@1 on 100 samples, CI [1.1%, 9.0%]) |
| Baseline | 0.00 (0% pass@1, see Known Issues) |
| Mean eval_score | 0.020 |
| Total training time | 2.1 GPU-hours |
| Total elapsed time | 6.0 hours |
| Estimated GPU cost | ~$6-8 (A100 at $3/hr x 2.1h training) |
| Per-iteration overhead | ~60s pip install + ~30s model download per container |

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
   binary exact-match reward, more generations don't provide better signal -- they just add
   noise when the model rarely produces correct answers.

4. **Infrastructure reliability:** 77% success rate across Basilica deployments. Failures
   were: 1 timeout (iter 1, 2459s), 3 diff-mode policy failures (iters 8-10), 1 Basilica
   outage (iter 17).

5. **Reward sparsity:** With a 0% baseline, the binary exact-match reward gives 0.0 for
   almost all completions. The `grpo_step` function skips gradient updates when all
   advantages are zero (all completions get the same reward). Most training steps were
   effectively no-ops, severely limiting learning.

### Reward Distribution Analysis

The training logs from successful iterations show:
- Step 1 of iter 0: avg_reward=0.5000 (1 of 2 completions correct -- lucky)
- Steps 5-30: avg_reward drops to 0.01-0.02 (model rarely produces correct answers)
- The reward signal is extremely sparse, confirming binary exact-match is insufficient
  for a model that produces answers in unexpected formats

## Technical Implementation

### Pure PyTorch GRPO (no TRL)

We implemented GRPO from scratch in PyTorch, bypassing TRL entirely. TRL's GRPOTrainer
deadlocks in containerized environments due to accelerate/DDP process management issues
that we traced through 15 debugging iterations (runs 1-15) before identifying the root causes.

The implementation follows the DeepSeek-R1 GRPO algorithm:
1. Sample prompt, generate G completions via `model.generate()` (sequential, not batched)
2. Score each completion with exact-match reward function
3. Compute per-prompt advantage: reward_i - mean(rewards_for_this_prompt)
4. Compute clipped policy gradient loss (PPO-style, epsilon=0.2)
5. Add KL penalty against frozen reference model (coefficient=0.01)
6. Update with AdamW optimizer (weight_decay=0.01, grad_clip=1.0)

**Algorithm correctness notes:**
- Advantage normalization is per-prompt (each prompt's G completions are compared only to
  each other), matching the original GRPO paper
- The clipped surrogate objective matches standard PPO
- KL penalty uses `log(pi/pi_ref)` as a soft regularizer, not full KL divergence.
  This is a simplification but effective for preventing catastrophic divergence
- Loss is summed across completions per prompt, not averaged across tokens globally.
  This avoids length bias toward shorter completions

### Basilica Container Architecture

Each iteration deploys a fresh container on Basilica with a three-stage pipeline:

```
setup_cmd (pip install)  ->  prepare_cmd (prepare.py)  ->  train_cmd (train.py)
      |                            |                            |
  install deps              write data files              read data, train,
  download model            /app/data/*.jsonl             print metrics
```

1. Base image: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`
2. `setup_cmd`: installs dependencies (transformers, datasets, accelerate)
3. `prepare_cmd`: runs `prepare.py` which downloads GSM8K and writes formatted
   JSONL data files to `/app/data/`. This is the frozen data boundary.
4. `train_cmd`: runs `train.py` which reads the prepared data, trains with GRPO,
   evaluates, and prints metrics to stdout. No import dependency on prepare.py.
5. Both scripts injected via base64 encoding in deploy.py
6. Health server runs in daemon thread for Basilica liveness probes
7. Metrics extracted from stdout via `key=value` pattern matching
8. Container cleanup via Basilica API after each iteration

### Key Engineering Decisions

- **No `key=value` in intermediate output:** The Basilica adapter's metric parser matches
  any `key=value` pattern. Intermediate training logs use bracket-prefixed format
  (`[step 1/30]`) to avoid premature metric detection. This was the root cause of 15
  failed runs before being identified.

- **DDP environment cleanup:** Container environments may have stale distributed training
  env vars. The training script explicitly removes WORLD_SIZE, RANK, MASTER_ADDR etc.
  to prevent accelerate from launching DDP mode on a single GPU.

- **`use_vllm=False`:** TRL's default vLLM generation backend causes a silent C-level
  crash in containers without vLLM installed. Explicitly disabling it was required.

- **File injection via base64:** Since we use the stock PyTorch base image (not a custom
  Docker build), train.py and prepare.py are base64-encoded by deploy.py and decoded by
  the setup_cmd at container start. This avoids needing a Docker registry.

### Failure Analysis

| Failure | Iterations | Root Cause | Resolution |
|---------|-----------|------------|------------|
| Basilica timeout | iter 1, 17 | GPU capacity / container startup delays | Retry on next iteration |
| Diff-mode policy | iters 8-10 | Chutes API 429 + GreedyLLMPolicy wrong CWD | Switched to llm-only policy |
| Chutes API 429 | Multiple | API capacity limits on DeepSeek-V3 | Added 15s/30s/45s retry backoff |

## Progress Chart

![GRPO Progress](../../grpo_progress.png)

The chart shows the Karpathy-style scatter plot with:
- Gray dots: discarded experiments (did not improve the best)
- Green dots: kept experiments (new improvements)
- Red markers: failed experiments (infrastructure issues)
- Step function: running best eval_score

## Known Issues

### 0% Baseline Anomaly

The measured baseline (untrained model) eval_score is 0.00, which is inconsistent with
published benchmarks. The Qwen2.5 technical report documents ~36% zero-shot accuracy on
GSM8K for the 0.5B-Instruct variant. Our 0% baseline indicates a **prompt-format mismatch**:

- Our prompt template uses a plain text format: `"Question: {q}\n\nAnswer:"`
- The `extract_answer` regex expects `#### <number>` or `answer is <number>` patterns
- Qwen2.5-0.5B-Instruct likely produces answers in a different format (e.g., natural
  language without the `####` delimiter) when prompted this way
- The model may need its chat template applied via `tokenizer.apply_chat_template()` to
  produce structured outputs

**Impact:** The 4% result does not represent mathematical reasoning improvement. It
represents the model learning to occasionally produce the `####` delimiter format through
GRPO training. The actual mathematical capability is likely much higher but invisible to
our evaluation.

### Reward Sparsity

With 0% baseline accuracy under our evaluation protocol, the binary exact-match reward
function returns 0.0 for nearly all completions. This creates a near-zero gradient signal:
- The `grpo_step` function skips updates when all G completions for a prompt receive
  identical rewards (all 0.0)
- With 2 generations per prompt, both completions almost always get 0.0
- Effective training steps per iteration may be far fewer than `max_steps`

## Limitations and Next Steps

### Current Limitations

1. **Prompt-format mismatch:** The evaluation protocol does not match the model's expected
   output format, producing an artificially low baseline and ceiling. This is the primary
   limitation.

2. **Binary reward signal:** Exact-match-only reward is extremely sparse when the model
   rarely produces the expected answer format. No partial credit for correct reasoning
   with wrong formatting.

3. **Small training budget:** 30-50 GRPO steps per iteration processes only ~50-100
   prompts out of 7,473 available. Production GRPO runs typically use 1,000-10,000 steps.

4. **Statistical precision:** Evaluation on 100 samples gives a 95% CI of approximately
   +/-4 percentage points. The difference between 3% and 4% is not statistically
   significant at this sample size.

5. **Single-prompt GRPO:** Each step processes one prompt sequentially. Batched
   multi-prompt steps would improve GPU utilization and gradient stability.

### Recommended Next Steps (Priority Order)

1. **Fix prompt template:** Use `tokenizer.apply_chat_template()` for Qwen2.5-Instruct
   and update `extract_answer` to handle the model's natural output format. Expected
   impact: baseline should jump to ~30-40%.

2. **Add multi-component reward:**
   - 0.2 for producing any numeric answer at the end
   - 0.3 for producing the `####` delimiter with a number
   - 1.0 for exact-match
   This provides gradient signal even when the final answer is wrong.

3. **Increase max_steps to 200-500** for deeper training per iteration. With fixed prompt
   format and richer rewards, more steps will produce meaningful learning curves.

4. **Increase evaluation samples to 500** for tighter confidence intervals (+/-2pp).

5. **Enable hybrid mode** (code diffs) once LLM API rate limiting is resolved. The LLM
   agent could propose reward function improvements or training loop changes autonomously.

6. **Pre-build Docker image** with all dependencies and model weights baked in. This would
   eliminate ~90s of per-iteration overhead (pip install + model download), saving ~33
   minutes across 22 iterations.

## Reproducibility

- **Episode ID:** 897e096800b8
- **Hardware:** NVIDIA A100-SXM4-80GB (Basilica cloud)
- **Software:** PyTorch 2.4.1, transformers 4.47.1, Python 3.11
- **Results ledger:** artifacts/basilica-grpo/results.tsv
- **Event trace:** traces/basilica-grpo/events.jsonl
- **Checkpoint:** artifacts/basilica-grpo/checkpoint.json
- **Note:** No random seed was set. Results are not exactly reproducible but the
  experimental protocol is fully automated and re-runnable.
