# Gap Analysis: prior work's autoresearch-rl vs. Our Implementation

**Date:** 2026-03-15
**Sources:** [prior-work/autoresearch-rl](https://github.com/prior-work/autoresearch-rl), 
**Method:** Three expert agents (LLM Architect, AI Engineer, MLOps Engineer) independently analyzed both codebases

---

## 1. Their Approach

the prior autoresearch-rl adaptation is a direct adaptation of Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) for RL post-training. The core insight: let an AI coding assistant autonomously run RL experiments overnight.

### Architecture (4 files)

| File | Role | Modified by agent? |
|------|------|-------------------|
| `prepare.py` | One-time setup (download model, verify GPUs) | No |
| `train.toml` | Full RL training config (optimizer, LR, loss, rollouts, etc.) | **Yes** |
| `run.py` | Experiment runner (launches an external RL framework, enforces time budget, extracts metrics) | No |
| `program.md` | Instructions for the LLM agent | By human only |

### How It Works

1. An LLM agent (Claude/Codex) receives `program.md` as instructions
2. The agent reads past results from `results.tsv`
3. The agent reasons about what hyperparameter change to try next
4. The agent edits `train.toml` and commits the change
5. The agent runs `uv run run.py > run.log 2>&1`
6. `run.py` launches [an external RL framework](https://github.com/external-org/an external RL framework) for actual GRPO training of Qwen2.5-0.5B-Instruct
7. After 10 minutes, metrics are extracted (eval_score = average Pass@1)
8. Keep (advance branch) or discard (git reset) based on improvement
9. Loop forever until human stops it

### Results

66 experiments over ~13 hours. Improved from **0.475 to 0.550** eval_score on GSM8K (Pass@1). Key findings:

| Finding | Detail |
|---------|--------|
| Best LR | 3e-6 (lower than initial 5e-6) |
| Best optimizer | AdamW (beat SGD and Muon) |
| Best scheduler | Constant (no warmup, no decay) |
| Best batch_size | 256 |
| Best max_steps | 20 |
| LoRA | Did not help at 0.5B scale |
| torch.compile | Did not help for short runs |
| Temperature | 1.0 (0.7 and 1.2 both worse) |

---

## 2. Fundamental Architectural Differences

### Who is the "agent"?

**Theirs:** The LLM coding assistant IS the outer loop. There is no programmatic optimization. The LLM reads results, reasons about hyperparameters using its training knowledge of ML, and edits the config file directly. The "policy" is literally the LLM's chain-of-thought.

**Ours:** We have a programmatic controller (`controller/continuous.py`) with pluggable policies (`GridPolicy`, `RandomPolicy`, `StaticPolicy`, `LearnedParamPolicy`). Decisions are mechanical: propose from distribution, run, compare score, keep/discard.

### What gets trained?

**Theirs:** A real LLM (Qwen2.5-0.5B-Instruct) via GRPO on math benchmarks using an external RL framework. Each experiment actually fine-tunes transformer weights on GPU. The agent optimizes the RL training hyperparameters.

**Ours:** The continuous loop is target-agnostic -- it calls external commands/URLs. The PPO/GAE/SDPO modules train a small numpy-based MLP (~20 parameters) to predict which hyperparameter configurations to try. This is meta-optimization over the search space, not RL training of a language model.

### Infrastructure coupling

**Theirs:** Tightly coupled to an external RL framework, 2 GPUs (GPU 0 = vLLM inference, GPU 1 = training), pynvml for VRAM monitoring. Purpose-built for one task.

**Ours:** Hardware-agnostic, framework-agnostic. CommandTarget + HttpTarget can drive any training stack. No GPU requirement. Designed as platform infrastructure.

---

## 3. What They Have That We Don't

### 3.1 Working RL Training Backend (CRITICAL)

Their `run.py` integrates with an external RL framework for actual GRPO training. They have:
- vLLM inference server lifecycle management
- an external RL framework training with GRPO/IPO/DPPO loss functions
- Real evaluation via `verifiers` library (GSM8K rubrics)
- Model checkpoint management between experiments
- GPU partitioning (CUDA_VISIBLE_DEVICES)

We have no working training backend. Our `examples/autoresearch-like/train.py` uses a `SyntheticBaselineTarget` that returns fabricated scores.

### 3.2 LLM-as-Agent Loop

Their LLM agent can:
- Reason about correlations between hyperparameters
- Apply ML domain knowledge (e.g., "LoRA won't help at 0.5B scale")
- Try unconventional changes based on intuition
- Adapt search strategy based on the trajectory of results

Our programmatic policies (grid/random) have no domain knowledge. The `LearnedParamPolicy` learns from rewards but starts from zero knowledge.

### 3.3 Fixed Time Budget Per Experiment

They enforce exactly 600 seconds per experiment (hardcoded in `prepare.py`) with a 120-second grace period. This makes all experiments directly comparable.

We have `max_wall_time_s` (total loop time) and `per_iteration_timeout_s` but no explicit per-experiment comparability enforcement.

### 3.4 Process Group Management

Their `run.py` uses `subprocess.Popen` with `os.setsid()` and kills the entire process group on timeout (`os.killpg`). Two-phase: SIGTERM, wait 5s, then SIGKILL.

Our `CommandTarget` uses `subprocess.run()` which sends SIGKILL to the child PID only, leaving child processes orphaned. The correct pattern exists in our `sandbox/runner.py` but isn't used by the active code path.

### 3.5 Real Experimental Results

66 experiments with empirical findings. We have zero real training runs.

---

## 4. What We Have That They Don't

### 4.1 Learned Meta-Policy (PPO over Hyperparameters)

`policy/ppo.py` + `policy/learned_search.py`: A trainable softmax distribution over discrete HP choices, updated via PPO with GAE advantages. This is a genuine meta-learning capability -- the system learns which hyperparameters to try based on trajectory feedback.

### 4.2 SDPO (Self-Distilled Policy Optimization)

`policy/sdpo.py`: DPO-style loss that learns from keep/discard preference pairs. Trains the search policy to prefer configurations that led to improvements.

### 4.3 SDFT (Softmax Divergence Fine-Tuning)

`distillation/sdft.py` + `distillation/trainer.py`: Distills the learned search policy into a compact probability table. Enables deploying the learned search strategy without the PPO machinery.

### 4.4 Comprehensive Telemetry

- JSONL event traces with structured iteration events
- TSV ledger with comparability metadata
- Hardware fingerprinting with strict mode
- Version artifacts with metadata
- Log rotation for long-running loops
- Metrics aggregation (mean, stdev, trend slope)
- Experiment tracking with history

### 4.5 Safety and Sandboxing

- AST-level diff policy blocking dangerous operations
- Git worktree isolation for trial execution
- Contract enforcement (frozen/mutable file boundaries)
- Power-law early-stop forecasting

### 4.6 Target Abstraction

`CommandTarget` + `HttpTarget` make the system target-agnostic. Can drive local commands, Docker containers, or remote vLLM/sglang endpoints.

### 4.7 Crash Recovery

Checkpoint save/load with atomic writes enables resume after interruption. They start from scratch on every crash.

### 4.8 Multiple Stop Guards

Wall time, no-improvement streak, failure rate threshold, max iterations, graceful shutdown (SIGINT/SIGTERM). They only have wall time.

---

## 5. Actionable Gaps

### P0 -- Must Close

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 1 | **No working RL training backend** | System cannot produce real results | High -- integrate an external RL framework or TRL |
| 2 | **No LLM-in-the-loop policy** | Missing the most powerful search strategy | Medium -- add LLMSearchPolicy |
| 3 | **Process group management in CommandTarget** | Orphaned child processes on timeout | Low -- port from sandbox/runner.py |

### P1 -- Should Close

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 4 | **No GPU device assignment** | Multi-GPU contention | Low -- inject CUDA_VISIBLE_DEVICES |
| 5 | **No grace period on timeout** | Training can't save checkpoint before kill | Low -- SIGTERM then SIGKILL |
| 6 | **No output directory cleanup** | Stale artifacts between iterations | Low -- rmtree before each run |
| 7 | **Last-match metric semantics** | extract_metric returns first match, not last | Low -- use re.findall()[-1] |
| 8 | **No per-experiment time budget config** | Experiments not directly comparable | Low -- add to ControllerConfig |

### P2 -- Nice to Have

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 9 | **VRAM monitoring** | Can't detect memory pressure | Medium -- optional pynvml |
| 10 | **Log file metric fallback** | Miss metrics written to files | Medium |
| 11 | **Config file template mode** | Agent edits TOML directly vs env vars | Medium |
| 12 | **Noise-aware keep/discard** | Improvements < 2% may be noise | Low |

---

## 6. Research Insights from Their 66 Experiments

### What We Can Learn

1. **Learning rate is the most impactful knob.** Our default search grid should center on [1e-6, 1e-5] for RL post-training, not the [1e-4, 1e-3] range typical for SFT.

2. **Constant scheduler wins for short runs.** With only 20 training steps, warmup and decay don't have time to help. Our system should default to constant LR for experiments under 100 steps.

3. **LoRA doesn't help at small scale.** At 0.5B parameters, full fine-tuning is feasible and better. LoRA's overhead (adapter management) isn't justified.

4. **AdamW is robust.** Despite alternatives (SGD, Muon), AdamW remains the reliable default. Our search space should include optimizer type but weight toward AdamW.

5. **Diminishing returns after ~30 experiments.** Most gains came early. This validates no-improvement-streak as a stopping criterion (our `no_improve_streak` guard).

6. **Batch size 256 is a sweet spot.** Larger batches provide more diverse rollouts per update. Our default should search [64, 128, 256, 512].

7. **Noise is real.** They warn that improvements < 2% may be noise. Our keep/discard should have a configurable significance threshold.

### Implications for Our LearnedParamPolicy

Their results provide a natural prior distribution. Rather than learning from scratch, we could initialize `LearnedParamPolicy` with logits reflecting their findings:
- Higher logit for lr=3e-6 than lr=1e-5
- Higher logit for batch_size=256 than batch_size=64
- AdamW weighted higher than SGD

This would accelerate convergence of our meta-learner.

---

## 7. Post-Basilica Integration: Updated Gap Status

### Gaps Closed

| # | Gap | Prior Status | Current Status | Evidence |
|---|-----|-------------|----------------|----------|
| 1 | No working RL training backend | CRITICAL | **CLOSED** | BasilicaTarget deployed on A100, GRPO train.py with TRL ready, metrics parsed end-to-end |
| 3 | Process group management in CommandTarget | P0 | **MITIGATED** | Basilica handles process lifecycle (container create/delete), no orphaned children |
| 4 | No GPU device assignment | P1 | **CLOSED** | BasilicaConfig.gpu_count/gpu_models configures GPU allocation |
| 6 | No output directory cleanup | P1 | **CLOSED** | Basilica deployments are deleted after each iteration, containers are ephemeral |
| 8 | No per-experiment time budget config | P1 | **CLOSED** | BasilicaConfig.ttl_seconds + target.timeout_s enforce per-experiment budgets |

### Gaps Remaining

| # | Gap | Status | Notes |
|---|-----|--------|-------|
| 2 | No LLM-in-the-loop policy | Open (P1) | `LearnedParamPolicy` provides PPO meta-learning but not LLM reasoning |
| 5 | No grace period on timeout | Open (P1) | Basilica handles this at container level, but CommandTarget still lacks it |
| 7 | Last-match metric semantics | Open (P2) | `_parse_metrics` uses findall (gets all matches), but CommandTarget's regex uses first match |
| 9 | VRAM monitoring | Open (P2) | Not yet integrated; Basilica tracks resource usage server-side |
| 12 | Noise-aware keep/discard | Open (P2) | No significance threshold in keep/discard logic |

### What Changed Architecturally

The Basilica integration closed the most critical gap identified by all three expert agents: **the system now has a working training backend that deploys on real GPU hardware**.

The architecture validates our dependency-agnostic design (MLOps Engineer's assessment): the orchestration layer (`continuous.py`) didn't change at all. The `BasilicaTarget` implements the same `TargetAdapter` protocol as `CommandTarget` and `HttpTarget`. The continuous loop, policies, telemetry, checkpointing, and all research modules (PPO, SDPO, SDFT, GAE, forecasting, promotion) work identically regardless of whether the target is a local command, HTTP endpoint, or Basilica deployment.

### Revised Strategic Assessment

The original assessment was:

> "They have a simple, working system that produces real results. We have a sophisticated, modular framework with novel research components but no working training backend."

**Updated assessment (post-Basilica):**

We now have a working training backend that deploys on production GPU infrastructure (A100/H100 via Basilica), verified end-to-end with metric parsing and keep/discard decisions. The critical gap is closed.

Our framework retains all architectural advantages over their approach:
- **Programmatic + learned search** (grid, random, static, PPO meta-learner) vs their LLM-only approach
- **Comprehensive telemetry** (JSONL traces, TSV ledger, aggregation, rotation, comparability) vs their plain TSV
- **Crash recovery** (atomic checkpoints with try/finally) vs their start-from-scratch
- **Target abstraction** (command, HTTP, Basilica) vs their an external RL framework-only coupling
- **Research modules** (SDPO, SDFT, GAE, forecasting, promotion) vs none

The remaining gap is the LLM-as-policy capability, which would combine our structured loop with the reasoning power of their approach. This is a P1 enhancement, not a blocker.

### Revised Summary Table

| Dimension | Theirs | Ours (post-Basilica) | Winner |
|---|---|---|---|
| RL training backend | an external RL framework (local 2-GPU) | Basilica cloud (A100/H100) | **Ours** (cloud-native, any GPU) |
| Agent intelligence | LLM reasoning | Grid/Random/PPO meta-learner | Theirs (for small spaces) |
| GPU management | Explicit 2-GPU split | Basilica manages allocation | Tie |
| Per-experiment budget | Fixed 600s | Configurable ttl_seconds + timeout_s | **Ours** (flexible) |
| Metrics extraction | Regex on stdout + log files | Structured JSON log parsing | **Ours** (more reliable) |
| Experiment tracking | results.tsv + git branches | JSONL + TSV + artifacts + checkpoints | **Ours** |
| Crash recovery | None | Atomic checkpoints + try/finally | **Ours** |
| Stop guards | Timeout only | Wall time + no-improve + failure rate + forecasting | **Ours** |
| Scalability | 1 machine, 2 GPUs | Cloud-native, any GPU count | **Ours** |
| Meta-learning | None | PPO + SDPO + SDFT + distillation | **Ours** |
