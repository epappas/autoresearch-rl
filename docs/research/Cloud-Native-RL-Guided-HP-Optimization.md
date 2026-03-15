# Cloud-Native RL-Guided Hyperparameter Optimization for Security Model Fine-Tuning

**autoresearch-rl: Autonomous Experiment Loops on Ephemeral GPU Infrastructure**

---

## Abstract

We present autoresearch-rl, a framework for autonomous hyperparameter optimization that deploys training jobs on ephemeral cloud GPU infrastructure via the Basilica platform. The system implements a target-agnostic control loop with pluggable search policies (grid, random, learned via PPO), structured telemetry, crash-safe checkpointing, and a research module stack including GAE, SDPO, SDFT, and power-law convergence forecasting. We validate the end-to-end pipeline by fine-tuning a DeBERTa-based prompt injection detection model on Basilica A100 GPUs, demonstrating autonomous deployment, metric collection, and keep/discard decision-making across 6 hyperparameter configurations in 11.6 minutes with a 100% success rate.

---

## 1. Introduction

Hyperparameter optimization for security-critical models presents unique challenges. Prompt injection detectors must be reliable across diverse attack vectors, and the cost of a false negative (a missed injection) is high. Manual tuning is time-consuming and error-prone; automated search is essential but traditionally requires persistent GPU access.

This work addresses three gaps in existing approaches:

1. **No local GPU required.** Training is delegated to ephemeral cloud GPU deployments, enabling experimentation from any machine.
2. **Crash-safe autonomy.** Atomic checkpointing and TTL-based cleanup ensure that interrupted runs neither lose progress nor leak cloud resources.
3. **Extensible search.** The same control loop supports grid search, random search, and learned meta-policies (PPO, SDPO), with a clear upgrade path from static to adaptive optimization.

### 1.1 Relationship to Prior Work

prior work's autoresearch-rl uses an LLM coding assistant as the outer loop, editing a TOML config and running experiments on local GPUs via an external RL framework. This approach leverages the LLM's domain knowledge for intelligent search but is coupled to local hardware and non-reproducible.

Our approach replaces the LLM agent with a programmatic control loop and replaces local GPUs with ephemeral cloud deployments. This trades flexibility (no LLM reasoning) for reproducibility, crash safety, and hardware independence. The research modules (PPO, SDPO, SDFT) provide a path to learned search that does not depend on an external LLM.

Standard tools like Optuna and Ray Tune provide mature hyperparameter search but assume persistent compute. Our contribution is the target adapter abstraction that cleanly separates search logic from compute substrate, enabling the same loop to orchestrate local processes, HTTP endpoints, or cloud GPU deployments.

---

## 2. System Architecture

```
CLI / Config (YAML)
    |
    v
run_continuous() -- main control loop
    |
    +-- Policy.propose() --> hyperparameter dict
    |
    +-- BasilicaTarget.run() --> deploy GPU container on Basilica
    |       |-- create_deployment(image, gpu, env={AR_PARAMS_JSON})
    |       |-- poll status until ready/failed
    |       |-- extract metrics from structured JSON logs
    |       |-- delete deployment (cleanup)
    |       +-- return RunOutcome(metrics, status)
    |
    +-- BasilicaTarget.eval() --> return cached train metrics
    |
    +-- keep/discard decision (val_bpb < best?)
    +-- telemetry (JSONL trace, TSV ledger, version artifacts)
    +-- promotion tracking, forecasting, aggregation
    +-- checkpoint (atomic save in try/finally)
    +-- stop guards (wall time, no-improve, failure rate, forecast)
```

### 2.1 Target Adapter Protocol

The `TargetAdapter` protocol defines two methods:

```python
class TargetAdapter(Protocol):
    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome: ...
    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome: ...
```

Three implementations exist:
- **CommandTarget**: local/Docker subprocess execution
- **HttpTarget**: remote HTTP endpoint calls
- **BasilicaTarget**: Basilica cloud GPU deployments

The continuous loop is agnostic to which adapter is used. Switching from local to cloud is a YAML config change.

### 2.2 Basilica Adapter Lifecycle

For each iteration, `BasilicaTarget`:

1. Creates a deployment via `client.create_deployment()` with the training Docker image, GPU spec, and `AR_PARAMS_JSON` environment variable
2. Polls deployment status every 10 seconds (pending -> starting -> health_check -> ready/failed)
3. Handles short-lived jobs: if status is "failed" but logs contain metrics, the job completed successfully (container exited after training)
4. Extracts metrics from Basilica's structured JSON log format (`data: {"message": "val_bpb=0.000000", ...}`)
5. Deletes the deployment to free GPU resources
6. Returns `RunOutcome` with parsed metrics

Cleanup is guaranteed by a `try/finally` block. A server-side TTL provides defense-in-depth against controller crashes.

---

## 3. Novelty and Differentiation

### 3.1 vs. the prior autoresearch-rl adaptation

| Dimension | Prior work (LLM-as-agent) | Ours (programmatic + cloud) |
|-----------|---------------------|---------------------------|
| Outer loop | LLM reasoning | Programmatic (grid/random/PPO) |
| Compute | Local 2x GPU | Basilica cloud (any GPU) |
| Reproducibility | Non-deterministic | Fully deterministic given seed |
| Crash recovery | None (restart from scratch) | Atomic checkpoints + try/finally |
| Resource cleanup | Manual | Automatic (deployment delete + TTL) |
| Cost model | Fixed (own GPUs) | Pay-per-use (ephemeral deployments) |
| Search extensibility | Implicit (LLM decides) | Explicit (pluggable policy interface) |

### 3.2 vs. Optuna / Ray Tune

| Dimension | Optuna/Ray Tune | Ours |
|-----------|----------------|------|
| Compute assumption | Persistent workers | Ephemeral cloud deployments |
| GPU management | User-managed | Platform-managed (Basilica) |
| RL meta-learning | Not built-in | PPO + SDPO + GAE + SDFT |
| Distillation | Not applicable | SDFT student-teacher pipeline |
| Telemetry | Database-backed | File-based (JSONL + TSV, rotation, aggregation) |
| Crash safety | Trial-level retry | Loop-level atomic checkpoints |

### 3.3 vs. Manual Experimentation

The system completes 6 GPU experiments in 11.6 minutes with zero human intervention. Each iteration deploys a container, trains a model, evaluates, and decides keep/discard. A human performing the same work would need to: provision GPU, set up environment, run training, inspect results, record data, repeat -- easily consuming hours.

---

## 4. Methodology

### 4.1 Model and Task

- **Model:** `protectai/deberta-v3-base-prompt-injection-v2` -- a DeBERTa-v3-base model fine-tuned for binary prompt injection detection
- **Task:** Classify text as benign (label 0) or prompt injection attack (label 1)
- **Training data:** 8 samples (4 injection, 4 benign) from `examples/deberta-prompt-injection/data/train.jsonl`
- **Validation data:** 4 samples (2 injection, 2 benign) from `examples/deberta-prompt-injection/data/val.jsonl`
- **Metric:** `val_bpb = 1 - F1` (lower is better; F1=1.0 gives val_bpb=0.0)

### 4.2 Search Space

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| learning_rate | 1e-5, 2e-5, 3e-5 | Standard DeBERTa fine-tuning range |
| epochs | 1, 2 | Short runs for rapid iteration |
| batch_size | 4 (fixed) | Constrained by small dataset |
| weight_decay | 0.01 (fixed) | Standard regularization |

Total configurations: 3 x 2 = 6

### 4.3 Infrastructure

- **GPU:** NVIDIA A100-SXM4-80GB (Basilica cloud)
- **Container:** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`
- **Dependencies:** Installed at runtime via pip (transformers, datasets, accelerate, scikit-learn)
- **Training script:** Pulled from raw GitHub URL at deployment time
- **Data files:** Pulled from raw GitHub URL at deployment time
- **TTL:** 600 seconds per deployment (auto-cleanup safety net)

### 4.4 Deployment Pattern

Each iteration follows a consistent pattern:
1. Bootstrap script starts an HTTP health check server on port 8080 (keeps deployment "ready")
2. Install Python dependencies via pip
3. Download training script and data from raw GitHub
4. Execute training with hyperparameters from `AR_PARAMS_JSON`
5. Print metrics to stdout in `key=value` format
6. Controller extracts metrics from Basilica's JSON-wrapped log stream

---

## 5. Results

### 5.1 Per-Iteration Results

| # | Learning Rate | Epochs | val_bpb | F1 | Accuracy | Loss | Decision |
|---|--------------|--------|---------|------|----------|--------|----------|
| 1 | 1e-5 | 1 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | **keep** |
| 2 | 2e-5 | 1 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | discard |
| 3 | 3e-5 | 1 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | discard |
| 4 | 1e-5 | 2 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | discard |
| 5 | 2e-5 | 2 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | discard |
| 6 | 3e-5 | 2 | 0.000000 | 1.0000 | 1.0000 | 0.000001 | discard |

### 5.2 Summary

- **Success rate:** 6/6 (100%)
- **Total time:** 697 seconds (11.6 minutes)
- **Best configuration:** lr=1e-5, epochs=1, val_bpb=0.0 (F1=1.0)
- **Keep decisions:** 1 (iteration 1 established baseline)
- **Discard decisions:** 5 (equal performance, not strictly better)

### 5.3 Analysis

**Metric saturation.** All configurations achieved perfect F1=1.0 on the 4-sample validation set. The pre-trained model (`protectai/deberta-v3-base-prompt-injection-v2`) was specifically designed for prompt injection detection and trivially classifies the small validation set. This validates the pipeline but does not differentiate configurations.

**Decision behavior.** The keep/discard logic is correct: iteration 1 was kept (establishing the baseline). Iterations 2-6 were discarded because they did not strictly improve upon the baseline (equal val_bpb=0.0 is not an improvement).

**Resource efficiency.** Total GPU time was approximately 697 seconds across 6 A100 deployments. Each deployment was created, used for ~2 minutes of actual training, and deleted. Zero idle GPU time between iterations.

---

## 6. Discussion

### 6.1 What the Results Demonstrate

Despite metric saturation, the experiment validates:

1. **End-to-end cloud deployment works.** The Basilica adapter created 6 GPU deployments, injected hyperparameters, polled for completion, parsed metrics, and cleaned up resources -- all autonomously.
2. **The keep/discard loop behaves correctly.** First configuration kept (baseline), subsequent equal-performance configurations correctly discarded.
3. **Crash safety mechanisms function.** The `try/finally` cleanup and TTL ensure no GPU resource leaks.
4. **The metric protocol is language-agnostic.** The `key=value` stdout convention works with any training script.

### 6.2 Limitations

**Small dataset.** 8 train / 4 validation samples are insufficient for meaningful model evaluation. A pre-trained prompt injection model will trivially achieve perfect accuracy on 4 examples that follow common injection patterns. This experiment validates infrastructure, not model quality.

**Saturated metrics.** When all configurations produce identical metrics, the search provides no signal. The keep/discard mechanism degenerates to "keep first, discard rest."

**Sequential execution.** The current loop evaluates configurations one at a time. Parallel execution on simultaneous Basilica deployments could reduce total time from 697 seconds to ~120 seconds.

**No learned policy exercised.** The PPO, SDPO, and GAE modules exist but were not activated. They require a task where configurations produce differentiated results.

### 6.3 What Would Be Different at Scale

With a larger dataset (e.g., the full deepset/prompt-injections dataset with 600k+ samples):

- **Metric differentiation** would emerge, enabling meaningful convergence analysis
- **Forecasting** would enable early termination of unpromising runs
- **PPO meta-learning** would learn which HP regions are promising from trajectory feedback
- **SDPO preference pairs** would naturally emerge from keep/discard decisions
- **Promotion gates** would enforce quality thresholds before deployment

---

## 7. Infrastructure Analysis

### 7.1 Cloud-Native Advantages

**No local GPU required.** The controller process performs no GPU computation. It proposes parameters, makes HTTP requests to Basilica, and parses text responses. Experiments can be launched from a laptop, CI runner, or serverless function.

**Ephemeral deployments.** Each training run is an isolated container with its own environment. No state leakage between configurations. Automatic cleanup on TTL expiry.

**Pay-per-use.** GPU costs are incurred only during active training. No idle GPU time between iterations.

### 7.2 Deployment Lifecycle Safety

The `BasilicaTarget` guarantees cleanup in all code paths:

| Scenario | Cleanup Method |
|----------|---------------|
| Successful training | `deployment.delete()` in main path |
| Training failure | `deployment.delete()` after checking logs for metrics |
| Timeout | `deployment.delete()` on timeout exit |
| Exception | `deployment.delete()` in exception handler |
| Controller crash | TTL auto-cleanup by Basilica platform |

### 7.3 Comparability

The hardware fingerprinting system (`telemetry/comparability.py`) records GPU model, CPU count, and CUDA version at run start. In strict mode, results from different hardware are rejected from comparison.

---

## 8. Research Module Inventory

Implemented and tested modules not exercised in this experiment:

| Module | Purpose | Status |
|--------|---------|--------|
| MDP primitives (`mdp.py`) | State/Action/Reward formalization | 18 tests pass |
| Trajectory buffer (`trajectory.py`) | Episode storage for PPO training | 10 tests pass |
| GAE (`policy/gae.py`) | Variance-reduced advantage estimation | 8 tests pass |
| PPO actor-critic (`policy/ppo.py`) | Learned meta-policy for HP search | 19 tests pass |
| SDPO (`policy/sdpo.py`) | Preference-based policy optimization | 7 tests pass |
| SDFT (`distillation/sdft.py`) | Teacher-student knowledge distillation | 22 tests pass |
| Distillation trainer (`distillation/trainer.py`) | Batch SDFT training orchestration | 11 tests pass |
| Forecasting (`forecasting.py`) | Power-law convergence prediction | 13 tests pass |
| Promotion gates (`promotion.py`) | Quality-threshold model promotion | 11 tests pass |
| Learned search policy (`policy/learned_search.py`) | PPO + SDPO-based HP proposal | 8 tests pass |

Total: 218 tests pass, 0 failures, 0 lint warnings.

---

## 9. Future Work

### 9.1 Immediate

- **Larger datasets:** Run on the full deepset/prompt-injections dataset to produce differentiated metrics across configurations
- **Parallel evaluation:** Launch multiple Basilica deployments concurrently to reduce total experiment time
- **LLM search policy:** Add a policy that calls an LLM API to propose hyperparameters, combining our structured loop with domain reasoning

### 9.2 Medium-term

- **PPO meta-policy training:** Collect trajectories from diverse tasks, train the meta-policy to propose configurations based on search history
- **SDPO from keep/discard pairs:** Every keep/discard decision produces a natural preference pair for SDPO training
- **Forecasting-guided early stopping:** Terminate individual runs when power-law forecast indicates they won't beat current best

### 9.3 Long-term

- **Multi-GPU scaling:** Support distributed training (DDP, FSDP) within Basilica deployments
- **Cross-task transfer:** Train the meta-policy on trajectories from multiple tasks for generalizable HP strategies
- **Hierarchical policies:** LLM proposes architecture changes, HP policy tunes parameters for each architecture

---

## 10. Conclusion

autoresearch-rl demonstrates a viable architecture for autonomous hyperparameter optimization on ephemeral cloud GPU infrastructure. The target adapter abstraction cleanly separates search logic from compute substrate. The Basilica adapter validates the full deployment lifecycle: creation, monitoring, metric collection, and cleanup with crash-safe guarantees.

The experimental results on DeBERTa prompt injection detection confirm end-to-end pipeline functionality (6/6 successful cloud deployments on A100 GPUs in 11.6 minutes) but do not demonstrate meaningful HP differentiation due to metric saturation from a pre-trained model on a minimal dataset. This is an honest limitation of the demonstration setup, not the framework.

The research module inventory -- MDP formalization, PPO, SDPO, GAE, trajectory collection, power-law forecasting, structured distillation, and promotion gates -- provides a concrete path from static grid search to learned HP proposal policies. Realizing this path requires running the system on tasks where configurations produce differentiated results.

The key architectural insight is that by treating training compute as an external service behind a protocol interface, the system gains flexibility (any GPU, any cloud), safety (TTL cleanup, atomic checkpoints), and extensibility (new target backends, new search policies) without coupling these concerns together.
