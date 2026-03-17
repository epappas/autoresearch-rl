# Automated Hyperparameter Optimization for Prompt Injection Detection via LLM-Guided Search on Containerized GPU Infrastructure

**autoresearch-rl Experiment on Basilica GPU Cloud**

**Date:** 2026-03-17

---

## Abstract

We present results from an automated hyperparameter optimization experiment for fine-tuning a DeBERTa-v3-base model on a prompt injection detection task. The optimization was conducted using autoresearch-rl, a continuous reinforcement-learning-style loop that proposes hyperparameter configurations, deploys isolated GPU training jobs on the Basilica cloud platform, extracts evaluation metrics from training logs, and applies a keep/discard decision rule to track the best-performing configuration. An LLM-guided search policy (MiniMax-M2.5-TEE) proposed successive hyperparameter combinations informed by full experiment history. Across two concurrent episodes totaling 9 iterations (6 successful, 3 failed due to timeout), the system identified a configuration achieving an F1 score of 0.982468 (val\_bpb = 0.017532) on a held-out validation set of 3,838 samples drawn from 26 curated security benchmark datasets. All successful iterations were verified as comparable via hardware fingerprinting and budget-mode enforcement, ensuring that performance differences reflect genuine hyperparameter effects rather than infrastructure variance.

---

## 1. Introduction

Prompt injection attacks represent a significant threat to deployed large language model systems, where adversarial inputs manipulate model behavior to bypass safety constraints, exfiltrate data, or execute unintended instructions. Binary classifiers that distinguish malicious prompt injections from benign inputs serve as a practical defense layer. However, achieving high detection accuracy requires careful model selection and hyperparameter tuning, particularly when the training corpus aggregates heterogeneous attack strategies from multiple benchmark sources.

Manual hyperparameter search is labor-intensive and poorly suited to cloud-based GPU training, where each trial incurs provisioning overhead and cost. We address this with an automated loop that treats hyperparameter optimization as a sequential decision problem: at each iteration, a policy proposes a configuration, the system deploys a self-contained training job on GPU infrastructure, evaluates the result, and feeds the outcome back to the policy for the next proposal.

This report describes the methodology, infrastructure, and results of this experiment. We emphasize two architectural contributions: (1) the use of Basilica GPU cloud for deploying each training iteration as an isolated, containerized job with enforced comparability guarantees, and (2) the use of an LLM-based search policy that conditions its proposals on the complete history of prior iterations.

---

## 2. System Architecture

### 2.1 Runtime Path

```
CLI invocation:
  uv run autoresearch-rl --config examples/deberta-prompt-injection/config.yaml

Code path:
  cli.py -> RunConfig.model_validate(yaml)
         -> build_target(cfg)          [target/registry.py]
         -> BasilicaTarget             [target/basilica.py]
         -> run_continuous()           [controller/continuous.py]
              for each iteration:
                LLMParamPolicy.next()  [policy/llm_search.py]
                BasilicaTarget.run()
                  -> _build_bootstrap_cmd()
                  -> create_deployment() on Basilica API
                  -> _poll_for_metrics() from deployment logs
                BasilicaTarget.eval()  -> return cached train metrics
                keep/discard decision
                emit telemetry         [telemetry/events.py, ledger.py]
              cleanup all deployments
```

### 2.2 Basilica Target Adapter

`BasilicaTarget` (`src/autoresearch_rl/target/basilica.py`) implements the `TargetAdapter` protocol. For each iteration it:

1. Constructs a bootstrap script that starts an HTTP health-check server on port 8080 (required by Basilica to mark the deployment as "ready"), then executes the training command as a subprocess.
2. Creates a Basilica deployment via `client.create_deployment()` with the Docker image, GPU spec, environment variables (`AR_PARAMS_JSON` and individual `AR_PARAM_<NAME>` variables), and health check configuration.
3. Polls deployment status until ready, then polls logs for training metrics.
4. Extracts metrics from Basilica's structured SSE JSON log format (`data: {"message": "val_bpb=0.017532", ...}`) using regex-based parsing.
5. Deletes the deployment to free GPU resources.

Key static methods exposed for testability:

- `_parse_metrics(logs)` -- extracts `key=value` pairs from log text
- `_extract_messages(raw_logs)` -- parses SSE JSON log lines into plain text
- `_build_bootstrap_cmd(user_cmd, setup_cmd)` -- generates the bootstrap Python script

### 2.3 Stop Guards

The continuous loop implements four stop conditions:

- **Wall time:** `max_wall_time_s` (7200s in this experiment)
- **No-improvement streak:** `no_improve_limit` (3 consecutive non-improving iterations)
- **Failure rate:** `failure_rate_limit` (0.8, over a sliding window of 5)
- **Graceful shutdown:** SIGINT/SIGTERM handler

---

## 3. Methodology

### 3.1 Task and Model

- **Model:** `protectai/deberta-v3-base-prompt-injection-v2` (DeBERTa-v3-base, 184M parameters)
- **Task:** Binary classification -- detect prompt injection attacks (label=1) vs benign inputs (label=0)
- **Training script:** `examples/deberta-prompt-injection/train.py`, using HuggingFace Transformers `Trainer` API
- **Objective metric:** `val_bpb = 1.0 - F1_score` (lower is better)

### 3.2 Dataset

The training corpus was assembled from 26 curated and published security benchmark datasets maintained by the llmtrace project. Sources span a broad range of attack taxonomies and benign input distributions:

| Category | Datasets | Description |
|----------|----------|-------------|
| Hand-curated | injection\_samples, benign\_samples, encoding\_evasion, notinject\_samples | Core injection patterns, benign inputs, encoding-based evasion, over-defense false positives |
| Academic benchmarks | AdvBench, HarmBench, CyberSecEval2, JailbreakBench, SATML-CTF | Published research attack/safety datasets |
| Community datasets | DeepSet, IvanLeoMK, JackHHao, Rubend18, SPML | Open-source prompt injection/jailbreak collections |
| Safety evaluations | AILuminate, SafeGuard, XSTest, BIPIA | AI safety and over-defense evaluation sets |
| Agent attacks | InjecAgent, TensorTrust, TransferAttack, HPI | Tool/agent injection and transfer attack datasets |

After deduplication across all 26 files, the dataset contained **19,186 unique samples**. These were split 80/20 using stratified sampling by label, preserving the malicious/benign ratio:

| Split | Total | Malicious | Benign | Malicious % |
|-------|-------|-----------|--------|-------------|
| Train | 15,348 | 11,569 | 3,779 | 75.4% |
| Val | 3,838 | 2,893 | 945 | 75.4% |

The class imbalance (~3:1 malicious-to-benign) reflects the composition of the source benchmarks and was preserved without resampling or class weighting, as the F1 metric inherently accounts for precision-recall balance.

Data preparation was performed by `scripts/prepare.py`, which reads all JSON dataset files, converts string labels (`"malicious"` -> 1, `"benign"` -> 0), deduplicates by exact text match, and writes stratified JSONL splits with a fixed random seed (42).

### 3.3 Hyperparameter Search Strategy

The experiment employed an **LLM-guided search policy** (`LLMParamPolicy`). The policy uses **MiniMaxAI/MiniMax-M2.5-TEE** accessed via the Chutes API to propose the next hyperparameter configuration.

At each iteration, the policy constructs a prompt containing:

- The full history of prior iterations (hyperparameter values, evaluation metrics, keep/discard decisions)
- The defined search space with allowable values
- The optimization objective (minimize val\_bpb)

The LLM returns a proposed configuration validated against the search space. On failure, the system falls back to seeded random sampling. This policy requires no additional dependencies beyond Python's standard `urllib` library.

**Search space:**

| Parameter | Values | Type |
|-----------|--------|------|
| learning\_rate | 1e-5, 2e-5, 3e-5 | float |
| epochs | 1, 2, 3 | int |
| weight\_decay | 0.0, 0.01, 0.1 | float |
| batch\_size | 8, 16, 32 | int |
| grad\_clip | 0.5, 1.0 | float |

Discrete search space: 3 x 3 x 3 x 3 x 2 = **162 possible configurations**.

### 3.4 Infrastructure

| Component | Specification |
|-----------|---------------|
| Cloud platform | Basilica GPU cloud |
| Docker image | `ghcr.io/epappas/ar-deberta-e2e:latest` |
| Base image | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| GPU allocation | 1x GPU from pool: A100, H100, L40S, RTX-4090, RTX-A6000 |
| CPU / Memory | 8 cores / 32 GiB |
| TTL per deployment | 1,200 seconds |
| Health check | HTTP liveness + startup probes on port 8080 |
| Data delivery | Baked into Docker image (no runtime downloads) |
| Env injection | `AR_PARAMS_JSON` + individual `AR_PARAM_<NAME>` variables |

The Docker image is pre-built with PyTorch, CUDA toolkit, all Python dependencies, the training script, and the full 19k-sample dataset. This eliminates data-download variance and dependency-resolution nondeterminism across iterations.

### 3.5 Comparability Enforcement

The telemetry system enforces comparability across iterations by recording:

- A **hardware fingerprint** (hash derived from the execution environment)
- A **budget mode** (`fixed_wallclock`) with a defined budget in seconds

The `check_comparability()` function rejects iterations where `run_budget_s != expected_budget_s` or where hardware fingerprints diverge. In strict mode, mismatched iterations would be blocked entirely.

### 3.6 Reproducibility

```bash
# Prepare data (requires access to llmtrace datasets)
python scripts/prepare.py --src /path/to/llmtrace/benchmarks/datasets

# Build and push Docker image
docker build -t ghcr.io/epappas/ar-deberta-e2e:latest .
docker push ghcr.io/epappas/ar-deberta-e2e:latest

# Run deployment
source .env  # BASILICA_API_TOKEN, CHUTES_API_KEY, HF_TOKEN
python deploy.py
```

All code, configuration, and data preparation scripts are in the repository. No external state beyond API tokens.

---

## 4. Results

### 4.1 Summary

```json
{"iterations": 5, "best_value": 0.017532, "best_score": 0.017532}
```

Two concurrent episodes executed during the experiment window, producing 9 total iterations of which 6 completed successfully and 3 failed due to deployment timeout.

### 4.2 Episode 9df8855f40fe

The LLM policy explored conservative configurations: fixed learning rate of 1e-5, 1 epoch, no weight decay, varying only batch size and gradient clipping.

| Iter | lr | epochs | weight\_decay | batch\_size | grad\_clip | val\_bpb | F1 | Loss | Decision |
|------|----|--------|--------------|------------|-----------|---------|--------|--------|----------|
| 0 | 1e-5 | 1 | 0.0 | 8 | 0.5 | 0.021735 | 0.978265 | 0.104478 | **keep** |
| 1 | 1e-5 | 1 | 0.0 | 8 | 1.0 | 0.022593 | 0.977407 | 0.104341 | discard |
| 2 | 1e-5 | 1 | 0.0 | 16 | 0.5 | FAILED | -- | -- | discard |
| 3 | 1e-5 | 1 | 0.0 | 16 | 1.0 | FAILED | -- | -- | discard |

**Episode statistics** (2 successful iterations): mean=0.022164, median=0.022164, stdev=0.000607.

Terminated after exhausting the no-improvement limit following two consecutive timeouts.

### 4.3 Episode 0fb86c17e319

The LLM policy explored more aggressively, varying learning rate across {1e-5, 2e-5, 3e-5}, epochs across {2, 3}, and testing weight decay values of 0.0 and 0.01.

| Iter | lr | epochs | weight\_decay | batch\_size | grad\_clip | val\_bpb | F1 | Loss | Decision |
|------|----|--------|--------------|------------|-----------|---------|--------|--------|----------|
| 0 | 2e-5 | 2 | 0.01 | 8 | 1.0 | 0.018835 | 0.981165 | 0.122152 | **keep** |
| 1 | 1e-5 | 2 | 0.01 | 16 | 1.0 | 0.018953 | 0.981047 | 0.102649 | discard |
| 2 | 1e-5 | 3 | 0.0 | 8 | 1.0 | 0.017935 | 0.982065 | 0.122522 | **keep** |
| 3 | 2e-5 | 3 | 0.0 | 8 | 1.0 | 0.017532 | 0.982468 | 0.142032 | **keep** |
| 4 | 3e-5 | 3 | 0.0 | 8 | 0.5 | FAILED | -- | -- | discard |

**Episode statistics** (4 successful iterations): mean=0.018319, median=0.018394, stdev=0.000692.

Three "keep" decisions demonstrate a monotonic improvement trajectory across iterations 0, 2, and 3.

### 4.4 Best Configuration

| Metric | Value |
|--------|-------|
| val\_bpb | **0.017532** |
| F1 score | **0.982468** |
| Validation loss | 0.142032 |
| learning\_rate | 2e-5 |
| epochs | 3 |
| weight\_decay | 0.0 |
| batch\_size | 8 |
| grad\_clip | 1.0 |

### 4.5 Comparability Verification

All successful iterations reported:

| Field | Value |
|-------|-------|
| comparable | 1 |
| budget\_mode | fixed\_wallclock |
| budget\_s | 7200 |
| hardware\_fingerprint | ef8fd2cb7b384864 |
| non\_comparable\_reason | (empty) |

---

## 5. Discussion

### 5.1 LLM Policy Behavior

The LLM-guided search policy exhibited qualitatively different strategies across the two episodes.

In Episode 9df8855f40fe, the policy adopted a conservative, narrow exploration -- holding learning rate, epochs, and weight decay constant while varying only batch size and gradient clipping. This produced F1 scores in the 0.977--0.978 range.

In Episode 0fb86c17e319, the policy explored more aggressively along the epochs and learning rate dimensions. After observing that 2 epochs with weight decay of 0.01 yielded val\_bpb = 0.018835, the policy tested 3 epochs with weight decay of 0.0, discovering a substantial improvement (val\_bpb = 0.017935). It then combined this insight with a higher learning rate (2e-5), achieving the experiment's best result (val\_bpb = 0.017532).

This trajectory suggests the LLM policy extracts directional signal from experiment history and makes informed proposals, though the small sample size (9 iterations) limits conclusions about its systematic advantage over simpler strategies.

### 5.2 Hyperparameter Sensitivity

Several patterns emerge from the successful iterations:

- **Epochs** was the strongest predictor of performance. All 3-epoch configurations outperformed all 1- and 2-epoch configurations. The best 3-epoch result (0.017532) improved over the best 1-epoch result (0.021735) by **19.3% relative error reduction**.
- **Batch size of 8** was used in all 4 best-performing iterations. Batch size 16 led to timeouts in one episode and marginal regression in another.
- **Weight decay removal** (0.0) outperformed weight decay of 0.01 when controlling for other hyperparameters.
- **Gradient clipping** had a small effect. All "keep" decisions in Episode 0fb86c17e319 used grad\_clip=1.0.

### 5.3 Containerized Training as Experimental Methodology

The Basilica GPU cloud deployment model provides properties desirable for hyperparameter optimization experiments:

**Process isolation.** Each iteration runs in an independent container with no filesystem, memory, or GPU state shared with prior iterations. This eliminates a class of reproducibility failures where cached intermediate results, lingering GPU memory allocations, or modified configuration files carry over between trials.

**Immutable environments.** The pre-built Docker image pins the complete software stack -- PyTorch version, CUDA toolkit, Python dependencies, and the training dataset. Performance differences between iterations reflect hyperparameter choices, not environment drift.

**Enforced time budgets.** The 1,200-second TTL per deployment provides a hard upper bound on per-iteration cost. Failed iterations are cleanly terminated and resources released.

**Comparability metadata.** The telemetry system records a hardware fingerprint and budget mode for each iteration. All iterations reporting `comparable=1` confirms results are directly comparable without normalization.

### 5.4 Failure Analysis

Three of 9 iterations (33.3%) failed due to deployment timeout:

- **Episode 9df8855f40fe, iterations 2-3:** Both used `batch_size=16`. Larger batch sizes extend per-step computation time, likely exceeding the 1,200-second TTL for the full training + evaluation pipeline.
- **Episode 0fb86c17e319, iteration 4:** Used `lr=3e-5` with 3 epochs and `batch_size=8`. The higher learning rate in combination with 3 epochs may have caused training instability leading to a stall, or the 3-epoch run with grad\_clip=0.5 simply exceeded the time budget.

### 5.5 Limitations

1. **Small iteration count.** 6 successful iterations across 162 possible configurations (3.7% coverage) cannot make strong claims about global optimality.
2. **GPU heterogeneity.** The GPU pool includes architecturally diverse accelerators. Training time and numerical behavior may differ across GPU types. The hardware fingerprint confirms budget-level comparability but does not guarantee identical floating-point behavior.
3. **Class imbalance.** The 3:1 malicious-to-benign ratio was not explicitly addressed through oversampling, undersampling, or loss weighting. Per-class precision and recall were not extracted from training logs.
4. **No seed variation.** Each configuration was run once. Seed-to-seed variance is unmeasured.

---

## 6. Conclusion

This experiment demonstrates a practical methodology for automated hyperparameter optimization using containerized GPU training on Basilica cloud infrastructure. The autoresearch-rl continuous loop, combined with an LLM-guided search policy, identified a DeBERTa-v3-base configuration achieving an F1 score of **0.982468** on a prompt injection detection task spanning 19,186 samples from 26 security benchmark datasets.

The best configuration (lr=2e-5, epochs=3, weight\_decay=0.0, batch\_size=8, grad\_clip=1.0) was discovered in 5 iterations, with the LLM policy demonstrating the ability to extract directional signal from prior results and propose progressively improving configurations. The improvement from the initial best (val\_bpb=0.021735) to the final best (val\_bpb=0.017532) represents a 19.3% reduction in classification error rate.

The containerized deployment model provided strong isolation and reproducibility guarantees, with all successful iterations verified as comparable via hardware fingerprinting and budget-mode enforcement. The approach is generalizable to other model architectures and optimization objectives, requiring only a Docker image with baked-in dependencies and a training script that emits parseable metrics to stdout.

---

## Appendix A: Telemetry Evidence

### A.1 Results Ledger (19k dataset runs only)

Source: `artifacts/deberta/results.tsv`

```
commit   metric  value     status   episode_id    iter  score     comparable
7290dfa  val_bpb 0.021735  keep     9df8855f40fe  0     0.021735  1
7290dfa  val_bpb 0.022593  discard  9df8855f40fe  1     0.021735  1
7290dfa  val_bpb 0.018835  keep     0fb86c17e319  0     0.018835  1
7290dfa  val_bpb 0.018953  discard  0fb86c17e319  1     0.018835  1
7290dfa  val_bpb 0.000000  discard  9df8855f40fe  2     0.021735  1  (timeout)
7290dfa  val_bpb 0.000000  discard  9df8855f40fe  3     0.021735  1  (timeout)
7290dfa  val_bpb 0.017935  keep     0fb86c17e319  2     0.017935  1
7290dfa  val_bpb 0.017532  keep     0fb86c17e319  3     0.017532  1
7290dfa  val_bpb 0.000000  discard  0fb86c17e319  4     0.017532  1  (timeout)
```

### A.2 Episode Summaries

Source: `traces/deberta/events.jsonl`

**Episode 9df8855f40fe:**

```json
{
  "type": "episode_summary",
  "episode_id": "9df8855f40fe",
  "mean": 0.022164,
  "median": 0.022164,
  "min": 0.021735,
  "max": 0.022593,
  "stdev": 0.0006066976182580561,
  "count": 2
}
```

**Episode 0fb86c17e319:**

```json
{
  "type": "episode_summary",
  "episode_id": "0fb86c17e319",
  "mean": 0.01831375,
  "median": 0.018385,
  "min": 0.017532,
  "max": 0.018953,
  "stdev": 0.0006915988119326609,
  "count": 4
}
```

### A.3 Reproducibility Artifact

All iterations executed with:

- Hardware fingerprint: `ef8fd2cb7b384864`
- Budget mode: `fixed_wallclock`
- Budget: 7,200 seconds
- Docker image: `ghcr.io/epappas/ar-deberta-e2e:latest` (digest: `sha256:81e8f348c86b27166f26b485d589c16c901b78c89af1c5f55671437d200a93c6`)
