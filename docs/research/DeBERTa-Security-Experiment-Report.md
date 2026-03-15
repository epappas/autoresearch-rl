# Cloud-Native Hyperparameter Search for Prompt Injection Detection

**autoresearch-rl Pilot Experiment on Basilica GPU Cloud**

**Date:** 2026-03-15

---

## Abstract

We present a pilot experiment using autoresearch-rl to perform automated hyperparameter search for a DeBERTa-v3-based prompt injection classifier. The system orchestrated GPU-accelerated training runs on Basilica cloud infrastructure through a keep/discard optimization loop, requiring no manual intervention beyond a single CLI invocation. Over five iterations exploring a grid of learning rate and epoch combinations, the model achieved perfect classification (F1=1.0, accuracy=1.0, val_bpb=0.0) on a minimal 8-train / 4-validation sample dataset. While the saturated metrics preclude meaningful hyperparameter comparison, the experiment validates the end-to-end pipeline: cloud deployment provisioning, remote training execution, metric extraction, artifact versioning, and automatic resource cleanup. We discuss the limitations imposed by dataset scale and outline requirements for follow-up experiments with realistic data volumes.

---

## 1. Introduction

Prompt injection is a security-critical failure mode in systems that incorporate large language models. Classifiers that detect injected prompts before they reach an LLM serve as a practical defense layer. Fine-tuning such classifiers requires iterating over hyperparameter configurations, and doing so on cloud GPU infrastructure introduces orchestration complexity: provisioning machines, deploying code, collecting results, and tearing down resources.

autoresearch-rl is an optimization loop that treats each training run as an environment step. A policy proposes hyperparameters, the system executes a training run on a provisioned target, evaluates the result, and decides whether to keep or discard the configuration. The loop terminates when a stop condition is met (wall time, no-improvement streak, or failure rate).

This report documents the first end-to-end experiment using the Basilica cloud target adapter. The goals are:

1. Validate that the CLI-to-cloud pipeline executes without manual intervention.
2. Confirm that metrics are correctly extracted and the keep/discard logic operates as specified.
3. Identify limitations that must be addressed before production-scale experiments.

---

## 2. System Architecture

### 2.1 Runtime Path

```
CLI invocation:
  python3 -m autoresearch_rl.cli --config examples/deberta-prompt-injection/basilica.yaml

Code path:
  cli.py -> RunConfig.model_validate(yaml)
         -> build_target(cfg)          [target/registry.py]
         -> BasilicaTarget             [target/basilica.py]
         -> run_continuous()           [controller/continuous.py]
              for each iteration:
                GridPolicy.next()      [policy/search.py]
                BasilicaTarget.run()
                  -> _build_bootstrap_cmd()
                  -> create_deployment() on Basilica API
                  -> _poll_for_metrics() from deployment logs
                BasilicaTarget.eval()  -> return cached train metrics
                keep/discard decision
                emit telemetry         [telemetry/events.py, ledger.py]
                save checkpoint        [checkpoint.py]
              cleanup all deployments
```

### 2.2 Basilica Target Adapter

`BasilicaTarget` (`src/autoresearch_rl/target/basilica.py`) implements the `TargetAdapter` protocol. For each iteration it:

1. Constructs a bootstrap script that starts an HTTP health-check server on port 8080 (required by Basilica to mark the deployment as "ready"), runs an optional `setup_cmd` (dependency installation, file downloads), then executes the training command.
2. Creates a Basilica deployment via `client.create_deployment()` with the Docker image, GPU spec, environment variables (`AR_PARAMS_JSON`), and health check configuration.
3. Polls deployment status until ready, then polls logs for training metrics.
4. Extracts metrics from Basilica's structured SSE JSON log format (`data: {"message": "val_bpb=0.000000", ...}`).
5. Deletes the deployment to free GPU resources.

### 2.3 Stop Guards

The continuous loop implements four stop conditions:
- **Wall time:** `max_wall_time_s` (7200s in this experiment)
- **No-improvement streak:** `no_improve_limit` (4 in this experiment)
- **Failure rate:** `failure_rate_limit` (0.8)
- **Graceful shutdown:** SIGINT/SIGTERM handler

---

## 3. Methodology

### 3.1 Task and Model

- **Model:** `protectai/deberta-v3-base-prompt-injection-v2` (184M parameters)
- **Task:** Binary classification -- detect prompt injection attacks (label=1) vs benign inputs (label=0)
- **Training script:** `examples/deberta-prompt-injection/train.py` (pulled from raw GitHub at deployment time)

### 3.2 Dataset

| Split | Samples | Injections | Benign |
|-------|---------|------------|--------|
| Train | 8 | 4 | 4 |
| Validation | 4 | 2 | 2 |

Data files: `examples/deberta-prompt-injection/data/{train,val}.jsonl`

### 3.3 Search Space

| Parameter | Values | Type |
|-----------|--------|------|
| learning_rate | 1e-5, 2e-5, 3e-5 | float |
| epochs | 1, 2 | int |

Grid size: 6 configurations. Exploration order: sequential (GridPolicy).

### 3.4 Optimization Objective

Minimize `val_bpb = 1 - F1` (lower is better). Perfect score: 0.0.

### 3.5 Infrastructure

| Component | Specification |
|-----------|---------------|
| Cloud | Basilica GPU cloud |
| GPU models | A100, H100, L40S, RTX-4090, RTX-A6000 |
| Container | pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel |
| Dependencies | Installed at runtime via `setup_cmd` |
| Training code | Pulled from raw GitHub at deployment time |
| TTL | 900s per deployment (auto-cleanup safety net) |

### 3.6 Reproducibility

```bash
export BASILICA_API_TOKEN=<your-token>
export HF_TOKEN=<your-hf-token>
PYTHONPATH=src python3 -m autoresearch_rl.cli \
  --config examples/deberta-prompt-injection/basilica.yaml
```

All code, configuration, and data are in the repository. No external state beyond API tokens.

---

## 4. Results

### 4.1 CLI Output

```json
{"iterations": 5, "best_value": 0.0, "best_score": 0.0}
```

### 4.2 Per-Iteration Results

Source: `artifacts/deberta-basilica/results.tsv`

| Iter | learning_rate | epochs | loss | val_bpb | f1 | accuracy | Decision |
|------|--------------|--------|------|---------|------|----------|----------|
| 0 | 1e-5 | 1 | 1e-06 | 0.0 | 1.0 | 1.0 | **keep** |
| 1 | 1e-5 | 2 | 1e-06 | 0.0 | 1.0 | 1.0 | discard |
| 2 | 2e-5 | 1 | 1e-06 | 0.0 | 1.0 | 1.0 | discard |
| 3 | 2e-5 | 2 | 1e-06 | 0.0 | 1.0 | 1.0 | discard |
| 4 | 3e-5 | 1 | 1e-06 | 0.0 | 1.0 | 1.0 | discard |

### 4.3 Versioned Artifact

Source: `artifacts/deberta-basilica/versions/v0000/version.json`

```json
{
    "iter": 0,
    "metrics": {"loss": 1e-06, "val_bpb": 0.0, "f1": 1.0, "accuracy": 1.0},
    "params": {"learning_rate": 1e-05, "epochs": 1},
    "status": "ok"
}
```

### 4.4 Stop Condition

The no-improvement counter reached 4 after iteration 4 (iterations 1-4 each produced val_bpb=0.0, which is not strictly less than the best of 0.0). The 6th configuration (lr=3e-5, epochs=2) was never evaluated.

---

## 5. Discussion

### 5.1 Metric Saturation

All configurations achieved perfect F1=1.0. This is expected: the pre-trained model was specifically designed for prompt injection detection, and 4 validation samples from the same distribution are trivially classifiable. The experiment cannot distinguish between hyperparameter configurations.

### 5.2 What the Experiment Validates

Despite metric saturation, the experiment confirms:

1. **End-to-end cloud execution.** Five GPU deployments created, trained, metrics parsed, cleaned up -- from one CLI command.
2. **Correct keep/discard logic.** Iteration 0 kept as baseline. Iterations 1-4 correctly discarded (equal is not strictly better).
3. **Stop guard activation.** No-improvement limit of 4 correctly terminated the loop after iteration 4.
4. **Artifact integrity.** Version directory `v0000/` contains valid `version.json` with correct data.
5. **Telemetry completeness.** Results ledger (5 rows) and JSONL event trace both contain accurate per-iteration data.
6. **Resource cleanup.** All 5 Basilica deployments deleted after loop completion.

### 5.3 Limitations

1. **Dataset scale.** 12 samples are insufficient for meaningful model evaluation.
2. **No statistical variance.** Each configuration run once. No measure of seed-to-seed variance.
3. **Metric design.** `val_bpb = 1 - F1` saturates at 0.0 for this model+data combination.
4. **Hardware heterogeneity.** Basilica may assign different GPU types across deployments.
5. **Single-objective.** The loop optimizes one scalar metric.

---

## 6. Conclusion

This pilot experiment validates that autoresearch-rl can orchestrate GPU-accelerated hyperparameter search on cloud infrastructure through a single CLI invocation. The system correctly provisioned five Basilica cloud deployments, extracted training metrics from deployment logs, applied keep/discard logic, persisted checkpoints and artifacts, and cleaned up all resources on termination.

The experiment's scientific value is limited by the trivial dataset: all configurations achieved perfect classification, yielding no hyperparameter signal. This is a limitation of the demonstration setup, not the framework.

### Recommended Next Steps

1. Replace the 12-sample dataset with a standard prompt injection benchmark of sufficient size and difficulty.
2. Add random seed variation (3-5 repetitions) to measure metric variance.
3. Expand the search space to include batch size, weight decay, and warmup ratio.
4. Introduce secondary metrics (inference latency, calibration error) to provide signal when F1 saturates.
