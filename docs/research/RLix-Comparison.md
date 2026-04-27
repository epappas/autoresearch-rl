# AutoResearch-RL vs RLix — Comparative Analysis

**Compared project**: [github.com/rlops/rlix](https://github.com/rlops/rlix) (Apache-2.0, Python, ~270 stars, 206 commits, single primary author + Devin bot, last push 2026-04-24).

## TL;DR

These projects are **orthogonal, not competitive**. AutoResearch-RL is an **algorithm-layer search primitive**: an autonomous LLM-driven loop that proposes hyperparameters or code diffs, runs one experiment at a time, and keeps what improves. RLix is an **infrastructure-layer sharing primitive**: a Ray-based scheduler that lets multiple ROLL training jobs share a fixed GPU pool by treating rollout as elastic, preemptible work. They could compose; they do not overlap.

---

## What each project IS

**AutoResearch-RL** — A controller loop where a policy (LLM, random, grid, hybrid, or learned PPO) proposes the *next experiment* — either a hyperparameter dict or a unified diff against `train.py`. Each iteration is a one-shot training run on local CPU/GPU, an HTTP target, or a Basilica GPU container. Outcomes feed a keep/discard ledger; checkpoints survive crashes. The frozen `prepare.py` / mutable `train.py` split is a **trust contract** preventing the LLM from gaming evaluation.

**RLix** — A Ray control plane (`Orchestrator` + `SchedulerImpl` + `ResourceManager` + per-pipeline `PipelineCoordinator`) that wraps Alibaba **ROLL** pipelines. A 7-tier priority enum (`INITIALIZATION=0` … `GENERATION=6`) makes rollout the only preemptible tier; a "gap-ratio" planner divides spare GPUs across pipelines proportional to remaining trajectory demand. Cooperative shrink/expand is implemented through `resize_infer` RPCs to vLLM workers with `sleep_level=2` enforced.

---

## What each does best

| | AutoResearch-RL | RLix |
|---|---|---|
| **Best at** | Autonomous, long-horizon experiment search; algorithmic exploration via LLM-generated code diffs; reproducible per-iteration ledger | Cross-job rollout-time GPU sharing; cooperative preemption of vLLM workers; making N concurrent RL jobs fit on M < N×full GPUs |
| **Iteration unit** | A trial (params or one diff) — minutes to hours | A full multi-step RL training job — hours to days |
| **What it optimizes** | Experiments per researcher-hour | GPU-hours per experiment |
| **Audience** | Researchers running expensive serial experiments | Teams running multiple concurrent RL pipelines on a shared cluster |

---

## Novelty

### AutoResearch-RL's novelty

1. **LLM as policy proposing unified diffs** with multi-turn correction (`LLMDiffPolicy` → `sandbox/validator.py` → retry on `git apply --check` failure). Stdlib-only HTTP, falls back to seeded random on API failure.
2. **Frozen/mutable evaluation contract** — `prepare.py` (data + reward + extraction) is immutable, `train.py` is mutable. Prevents the LLM from gaming the metric.
3. **Hybrid stall escalation** — start with param search, switch to code diffs after `stall_threshold` non-improving iterations, fall back to params on consecutive diff failures.
4. **PPO-over-experiment-history learned policy** with optional sDPO KL regularization against a teacher snapshot (`policy/learned_search.py`).
5. **Power-law forecasting early-stop** for doomed campaigns.
6. **Hardware fingerprinting + budget-mode comparability** baked into the ledger.

### RLix's novelty

1. **Cross-pipeline rollout sharing under a 7-tier priority lattice** — the gap-ratio planner (`scheduler/planner.py::plan_generation_gap_ratio`) is the actual unique contribution. veRL / OpenRLHF / TRL are single-job; RLix is multi-tenant.
2. **Cooperative vLLM preemption via `sleep_level=2` + `offload_nccl`** — workers actually drop weights and NCCL buffers from VRAM on shrink, so reclaimed capacity is real, not nominal.
3. **In-place weight sync to newly-expanded infer workers** via CUDA IPC or NCCL broadcast (`pipeline/model_update_service.py`) — required so elastically added rollout workers serve the freshest policy.
4. **Multi-LoRA pipeline** — multiple adapters sharing a base model in VRAM, separate optimizers, dirty-set training (`pipeline/multi_lora_pipeline.py`).
5. **Perfetto-format scheduling timeline** (`scheduler/tracer.py`).

---

## Use cases each covers

### AutoResearch-RL

- Autonomous hyperparameter sweeps with intelligent proposal (LLM-guided > random > grid).
- Algorithmic improvement via code mutation: GRPO reward shaping, partial-credit rewards, sampling strategy changes (`examples/basilica-grpo` — Qwen2.5-0.5B on GSM8K, 8h).
- Classifier post-training with diff-based exploration (`examples/deberta-prompt-injection`).
- Tiny on-CPU "smoke" experiments (`examples/minimal-trainable-target`).
- Long unattended cloud campaigns with checkpoint/resume and Basilica deploy/cleanup.
- Best-model promotion + push to HF Hub (`autoresearch-rl upload`).

### RLix

- Running 2–N concurrent ROLL/GRPO pipelines on a fixed cluster where each would otherwise idle GPUs during rollout.
- Concurrent multi-LoRA training of several agents on one base model.
- Empirically: only **GRPO + Sokoban + Qwen2.5-0.5B-Instruct, `max_steps: 3`** in shipped examples — clearly a research prototype.
- Open PRs are working on a NeMo-RL adapter (not yet merged).

---

## Common ground

- Python ≥ 3.10, AI-assisted projects, NVIDIA GPU targets.
- YAML-driven configuration.
- Both ultimately exist to reduce wall-clock-per-result for RL training; they attack different ends of the bottleneck.
- Neither does formal verification of code/diffs; both rely on cooperative protocols (a validator in one, `sleep_level=2` discipline in the other).

## Non-common ground

| Concern | AutoResearch-RL | RLix |
|---|---|---|
| **Concurrency model** | Strictly **serial** iterations, one trial at a time | Many pipelines concurrent, sharing GPUs cooperatively |
| **Distributed training** | None in core — experiment scripts BYO | Megatron + vLLM via ROLL submodule (`rlops/ROLL` branch `rlix`) |
| **GPU primitive** | Treats GPU as opaque (one container/iter on Basilica) | Allocates DP ranks within TP-aligned bundles, plans shrink/expand |
| **Inference engine awareness** | None | vLLM-specific (sleep level, NCCL offload validation) |
| **Search/exploration logic** | Core feature (5 policy types) | None — pipelines are predetermined |
| **Evaluation integrity** | Frozen `prepare.py` contract | N/A — RLix does not run evaluation |
| **State persistence** | Checkpoint, resume, ledger, manifests, traces | Fail-fast only — restart re-registers everything |
| **Hardware diversity** | Local + HTTP + Basilica cloud | CUDA only, homogeneous per-node, contiguous global GPU IDs |
| **Algorithm coupling** | Algorithm-agnostic (any train script that prints metrics) | GRPO / REINFORCE++ / GAE via ROLL only |

---

## Pointers to inherit from RLix into AutoResearch-RL

Ranked by fit. None require abandoning the autonomous-loop thesis.

### Strong fit

1. **Concurrent iterations on shared GPUs.** Today `controller/continuous.py` is strictly serial. Even a modest "fan out N candidate iterations, take the best" mode (Bayesian-optimization style batched proposals) would be a real gain for `LLMParamPolicy` and `RandomPolicy`. RLix's planner shows how to think about contiguous TP groups and per-DP-rank allocation if/when a single iteration ever uses multi-GPU.

2. **Cooperative preemption of in-flight iterations** when the policy receives evidence the trial is doomed. Right now power-law forecasting runs *after* iterations complete (post-hoc early stop on the next iteration). A `resize_infer`-style cooperative cancel — train scripts checking a "should I stop" signal at each gradient step — would let `power_law` actually save GPU-hours on the *current* iteration.

3. **A scheduling timeline trace** in Perfetto / Chrome-trace format. RLix's `SchedulerTracer` produces one-line-per-cluster timelines that visualize GPU utilization. An analogous `traces/timeline.json` for autoresearch-rl (when each iteration started/ended on each Basilica deployment) would diagnose campaign latency far better than the JSONL events alone.

4. **An explicit `ProgressReport`-style protocol** between target and controller. Today the controller polls Basilica logs every 20s and parses metrics with regex. A small structured progress endpoint (alongside the existing `/model/files`) — `step_completed`, `step_target`, `latest_score` — would let the controller cut iterations early and feed *intra-iteration* signal to the policy.

### Medium fit

5. **Multi-LoRA "iteration sharing"** — when the search space is e.g. "5 different reward functions" and the underlying base model is shared, train all 5 LoRAs in one container, evaluate each, return 5 outcomes per "campaign step". This is exactly RLix's multi-LoRA pattern repurposed as a search efficiency primitive. Would slot in as a new target type, e.g. `multi_lora_basilica`.

6. **Operational fail-fast policy** for clearly malformed configs. RLix's two-phase validate-then-mutate under lock is overkill for a single-process loop, but the *ethos* — refuse to start rather than silently degrade — could be cleaner in `config.py` validation, especially around `objective.metric` not appearing in any iteration's metrics.

### Weak fit (mention, don't adopt)

7. **The 7-tier priority enum** — only meaningful when one process owns multiple distinct compute phases. A serial iteration loop has nothing to prioritize between.
8. **Ray-actor architecture** — adds operational weight that does not pay off until you have concurrent pipelines + cross-pipeline state. Premature for the current scope.
9. **Tight ROLL / Megatron / vLLM coupling** — antithetical to autoresearch-rl's "any train script that prints metrics" contract. Do not take this.

---

## Bottom line

RLix is a sharp, single-purpose tool: *"share rollout GPUs across concurrent ROLL pipelines."* AutoResearch-RL is a sharp, different-purpose tool: *"let an LLM autonomously search hyperparameters AND code for a single experiment family."* The two strongest things to borrow are **concurrent / batched iterations with cooperative cancellation** and **structured intra-iteration progress reporting** — both of which would compress the wall-clock of a search campaign without changing the autonomous-loop thesis. Everything else in RLix presupposes a multi-tenant Ray cluster running ROLL, which is a different product.
