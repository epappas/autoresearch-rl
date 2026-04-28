# Phase 5 — Multi-LoRA Iteration-Sharing Target (Deferred)

**Status**: deferred 2026-04-28. **Trigger to re-open**: see §5 below.

Companion to [`RLix-Adoption-Plan.md`](./RLix-Adoption-Plan.md) and [`RLix-Adoption-Remediation.md`](./RLix-Adoption-Remediation.md). The other six phases (R0, 6, 1, 2, 3, 4 plus paired Phase 7 work) shipped between 2026-04-27 and 2026-04-28; this one was marked medium-fit / *defer until needed* in the original plan and remains unimplemented by design.

This document captures (1) why we deferred, (2) what implementing it would entail, (3) what triggers a re-open, and (4) an implementation sketch so future-us can pick it up without redoing the analysis.

---

## 1. What it would do

One Basilica deployment trains **N LoRA adapters concurrently** over a shared base model in VRAM, with separate optimizers per adapter. Each adapter gets its own per-tag reward function or hyperparameter overlay. The container returns N per-tag results; the engine unpacks them into N consecutive ledger rows under one `episode_id`.

Conceptually: a single search-iteration becomes N candidates evaluated in parallel inside one container, sharing the expensive base-model load and VRAM footprint. RLix's `RollMultiLoraPipeline` is the inspiration (`pipeline/multi_lora_pipeline.py` in `rlops/rlix`).

## 2. Why it pays off (when it does)

- Base model is fixed across the N candidates being compared.
- The only varying axis is something a LoRA can capture: per-adapter reward shaping, prompt template, instruction style, small per-adapter scalar hyperparameters.
- Inference / training memory budget would not fit N independent deployments: e.g., Qwen2.5-7B base model is ~14 GB in fp16, but adding N LoRAs of rank 16 is ~hundreds of MB per adapter — N=4 fits one A100 trivially.

In that regime, Phase 5 ≈ N× more candidates per Basilica-hour than the current Phase 4 path of "N parallel deployments each loading the base model".

## 3. Why it does NOT pay off (most of our actual workload)

- Hyperparameter search where candidates differ in what LoRAs cannot isolate:
  - Base learning rate
  - Optimizer choice (AdamW vs Lion vs Muon)
  - Model architecture / layer counts
  - Tokenizer / dataset slicing
  - Batch size that shifts the VRAM footprint
- Code-diff-based search (`llm_diff` / `hybrid` policies). Each diff is a different `train.py`; LoRAs cannot factor that out.
- CPU-only or small-model examples (`minimal-trainable-target`, `deberta-prompt-injection`'s local mode). No memory pressure, so sharing a container saves nothing.
- Single-LoRA training (one adapter, fixed reward). Already handled by the existing `BasilicaTarget` + Phase 4 parallel engine.

The shipped examples (`examples/minimal-trainable-target`, `examples/basilica-grpo`, `examples/deberta-prompt-injection`, `examples/autoresearch-like`) all fall into the "doesn't pay off" bucket. There is no in-tree campaign that would benefit from Phase 5 today.

## 4. Cost

Estimated **M** (1–3 days actual). Concretely:

- New target type (`target/multi_lora_basilica.py`) — ~300 lines following the existing `BasilicaTarget` shape, with bootstrap modifications for parallel adapter spawning.
- Engine schema extension. Currently each iteration produces one `Outcome`. Phase 5 needs the engine to accept a `MultiOutcome` and unpack into N consecutive ledger rows, history entries, and version directories. Touches `controller/engine.py`, `controller/parallel_engine.py`, `controller/executor.py`, `controller/types.py`.
- New policy wrapper (`policy/lora_batch.py::LoraBatchPolicy`) — takes a base param policy, returns N proposals tagged with `_lora_tag = "lora_0".."lora_{n-1}"`. The trial reads the tag to select per-LoRA behavior.
- New example (`examples/multi-lora-grpo`) demonstrating the actual win on a Qwen2.5-0.5B base + 4 reward variants.
- Tests: ~15-20 new tests covering MultiOutcome unpacking, ledger row attribution, version directory layout, LoraBatchPolicy proposal generation.
- Docs: extend `RLix-Adoption-Plan.md` Phase 5, update example README.

## 5. Triggers for re-opening

Re-open when **any one** of these is true:

1. **Real campaign needs it.** A user wants to compare ≥3 reward functions / instruction templates / per-adapter scalars *on the same base model* in a single experiment. The break-even vs Phase 4 parallel deployments is roughly when the base model is large enough that N independent loads exceed cluster GPU-hours budget.
2. **Cluster GPU pool is constrained.** Phase 4 parallel mode runs N trials on N GPUs (or one GPU per `resource_cost`). If the cluster has e.g. 4 A100s and the user wants to compare 8 reward shapings of a 7B model, Phase 5 fits 8 LoRAs on 4 GPUs (2 per GPU) where Phase 4 would queue.
3. **A new RL recipe lands** that benefits from per-adapter critic isolation (e.g., per-tag advantage normalization), where running it as N separate containers loses the cross-adapter feedback loop.

Skip-the-trigger conditions: if the user only ever runs single-model search or code-diff search, Phase 5 will never pay off and should stay deferred.

## 6. Implementation sketch (for the future implementer)

Hand this to whoever picks it up. The order assumes the existing codebase shape as of commit `d6577ae` and reuses everything Phase 1–4 built.

### 6.1 New `Outcome` variant — `MultiOutcome`

```python
# controller/executor.py
@dataclass
class MultiOutcome:
    """N per-tag results from a single executor invocation.
    Each tag becomes its own ledger row + history entry.
    """
    status: str
    per_tag: dict[str, Outcome]   # tag -> single-iter Outcome
    elapsed_s: float
    run_dir: str
```

Engine accepts `Outcome | MultiOutcome` from `executor.execute`. When `MultiOutcome`:
- Unpack into N pseudo-iterations.
- Each pseudo-iter gets its own `iter_idx` in the contiguous block, same `episode_id`, distinct `params["_lora_tag"]`.
- Best-tracking, keep/discard, version-saving, ledger-row, history-append all run per tag.
- `Learnable.record_reward` gets one reward per tag, drained in submission order (R3.a applies).

### 6.2 New `MultiLoraBasilicaTarget`

Path: `src/autoresearch_rl/target/multi_lora_basilica.py`. Shape mirrors `BasilicaTarget` but the bootstrap script:

- Reads `AR_LORA_TAGS` env (comma-separated list of tag names) injected by the controller.
- For each tag, spawns a parallel adapter training (within Python: `multiprocessing` or thread-per-adapter coordinated via vLLM-style `sleep_level` if using vLLM inference; for plain SFT/GRPO via TRL, just iterate adapters in inner loop).
- Writes per-tag metrics to `progress.jsonl` with a `tag` field.
- The engine-side `_propagate_control` already handles cancel; under multi-LoRA, cancel applies to the entire deployment (all tags) — that's a feature, not a limitation, since they share the base model.

Resource cost declared via `def resource_cost(self, params)`:
```python
{"gpu": self._bcfg.gpu_count}  # one deployment, N LoRAs
```

### 6.3 New `LoraBatchPolicy`

Path: `src/autoresearch_rl/policy/lora_batch.py`. Wraps a base `Policy`:

```python
class LoraBatchPolicy:
    def __init__(self, base: Policy, *, n_loras: int, tag_axis: dict[str, list]):
        self._base = base
        self._n_loras = n_loras
        # tag_axis: per-tag overrides, e.g.
        #   {"reward_kind": ["partial_credit", "exact_match", "graded"]}
        self._tag_axis = tag_axis

    def propose(self, state: dict) -> ParamProposal:
        base_params = self._base.propose(state).params
        per_tag = []
        for i in range(self._n_loras):
            tag_params = {**base_params, "_lora_tag": f"lora_{i}"}
            for k, vs in self._tag_axis.items():
                tag_params[k] = vs[i % len(vs)]
            per_tag.append(tag_params)
        return ParamProposal(params={"_lora_batch": per_tag})

    def propose_batch(self, state: dict, k: int) -> list[ParamProposal]:
        return [self.propose(state) for _ in range(max(0, k))]
```

The target then unpacks `params["_lora_batch"]` into per-adapter env injection.

### 6.4 Engine wiring

`controller/engine.py::run_experiment`:
```python
outcome = executor.execute(proposal, run_dir)
if isinstance(outcome, MultiOutcome):
    for tag, single in outcome.per_tag.items():
        # Process as if it were its own iteration:
        # - increment iter_idx
        # - run the existing keep/discard, ledger, history, telemetry path
        # - tag the params with _lora_tag for ledger attribution
    iter_idx += len(outcome.per_tag)
else:
    # existing single-iter path
```

Same surgery in `controller/parallel_engine.py`. Both engines share the unpacking helper.

### 6.5 Example

`examples/multi-lora-grpo/`:
- `config.yaml` with `target.type: multi_lora_basilica`, `policy.type: lora_batch`, base policy `random` over `learning_rate`.
- `train.py` reads `AR_LORA_TAGS` and trains all adapters in one call.
- `program.md` documents the protocol and the per-tag reward variants.

### 6.6 Tests (~15–20)

- MultiOutcome unpacks into N ledger rows with correct `_lora_tag` attribution.
- Best-tracking picks the best tag across the N candidates.
- LoraBatchPolicy generates N tagged proposals with stable ordering.
- Cancel signal cancels the whole deployment (all tags marked cancelled).
- Resource pool admits the deployment as one unit.
- End-to-end synthetic test with a fake `MultiLoraBasilicaTarget` returning canned per-tag metrics.

### 6.7 Risks to watch

1. **Cancellation granularity**. Phase 2's cooperative cancel writes one control file per `run_dir`. Multi-LoRA shares one `run_dir`, so cancel applies to all tags. This is the correct semantic (they share the base model), but the engine must record `decision="cancelled"` for *all* tags in the batch, not just the one whose forecast triggered.
2. **Ledger schema attribution**. The current ledger has no column for `_lora_tag`. Encode in the `description` column the same way Phase 4 encodes `conc=K`: `{label}|lora_tag=lora_3`. Avoids schema migration.
3. **R3.a reward ordering** with batch unpacking. Rewards must still arrive in monotonic `iter_idx` order. The engine's existing `pending_rewards` drain handles this naturally if the unpacking assigns contiguous iter_idxs to the batch.
4. **Forecasting / IntraIterationGuard** semantics. The guard is per-trial (per-deployment in this case). It would need a "best across tags" view to fairly cancel a deployment whose worst-tag is doomed but best-tag is promising. Simplest first version: don't activate the guard in multi-lora mode; revisit if it becomes a bottleneck.

## 7. What this document is NOT

- Not a commitment. Phase 5 stays deferred unless §5 triggers fire.
- Not a substitute for re-validating the design when re-opening. The codebase will have moved by then.
- Not an exhaustive design review. The sketch in §6 is precise enough to start; details (per-tag versioning layout, tag naming convention, multi-tag forecaster) need fleshing during implementation.

## Appendix: pointers

- Original plan: [`RLix-Adoption-Plan.md`](./RLix-Adoption-Plan.md) §5
- RLix reference: [`pipeline/multi_lora_pipeline.py`](https://github.com/rlops/rlix/blob/main/rlix/pipeline/multi_lora_pipeline.py) in the upstream repo
- Existing Basilica target shape to mirror: `src/autoresearch_rl/target/basilica.py`
- Existing parallel-engine reward ordering pattern (R3.a): `src/autoresearch_rl/controller/parallel_engine.py::run_experiment_parallel` (`pending_rewards` + `_drain_rewards`)
