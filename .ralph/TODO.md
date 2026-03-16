# AutoResearch-RL: Research-Aligned Implementation TODO

This document captures the gap between the current codebase and the research corpus
(AutoResearch-RL, SDPO, SDFT papers). Tasks are organized by phase, with dependencies noted.

---

## Phase A: Structural Cleanup (no new features, pure refactor)

### A.1 Unify loop controllers
**Problem:** Two independent loop systems (`controller/continuous.py` and `controller/loop.py`)
with duplicated logic: separate `LoopResult` dataclasses, separate `_current_commit` helpers,
separate stop-guard implementations, separate telemetry wiring.

**Resolution:** Merge into a single loop controller that supports both modes (bounded iteration
count and continuous wall-clock) via configuration. Extract shared helpers (commit lookup,
stop-guard evaluation, telemetry emission) into reusable functions.

**Files:** `controller/continuous.py`, `controller/loop.py`, `cli.py`, `scripts/run_once.py`
**Dependencies:** None. All subsequent tasks build on the unified loop.

### A.2 Fix duplicate seed field in ControllerConfig
**Problem:** `config.py` `ControllerConfig` declares `seed: int | None = None` twice (lines 37, 42).

**Files:** `config.py`
**Dependencies:** None.

### A.3 Unify policy hierarchies
**Problem:** Two unrelated policy abstractions coexist:
- `policy/search.py`: `ParamPolicy` base class with `GridPolicy`, `RandomPolicy`, `StaticPolicy`
  (used by continuous loop for hyperparameter proposals)
- `policy/interface.py`: `ProposalPolicy` protocol with `propose()` / `propose_diff()`
  (used by legacy loop for code-diff proposals)

Both are called "policies" but have incompatible interfaces and serve the same conceptual role
(propose the next action given history).

**Resolution:** Define a single `Policy` protocol with a `propose(state) -> Proposal` method.
Param-based and diff-based proposals are both subtypes of `Proposal`. Remove `policy/interface.py`
and fold its protocol into the unified base.

**Files:** `policy/search.py`, `policy/interface.py`, `policy/baselines.py`, `policy/learned.py`,
`controller/continuous.py`, `controller/loop.py`
**Dependencies:** A.1

### A.4 Fix version mismatch
**Problem:** `pyproject.toml` says `0.2.0`, `__init__.py` says `0.1.0`.

**Files:** `__init__.py`
**Dependencies:** None.

### A.5 Cache hardware_fingerprint()
**Problem:** `comparability.py` shells out to `nvidia-smi` on every call. This is called per-iteration
in the loop. Hardware does not change during process lifetime.

**Resolution:** Compute once on first call, cache in module-level variable.

**Files:** `telemetry/comparability.py`
**Dependencies:** None.

### A.6 Validate config in run_once.py
**Problem:** `scripts/run_once.py` calls `yaml.safe_load()` directly, bypassing `RunConfig` validation
that the CLI uses. Can silently run with missing/invalid fields.

**Resolution:** Use `RunConfig.model_validate()` like `cli.py` does.

**Files:** `scripts/run_once.py`
**Dependencies:** None.

---

## Phase B: MDP Foundation (research formalization)

### B.1 Define typed Research MDP primitives
**Paper ref:** AutoResearch-RL Section 2 -- Research MDP with State, Action, Reward, Transition.

**Problem:** States are implicit dicts. Actions are untyped. Rewards are computed ad-hoc in multiple
places. There is no formal MDP structure.

**Resolution:** Create `mdp.py` with:
- `State` dataclass: code snapshot hash, experiment history window, metric history, resource budget
- `Action` dataclass: either a `ParamProposal` or a `DiffProposal` (union type)
- `Reward` dataclass: scalar value + breakdown components
- `Transition` function signature: `(State, Action) -> (State, Reward)`

**Files:** New `mdp.py`, refactoring of `controller/`, `policy/`, `eval/scoring.py`
**Dependencies:** A.1, A.3

### B.2 Implement trajectory buffer
**Paper ref:** AutoResearch-RL Section 2 -- "trajectories tau = (s_0, a_0, r_0, ..., s_T)"

**Problem:** No episode storage. Policy updates process one sample at a time. No batching for
multi-epoch PPO updates.

**Resolution:** Create `trajectory.py` with a typed `Trajectory` buffer that stores
`(state, action, reward, next_state, log_prob, value_estimate)` tuples. Support batch retrieval
and episode windowing.

**Files:** New `trajectory.py`
**Dependencies:** B.1

### B.3 Implement GAE (Generalized Advantage Estimation)
**Paper ref:** AutoResearch-RL Section 2 -- GAE for variance-reduced advantage estimation.

**Problem:** Advantage in `learned.py` is `reward - running_mean`. No temporal structure, no
value function V(s), no lambda parameter.

**Resolution:** Implement `gae.py` with standard GAE computation:
`A_t = sum_{l=0}^{T} (gamma * lambda)^l * delta_{t+l}` where `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`.

**Files:** New `policy/gae.py`
**Dependencies:** B.1, B.2

---

## Phase C: PPO Policy Training

### C.1 Implement PPO policy network
**Paper ref:** AutoResearch-RL Section 2 -- PPO with clipped objective for edit generation.

**Problem:** `LearnedDiffPolicy` is a 5-weight linear scorer over hand-crafted features. This is
not a parameterized policy distribution. There is no actor network, no critic network, no action
distribution.

**Resolution:** Implement a proper PPO actor-critic:
- Actor: maps State -> action distribution (over param proposals or diff proposals)
- Critic: maps State -> scalar value estimate V(s)
- Loss: clipped surrogate + value loss + entropy bonus
- Multiple epochs of minibatch updates over collected trajectories

**Files:** `policy/learned.py` (rewrite), new `policy/ppo.py`
**Dependencies:** A.3, B.1, B.2, B.3

### C.2 Add entropy regularization and novelty bonus
**Paper ref:** AutoResearch-RL Section 5 -- entropy bonus H(pi) + novelty based on distance from
previously explored states.

**Problem:** No entropy term anywhere in the policy objective. No novelty computation.

**Resolution:** Add entropy bonus to PPO loss: `L = L^CLIP + c1 * L^VF - c2 * H(pi)`.
Add novelty bonus based on history dissimilarity.

**Files:** `policy/ppo.py` (from C.1)
**Dependencies:** C.1

---

## Phase D: Infrastructure Robustness

### D.1 Implement checkpoint management for loop state
**Paper ref:** AutoResearch-RL Section 3 -- "policy checkpointing to enable rollback."

**Problem:** If the process dies mid-run, all progress is lost. `best_value` is only in memory.
No save/restore for: episode number, best model state, telemetry refs, convergence history.

**Resolution:** Create `checkpoint.py` with save/restore for full loop state. Checkpoint after
each iteration. Support resume from checkpoint on startup.

**Files:** New `checkpoint.py`, unified loop controller, `config.py`, `cli.py`
**Dependencies:** A.1

### D.2 Implement graceful shutdown with signal handling
**Problem:** `while True` loop with no signal handling. SIGTERM/SIGINT corrupt mid-write files.

**Resolution:** Register SIGTERM/SIGINT handlers that set a `shutdown_requested` flag. Loop checks
flag between iterations. On shutdown: finish current iteration, persist checkpoint, flush telemetry.

**Files:** Unified loop controller, `cli.py`
**Dependencies:** A.1, D.1

### D.3 Add telemetry rotation
**Problem:** Append-only JSONL/TSV files grow unbounded in perpetual loops.

**Resolution:** Rotate telemetry files when they exceed a configurable size. Keep configurable
number of rotated files.

**Files:** `telemetry/events.py`, `telemetry/ledger.py`, `config.py`
**Dependencies:** A.1

### D.4 Add metrics aggregation and summary reporting
**Problem:** No episode-level summaries, no rolling averages, no trend detection across episodes.
The research requires "comprehensive telemetry with trend analysis" for the forecasting module.

**Resolution:** Create a metrics aggregation module that computes per-episode and rolling-window
statistics (mean, median, min, max, trend slope) over the telemetry stream.

**Files:** New module under `telemetry/`, unified loop controller
**Dependencies:** A.1, D.3

---

## Phase E: Evaluation and Scoring Alignment

### E.1 Implement real multi-judge diversity
**Paper ref:** AutoResearch-RL Section 4 -- "Multi-Judge Evaluation with diverse perspectives."

**Problem:** `eval/judge.py` calls `_heuristic_vote()` N times with identical inputs, producing
identical votes. This is not majority voting -- it's vote duplication.

**Resolution:** Implement multiple distinct judge strategies (heuristic, metric-threshold,
trend-based) that can disagree. `judge_next_state()` should aggregate truly diverse votes.

**Files:** `eval/judge.py`
**Dependencies:** None.

### E.2 Integrate early-stop forecaster into continuous loop
**Paper ref:** AutoResearch-RL Section 4 -- "power-law forecasting for early abort."

**Problem:** The power-law forecaster exists in `sandbox/runner.py` (`_fit_power_law`,
`_forecast_value`) and works in the legacy loop's trial runner. But the continuous loop
(`controller/continuous.py`) has no early-stop forecasting -- it always waits for target
commands to complete.

**Resolution:** Expose forecasting as a shared utility. Wire it into the continuous loop's
target execution (for command targets, monitor stdout stream and forecast).

**Files:** Extract from `sandbox/runner.py` into shared module, `controller/continuous.py`,
`target/command.py`
**Dependencies:** A.1

### E.3 Make scoring weights configurable
**Problem:** `eval/scoring.py` has hardcoded magic numbers for penalties and bonuses
(`fail_penalty=0.8`, `timeout_penalty=1.2`, `neutral_penalty=0.05`, `directional_bonus=0.2`).
No connection to SDPO adaptive weighting.

**Resolution:** Move weights into `ScoreWeights` config (already exists as dataclass but never
exposed in YAML config). Add SDPO-style adaptive alpha: `alpha_t = min(1, R_prev / R_target)`.

**Files:** `eval/scoring.py`, `config.py`
**Dependencies:** None.

---

## Phase F: Self-Distillation Pipeline (SDPO)

### F.1 Implement policy checkpointing for teacher snapshots
**Paper ref:** SDPO Section 3 -- "freeze pi_{t-1} as teacher at each iteration."

**Problem:** No mechanism to snapshot a policy version. `LearnedDiffPolicy` writes weights to a
single JSON file that is overwritten on each update.

**Resolution:** After each policy update, snapshot the current weights as a versioned checkpoint.
The previous checkpoint serves as teacher for the next SDPO update.

**Files:** `policy/learned.py` (or `policy/ppo.py` after C.1), `checkpoint.py`
**Dependencies:** C.1, D.1

### F.2 Implement SDPO loss function
**Paper ref:** SDPO -- `L_SDPO = L_RL + alpha_t * D_KL(pi_teacher || pi_student)`.

**Problem:** No SDPO loss. The `telemetry/distill.py` appends JSONL records but performs no
distillation computation.

**Resolution:** Implement SDPO loss that combines the PPO objective with a KL divergence term
between teacher (previous checkpoint) and student (current) policy distributions. Use adaptive
`alpha_t = min(1, R_prev / R_target)` weighting.

**Files:** New `policy/sdpo.py`, `telemetry/distill.py` (schema upgrade)
**Dependencies:** F.1, C.1

### F.3 Add distillation sample schema validation
**Paper ref:** SDPO Section 4 -- distillation records must include KL-divergence bounds and
preference pair data.

**Problem:** `telemetry/distill.py` is a bare JSONL appender with no schema validation.

**Resolution:** Define a Pydantic model for distillation samples. Validate on write.

**Files:** `telemetry/distill.py`, `config.py`
**Dependencies:** None.

---

## Phase G: SDFT Integration

### G.1 Implement SDFT token-level distillation module
**Paper ref:** SDFT -- softmax divergence fine-tuning with token-level teacher distribution matching.

**Problem:** No SDFT implementation. The directional branch (`hint` in judge output) is captured
but never used for distillation.

**Resolution:** Implement SDFT loss: `L_SDFT = sum_t softmax(z_teacher/T) * log(softmax(z_teacher/T) / softmax(z_student/T))`.
Start with top-K teacher logits only for cost control. Gate by confidence.

**Files:** New `distillation/sdft.py`
**Dependencies:** F.1 (needs teacher/student framework)

### G.2 Wire directional feedback into distillation sink
**Paper ref:** SDFT Section 3 -- "convert hints into teacher logits for distillation."

**Problem:** `eval/judge.py` produces `hint` strings that are logged but never consumed by any
training component.

**Resolution:** Create a distillation sink that consumes directional hints and teacher signals,
formats them for SDFT-style updates.

**Files:** `eval/judge.py`, `telemetry/distill.py`, new distillation sink module
**Dependencies:** G.1, F.2

---

## Phase H: Policy Promotion and Lifecycle

### H.1 Implement policy promotion gates
**Paper ref:** AutoResearch-RL Section 3 -- "commit/revert to best-known config."
**Paper ref:** SDPO Section 5 -- "gradual policy promotion with rollback checkpoints."

**Problem:** No promotion gate. Keep/discard decisions exist but only affect artifact versioning,
not the active policy.

**Resolution:** After N consecutive improvements (configurable), promote candidate policy to active.
On sustained degradation (detected by forecaster), rollback to last promoted checkpoint.

**Files:** Unified loop controller, `checkpoint.py`
**Dependencies:** D.1, F.1, E.2

### H.2 Implement experiment tracking interface
**Paper ref:** AutoResearch-RL Section 5 -- "replayability and auditability."

**Problem:** All experiment data goes to local files with no structured query, comparison, or
visualization capability.

**Resolution:** Create an abstract `ExperimentTracker` protocol with a local-file backend
(and optionally MLflow/W&B). Hook into episode lifecycle for: parameter logging, metric tracking,
artifact storage.

**Files:** New `tracking.py`, unified loop controller, `config.py`
**Dependencies:** A.1, D.3

---

## Dependency Graph

```
Phase A (cleanup):
  A.1 (unify loops) ----+----> A.3 (unify policies) ----> Phase B, C
  A.2 (fix seed)        |
  A.4 (fix version)     |
  A.5 (cache hw fp)     |
  A.6 (fix run_once)    +----> Phase D, E

Phase B (MDP):
  B.1 (MDP types) ----> B.2 (trajectory) ----> B.3 (GAE)

Phase C (PPO):           depends on A.3, B.1, B.2, B.3
  C.1 (PPO) ----> C.2 (entropy + novelty)

Phase D (infra):         depends on A.1
  D.1 (checkpoints) ----> D.2 (graceful shutdown)
  D.3 (telemetry rotation) ----> D.4 (metrics aggregation)

Phase E (eval):
  E.1 (diverse judges)    independent
  E.2 (forecaster)        depends on A.1
  E.3 (scoring config)    independent

Phase F (SDPO):           depends on C.1, D.1
  F.1 (teacher snapshots) ----> F.2 (SDPO loss)
  F.3 (distill schema)    independent

Phase G (SDFT):           depends on F.1
  G.1 (SDFT module) ----> G.2 (directional sink)

Phase H (lifecycle):      depends on D.1, F.1, E.2
  H.1 (promotion gates)
  H.2 (experiment tracking)  depends on A.1, D.3
```
