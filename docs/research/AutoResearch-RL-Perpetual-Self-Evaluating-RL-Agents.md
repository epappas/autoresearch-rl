# AutoResearch-RL — Perpetual Self-Evaluating RL Agents

- **Paper:** AutoResearch-RL: *Perpetual Self-Evaluating Reinforcement Learning Agents for Autonomous Neural Architecture Discovery*
- **arXiv:** https://arxiv.org/abs/2603.07300
- **Role in our stack:** direct conceptual blueprint for autonomous code-edit research loops

## 1) Core thesis
The paper frames autonomous ML research as an RL problem where an agent repeatedly edits a training script, runs a bounded experiment, observes a scalar metric, and updates policy.

The key decomposition is:
1. **Frozen environment** (dataset/protocol/constants fixed for comparability)
2. **Mutable artifact** (`train.py` as editable state)
3. **Meta-policy** (agent learns edit strategy over time)

This maps almost 1:1 to what this repository is trying to become.

## 2) Formalization used in the paper
The paper defines a Research MDP with:
- **State:** current code + experiment history + diagnostics
- **Action:** structured code diff
- **Reward:** improvement in validation bpb (+ optional efficiency term)
- **Transition:** deterministic patch application + stochastic training outcome

It uses PPO with clipped objective and GAE for policy updates.

## 3) System architecture breakdown
The loop described is:
1. Generate code diff proposal
2. Apply patch atomically
3. Compile/sanity gate
4. Execute training under fixed wall-clock budget
5. Optionally early-stop bad runs
6. Evaluate metric (`val_bpb`)
7. Commit/revert to best-known config
8. Append trajectory and update policy

This is exactly the control-plane/data-plane split we’ve been adding to the scaffold.

## 4) Self-evaluation module (important practical contribution)
A notable engineering idea is online early-stop forecasting:
- Fit power-law loss curve during run
- Forecast final outcome at `T_max`
- Abort if predicted to underperform threshold

The paper reports substantial throughput gains from aborting bad runs early. Even if exact numbers vary in practice, the concept is high-value and directly usable.

## 5) Claimed theoretical results
The paper claims:
- monotone improvement in best-seen metric under assumptions,
- sample complexity bound tied to probability of finding an improvement,
- exploration-exploitation control via entropy regularization and novelty bonus.

Interpretation for us: these are useful guiding intuitions, but in real systems the assumptions (stationarity, independence, support coverage) are often partially violated.

## 6) Mapping to this repository
### Already aligned
- **Controller loop:** async proposal → trial → score pipeline
- **Next-state scoring:** delayed credit assignment in v0.2
- **Artifacts/telemetry:** replayable logs and run manifests
- **Example targets:** synthetic + DeBERTa projects

### Still missing for full parity
1. True PPO policy training for edit generation (currently baseline/stub policy)
2. Real patch-apply/rollback against target worktrees
3. Early-stop forecaster integrated into trial runner
4. Policy versioning and promotion/rollback gates
5. Reproducibility hardening for parallel long-running loops

## 7) Credibility / red-flag check
This paper has strong ideas but should be treated with caution until independently reproduced.

Potential red flags:
- Unusual authorship metadata in title block (includes “Claude AI” as author line in the manuscript text).
- Very strong empirical/theoretical claims with lightweight presentation depth.
- Some proof assumptions appear optimistic for real-world non-stationary code-edit search.

That does **not** make it useless; it means we should treat it as an architectural hypothesis + inspiration, and validate experimentally in-repo.

## 8) Practical lessons we should keep
1. Keep strict frozen-vs-mutable boundary for fair comparison.
2. Track best-known config and commit/revert semantics explicitly.
3. Add early-stop forecasting ASAP for throughput.
4. Separate scalar (evaluative) and directional feedback channels.
5. Optimize for replayability and auditability first; scale second.

## 9) Recommended next implementation steps
1. Add a patch-application layer with safe rollback (`git worktree`/temp copy).
2. Add early-stop curve forecaster to `sandbox/runner.py`.
3. Add policy training sink abstraction (`samples.jsonl` or queue) for future PPO learner.
4. Add benchmark report schema for long-run experiment comparability.

## 10) Bottom line
The paper is directionally very relevant to this repo and validates the current architectural trajectory. Use it as a **design reference**, not as unquestioned proof, and prioritize reproducible in-repo ablations.
