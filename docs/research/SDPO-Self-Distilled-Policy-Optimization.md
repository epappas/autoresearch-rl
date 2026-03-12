# SDPO — Self-Distilled Policy Optimization

- **Paper:** SDPO (arXiv:2601.20802)
- **URL:** https://arxiv.org/abs/2601.20802
- **Role in our stack:** stable self-distillation style update for directional refinement

## 1) Core idea
SDPO emphasizes improving policies via distillation-style objectives that are more stable than pure high-variance policy gradient updates in many settings. It uses preference/directional information to shape policy updates with tighter control over optimization behavior.

In practice this often appears as:
- student-vs-teacher policy comparisons,
- constrained or divergence-aware updates,
- better sample efficiency where reward-only signals are weak.

## 2) Objective intuition
Rather than relying solely on scalar returns, SDPO-like methods transform supervisory signal into structured preference/distribution targets and optimize toward them. This reduces update noise and can improve convergence quality.

## 3) Why this matters for AutoResearch-RL
Our loop now has explicit **next-state judging** and directional hints. SDPO-style lessons apply directly:
- treat directional feedback as first-class optimization signal,
- maintain controlled updates (avoid unstable swings from single trial outcomes),
- blend evaluative + directional pathways under clear weighting.

## 4) Mapping to scaffold modules
- `src/autoresearch_rl/controller/loop.py`  
  Produces event-driven samples with next-state context.
- `src/autoresearch_rl/eval/judge.py`  
  Generates evaluative and directional outputs suitable for SDPO-like update inputs.
- `src/autoresearch_rl/eval/scoring.py`  
  Currently approximates a weighted mixed objective; can evolve into true policy update weighting.

## 5) Practical implementation notes
1. Keep update channels explicit:
   - scalar channel (next-state score),
   - directional channel (hint/teacher guidance).
2. Add confidence gating and majority-vote reliability before consuming directional supervision.
3. Track per-sample type metrics (`scalar_only`, `directional_only`, `combined`).
4. Prefer gradual policy promotion with rollback checkpoints.

## 6) Risks
- Feedback contamination (incorrect directional hints causing drift).
- Distribution shift between judged sandbox traces and deployment traces.
- Hidden coupling between evaluative and directional channels if not logged separately.

## 7) Recommended usage in this repo
Use SDPO concepts to design the **policy update contract** once training is added:
- typed sample schema,
- weighted branch contribution,
- explicit trust controls on directional signals,
- promotion gating from candidate to active policy.
