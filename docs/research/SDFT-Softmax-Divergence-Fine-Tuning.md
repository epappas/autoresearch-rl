# SDFT — Softmax Divergence Fine-Tuning

- **Paper:** SDFT (arXiv:2601.19897)
- **URL:** https://arxiv.org/abs/2601.19897
- **Role in our stack:** directional, token-level distillation signal (teacher distribution guidance)

## 1) Core idea
SDFT reframes post-training as **distribution matching** at the token level rather than only scalar reward optimization. Instead of learning from a single sampled token target, the student is nudged to match a richer teacher distribution signal over candidate tokens.

This helps when:
- reward is sparse/noisy,
- there are multiple acceptable continuations,
- directional supervision is more informative than binary outcomes.

## 2) Objective intuition
Given teacher and student token distributions, optimize a divergence objective that pushes student probabilities toward teacher-preferred mass assignment.

In practical implementations used in RL-adjacent loops:
- teacher provides top-K token log-probs at each step,
- student computes log-probs on those tokens,
- loss minimizes divergence over that reduced support (+ optional tail bucket).

## 3) Why this matters for AutoResearch-RL
Our scaffold now has:
- evaluative signal (`eval_score`), and
- directional signal (`hint`).

SDFT-style learning is the natural extension for the directional branch:
- convert hints / corrected-context generations into teacher logits,
- distill token-level guidance into candidate policy,
- combine with scalar objective for robustness.

## 4) Mapping to scaffold modules
- `src/autoresearch_rl/eval/judge.py`  
  Produces directional evidence (`hint`) and evaluative score.
- `src/autoresearch_rl/eval/scoring.py`  
  Composite score currently approximates this with scalar weighting.
- **Future trainer hook:** add a distillation sink that stores per-token teacher info for update steps.

## 5) Practical implementation notes
1. Start with top-K teacher logits only (K=20/50) for cost control.
2. Keep strict schema in traces (`teacher_topk_indices`, `teacher_topk_logprobs`).
3. Gate distillation by confidence to avoid propagating weak teacher guidance.
4. Keep a scalar fallback path (do not make distillation mandatory).

## 6) Risks
- Teacher quality bottleneck (bad teacher => bad student updates).
- Over-regularization toward teacher style.
- Added infra complexity (token-level data capture/storage).

## 7) Recommended usage in this repo
Use SDFT concepts for the **directional branch only** while preserving evaluative next-state scoring. Keep both branches separable so ablations are easy (`scalar_only`, `directional_only`, `combined`).
